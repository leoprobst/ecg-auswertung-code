from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import pandas as pd


def relation_columns(connection, relation_sql: str) -> set[str]:
    """Column names of a DuckDB relation, for optional-column selects."""

    return {
        str(row[0])
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM {relation_sql}"
        ).fetchall()
    }


def optional_column_select(
    available: set[str],
    name: str,
    sql_type: str,
    *,
    prefix: str = "",
) -> str:
    """Select ``name`` when the source has it, otherwise a typed NULL.

    Used for columns that a newer analysis contract requires but an artifact
    written by an older stage version may not carry. The downstream covariate
    builders treat NULL as "unavailable" and report it, so a stale artifact
    degrades into a disclosed missing covariate instead of a hard crash.
    """

    if name in available:
        return f"{prefix}{name}"
    return f"NULL AS {name}" if not sql_type else f"NULL::{sql_type} AS {name}"


def canonical_json(data: Any) -> str:
    return json.dumps(
        _strict_json_value(data),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=_strict_json_default,
        allow_nan=False,
    )


def content_hash(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def scientific_config_hash(config: Any) -> str:
    """Hash only result-defining project settings, never resource knobs."""

    from .config import scientific_config_payload

    return content_hash(scientific_config_payload(config))


def operational_config_hash(config: Any) -> str:
    """Hash resource knobs separately for execution provenance."""

    from .config import operational_config_payload

    return content_hash(operational_config_payload(config))


def file_sha256(path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(
    path: Path | str,
    include_sha256: Optional[bool] = None,
    *,
    mode: Optional[str] = None,
) -> dict[str, Any]:
    """Return a stable source fingerprint.

    ``metadata`` preserves the inexpensive legacy size/mtime behaviour while
    ``sha256`` adds a full content digest.  Callers can select a mode directly
    or set ``VCG_FILE_FINGERPRINT_MODE``.  An explicit ``include_sha256``
    remains supported and takes precedence for backwards compatibility.
    """

    source = Path(path)
    stat = source.stat()
    result: dict[str, Any] = {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    selected_mode = _fingerprint_mode(
        mode,
        include_sha256=include_sha256,
    )
    if selected_mode == "sha256":
        result["sha256"] = file_sha256(source)
    return result


def file_content_fingerprint(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    return {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "sha256": file_sha256(source),
    }


def optional_file_content_fingerprint(path: Path | str) -> dict[str, Any]:
    """Strongly identify a small upstream artifact, including absence.

    Result stages previously hashed only the *path* to a feature glob.  That
    lets a changed extraction, evidence build, or label build look identical
    to the cache whenever it is written back to the same directory.  Stage
    manifests are deliberately small JSON files, so hashing their complete
    content is cheap and makes that replacement visible.  A missing manifest
    is represented explicitly; should it appear later, the input hash changes
    and the result is recomputed.
    """

    source = Path(path)
    if not source.is_file():
        return {
            "path": str(source.resolve()),
            "status": "missing",
        }
    return file_content_fingerprint(source)


def input_artifact_fingerprints(
    artifacts: Mapping[str, Path | str],
) -> dict[str, dict[str, Any]]:
    """Return deterministic, named SHA-256 fingerprints for stage inputs."""

    return {
        str(name): optional_file_content_fingerprint(path)
        for name, path in sorted(artifacts.items(), key=lambda item: str(item[0]))
    }


def feature_manifest_path(feature_glob: Path | str) -> Path:
    """Resolve the manifest belonging to a scoped feature-partition glob.

    The canonical shape is ``<scope>/split=*/part-*.parquet``.  Keeping this
    derivation in one place prevents analysis stages from accidentally
    fingerprinting the all-development manifest while reading a 1k/10k pilot
    scope (or vice versa).
    """

    path = Path(str(feature_glob))
    if path.name.casefold() == "features.manifest.json":
        return path
    if path.parent.name.startswith("split="):
        return path.parent.parent / "features.manifest.json"
    return path.parent / "features.manifest.json"


def parquet_file_integrity(
    path: Path | str,
    *,
    include_sha256: bool = True,
) -> dict[str, Any]:
    """Fingerprint a Parquet file and prove that its footer is readable."""

    import pyarrow.parquet as pq

    source = Path(path)
    parquet = pq.ParquetFile(source)
    metadata = parquet.metadata
    fingerprint = file_fingerprint(
        source,
        include_sha256=include_sha256,
    )
    fingerprint.update(
        {
            "kind": "parquet",
            "row_count": int(metadata.num_rows),
            "row_group_count": int(metadata.num_row_groups),
            "column_count": int(metadata.num_columns),
            "schema_hash": content_hash(str(parquet.schema_arrow)),
        }
    )
    return fingerprint


def output_file_integrity(
    path: Path | str,
    *,
    include_sha256: bool = True,
) -> dict[str, Any]:
    """Capture strong integrity metadata for one stage output file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Stage output file is missing: {source}")
    if source.suffix.casefold() == ".parquet":
        return parquet_file_integrity(
            source,
            include_sha256=include_sha256,
        )
    fingerprint = file_fingerprint(
        source,
        include_sha256=include_sha256,
    )
    fingerprint["kind"] = "file"
    return fingerprint


def capture_output_integrity(
    paths: Iterable[Path | str],
    *,
    include_sha256: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Capture integrity records for a collection of file outputs."""

    return tuple(
        output_file_integrity(path, include_sha256=include_sha256)
        for path in paths
    )


def verify_output_integrity(
    records: Iterable[Mapping[str, Any]],
    *,
    mode: Optional[str] = None,
) -> bool:
    """Verify recorded outputs without trusting file existence alone.

    Modes are ``off``, ``metadata`` and ``sha256``.  The default is read from
    ``VCG_OUTPUT_INTEGRITY_MODE`` and is ``sha256``.  Parquet metadata is
    always opened in metadata/sha256 modes, which checks readability and the
    recorded row, row-group, column and schema counts without scanning rows.
    """

    selected_mode = _output_integrity_mode(mode)
    if selected_mode == "off":
        return all(Path(str(record.get("path", ""))).exists() for record in records)
    try:
        for record in records:
            path = Path(str(record.get("path", "")))
            if not path.is_file():
                return False
            current = output_file_integrity(
                path,
                include_sha256=selected_mode == "sha256",
            )
            for key in ("size", "kind"):
                if current.get(key) != record.get(key):
                    return False
            if record.get("kind") == "parquet":
                for key in (
                    "row_count",
                    "row_group_count",
                    "column_count",
                    "schema_hash",
                ):
                    if current.get(key) != record.get(key):
                        return False
            if selected_mode == "sha256":
                expected_sha = record.get("sha256")
                if not expected_sha or current.get("sha256") != expected_sha:
                    return False
    except (OSError, ValueError, RuntimeError):
        return False
    return True


def development_feature_glob(
    features_root: Path | str,
    panel_ids: Iterable[str],
    *,
    scope: str = "development",
) -> str:
    """The partitions of one panel set, and nothing else.

    Three stages defaulted to ``features/<panel>/**/*.parquet``. That
    recursive pattern matches 2,568 files where 2,563 are wanted. The five
    extra ones are what make it a data defect rather than a tidiness
    complaint:

    * ``feature_failure_report.parquet`` -- a four-column diagnostic table
      (``extraction_status, error_type, error_message, record_count``), and
      alphabetically the *first* match, so it decides the bound schema.
    * the ``smoke-development-1000`` partitions, i.e. the **pilot** cohort,
      pulled into a full-cohort analysis.

    ``union_by_name=true`` hid this: DuckDB merged every schema, the query
    bound, and the stage would have counted pilot recordings and failure-report
    rows as ECGs. Reading plainly turns the same defect into a binder error --
    which is how it was found, in a stage that had never run.

    The explicit pattern is what every stage that has run already uses.
    """

    return str(
        Path(features_root)
        / "_".join(str(item) for item in panel_ids)
        / str(scope)
        / "split=*"
        / "part-*.parquet"
    )


def parquet_columns(connection, sql_path: str) -> set[str]:
    """Column names of a parquet path or glob, without unifying schemas.

    ``union_by_name=true`` makes DuckDB open every file and merge every
    schema.  On the feature partitions -- 1,889 columns across 1,527 files --
    that was measured at 42 s and 3.6 GB, against stage profiles of 2 GB, and
    it paused a running study seconds after each resume.  Reading the schema
    plainly costs 0.07 s and no measurable memory, because the partitions of
    one extraction share a schema.

    A plain read does **not** raise when the schemas differ -- measured:
    DuckDB silently reports the first file's columns.  A first-file answer
    would then quietly drop markers, so the first and the last partition are
    both read and their columns combined.  That catches the realistic drift
    (a schema that changed partway through a run) at two schema reads instead
    of 1,527, and it is honest about what it does not catch: a column that
    appears only in the middle.

    Callers that need the *data* union-merged still pass ``union_by_name`` on
    their own query.  This is only for asking which columns exist.
    """

    import glob as _glob

    matches = sorted(_glob.glob(sql_path)) if any(
        ch in sql_path for ch in "*?["
    ) else [sql_path]
    probes = [sql_path] if not matches else sorted({matches[0], matches[-1]})

    columns: set[str] = set()
    for probe in probes:
        escaped = str(probe).replace("\\", "/").replace("'", "''")
        try:
            rows = connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{escaped}')"
            ).fetchall()
        except Exception:
            continue
        columns |= {str(row[0]) for row in rows}
    if columns:
        return columns

    try:
        rows = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{sql_path}', union_by_name=true)"
        ).fetchall()
    except Exception as exc:
        raise RuntimeError(
            f"Parquet-Schema unter {sql_path} ist nicht lesbar"
        ) from exc
    return {str(row[0]) for row in rows}


def feature_parquet_clause(
    connection=None,
    sql_path: str = "",
    *,
    options: Sequence[str] = (),
) -> str:
    """A ``read_parquet`` clause that only unifies schemas when it must.

    ``union_by_name=true`` makes DuckDB hold per-file schema state for every
    partition it opens. Measured on the real therapy-response query over 1,527
    feature partitions of 1,889 columns each:

    ==================  =========  ============  ==============
    variant             duration   peak RSS      free memory
    ==================  =========  ============  ==============
    union_by_name=true    109.3 s     19.36 GB         0.22 GB
    plain                   9.5 s      2.11 GB        18.01 GB
    ==================  =========  ============  ==============

    Identical results -- 3,357 rows either way. The stage ran with an 8 GB
    DuckDB limit and still reached 19.36 GB resident, because that state is
    not what the limit governs. A machine with 32 GB and a third already
    spoken for has nowhere to put the difference, which is what paused a
    month-long study over and over.

    The option is not decoration. ``broad_axis_stage`` records why it is
    there: partitions written at different times genuinely differ, because a
    marker added later exists only in the newer parts, and a plain read then
    silently takes the first file's schema -- measured, DuckDB does not
    raise. Dropping it on a hunch would trade a memory problem for markers
    that quietly go missing.

    So the answer is *measured*, not guessed: every partition's footer is
    read and the schemas compared. On this project that is 2,562 footers in
    41 seconds -- far too slow per query, which is why the result is cached
    both in the process and in a sidecar keyed by the partition count and the
    newest modification time. A new extraction changes both and invalidates
    it; nothing else does.
    """

    # ``connection`` is accepted for call-site symmetry and is not used:
    # the answer comes from the partition footers, not from a session.
    #
    # ``options`` carries anything the call site already passed -- today only
    # ``hive_partitioning=true``. It is preserved verbatim rather than judged:
    # ``split`` happens to be a physical column too, so the option is
    # redundant here, but dropping it would be a second, unmeasured change
    # riding along with a memory fix.
    extra = "".join(f", {item}" for item in options)
    return f"read_parquet('{sql_path}'{_uniformity_suffix(sql_path)}{extra})"


_UNIFORMITY_CACHE: dict[tuple, bool] = {}
_UNIFORMITY_SIDECAR = ".schema_uniformity.json"
_UNIFORMITY_SIDECAR_VERSION = 2  # 2: names and types compared


def _uniformity_suffix(sql_path: str) -> str:
    return "" if feature_partitions_share_a_schema(sql_path) else ", union_by_name=true"


# Listing 2,562 partitions and stat-ing each one costs 191 ms. The result
# cache below sits *behind* that work, so it never saved any of it -- and
# validation builds this clause once per endpoint and marker, which would have
# added 32 minutes of pure directory traffic to a stage that is already the
# longest in the run. The probe is therefore memoised on its own.
#
# The window is short on purpose. Only an extraction changes these files, it
# takes days, and every consumer stage runs after it; thirty seconds cannot
# span a write yet collapses a stage's worth of repeated calls into one.
_PROBE_TTL_SECONDS = 30.0
_PROBE_CACHE: dict[str, tuple[float, Optional[tuple[list[str], tuple]]]] = {}


def _partition_probe(
    sql_path: str,
) -> Optional[tuple[list[str], tuple]]:
    """The partition list and its cache key, or None if unusable."""

    now = time.monotonic()
    cached = _PROBE_CACHE.get(str(sql_path))
    if cached is not None and now - cached[0] < _PROBE_TTL_SECONDS:
        return cached[1]

    import glob as _glob

    result: Optional[tuple[list[str], tuple]] = None
    try:
        paths = sorted(_glob.glob(str(sql_path)))
    except Exception:
        paths = []
    if paths:
        try:
            newest = max(Path(item).stat().st_mtime_ns for item in paths)
        except OSError:
            newest = None
        if newest is not None:
            result = (paths, (str(sql_path), len(paths), newest))
    _PROBE_CACHE[str(sql_path)] = (now, result)
    return result


def feature_partitions_share_a_schema(sql_path: str) -> bool:
    """Whether every partition behind ``sql_path`` has the same columns.

    False on any doubt: an unreadable footer, no matches, a missing library.
    The expensive read is the safe one, so uncertainty must choose it.
    """

    probe = _partition_probe(sql_path)
    if probe is None:
        return False
    paths, key = probe
    cached = _UNIFORMITY_CACHE.get(key)
    if cached is not None:
        return cached

    sidecar = _uniformity_sidecar_path(sql_path, paths[0])
    stored = _read_uniformity_sidecar(sidecar, key)
    if stored is not None:
        _UNIFORMITY_CACHE[key] = stored
        return stored

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return False
    # Names *and* types: a column that is null in every row of one partition
    # is written with the ``null`` type there and ``double`` elsewhere, and a
    # plain read then fails to cast the second to the first. A092's
    # validate-pilot-10000 stopped on exactly that. The writer now casts such
    # columns to double, but partitions written before it did are still on
    # disk and must be read with schema unification.
    reference: Optional[tuple[str, ...]] = None
    uniform = True
    try:
        for item in paths:
            schema = pq.ParquetFile(item).schema_arrow
            names = tuple(f"{name}:{dtype}" for name, dtype in zip(schema.names, schema.types))
            if reference is None:
                reference = names
            elif names != reference:
                uniform = False
                break
    except Exception:
        uniform = False
    _UNIFORMITY_CACHE[key] = uniform
    _write_uniformity_sidecar(sidecar, key, uniform)
    return uniform


def _uniformity_sidecar_path(sql_path: str, first_match: str) -> Path:
    """Where the cached answer lives: the fixed part of the pattern.

    Taking the first match's grandparent happened to be right for
    ``.../development/split=*/part-*.parquet`` and wrong in general -- for a
    pattern without wildcards it writes one directory *above* the path the
    caller named. The stable root is the pattern itself, cut at its first
    wildcard, which never climbs higher than what was asked for.
    """

    text = str(sql_path).replace("\\", "/")
    cut = min(
        (text.index(ch) for ch in "*?[" if ch in text),
        default=len(text),
    )
    root = Path(text[:cut]).parent if cut < len(text) else Path(text).parent
    if not root.is_dir():
        root = Path(first_match).parent
    return root / _UNIFORMITY_SIDECAR


def _read_uniformity_sidecar(path: Path, key: tuple) -> Optional[bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        str(payload.get("glob")) != key[0]
        or int(payload.get("partition_count", -1)) != key[1]
        or int(payload.get("newest_mtime_ns", -1)) != key[2]
        or int(payload.get("version", 1)) != _UNIFORMITY_SIDECAR_VERSION
    ):
        return None
    value = payload.get("uniform")
    return bool(value) if isinstance(value, bool) else None


def _write_uniformity_sidecar(path: Path, key: tuple, uniform: bool) -> None:
    try:
        atomic_write_json(
            path,
            {
                "glob": key[0],
                "partition_count": key[1],
                "newest_mtime_ns": key[2],
                "version": _UNIFORMITY_SIDECAR_VERSION,
                "uniform": bool(uniform),
            },
        )
    except Exception:
        # A read-only artifact area costs the cache, not the run.
        return


def source_tree_hash(root: Path | str, patterns: tuple[str, ...] = ("*.py",)) -> str:
    base = Path(root)
    entries = []
    for pattern in patterns:
        for path in sorted(base.rglob(pattern)):
            if "__pycache__" in path.parts:
                continue
            entries.append(
                {
                    "relative_path": str(path.relative_to(base)).replace("\\", "/"),
                    "sha256": file_sha256(path),
                }
            )
    return content_hash(entries)


def atomic_write_json(path: Path | str, data: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(
            _strict_json_value(data),
            ensure_ascii=False,
            indent=2,
            default=_strict_json_default,
            allow_nan=False,
        ),
    )


def atomic_write_text(path: Path | str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def flush_to_disk(path: Path | str) -> None:
    """Force one written file out of the operating system's cache.

    Renaming is atomic, but atomicity is not durability: a rename can reach
    the disk while the bytes it points at are still in the page cache, so a
    power loss in that window leaves a correctly named file with truncated or
    absent content -- and unlike a half-written temporary file, that one is
    indistinguishable from a complete result. ``atomic_write_text`` has always
    flushed for this reason; the parquet path had not, and a study that runs
    for days is exactly the case where the window gets hit.

    A failure to flush is not raised. The file has been written; refusing to
    return because the platform would not confirm the flush would turn a
    durability weakness into a lost stage.
    """

    target = Path(path)
    try:
        handle = os.open(target, os.O_RDWR)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)
    # The directory entry itself is only guaranteed on POSIX, and only by
    # flushing the directory; Windows has no equivalent and refuses to open a
    # directory as a file, which the OSError above absorbs.
    if os.name == "posix":
        try:
            directory = os.open(target.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory)
        except OSError:
            pass
        finally:
            os.close(directory)


def atomic_write_parquet(
    frame: pd.DataFrame,
    path: Path | str,
    *,
    index: bool = False,
    compression: str = "zstd",
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        frame.to_parquet(temporary, index=index, compression=compression)
        flush_to_disk(temporary)
        atomic_replace(temporary, target)
        flush_to_disk(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_replace(
    source: Path | str,
    target: Path | str,
    *,
    attempts: int = 8,
    initial_delay_seconds: float = 0.05,
) -> None:
    delay = max(0.0, float(initial_delay_seconds))
    for attempt in range(max(1, int(attempts))):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 >= max(1, int(attempts)):
                raise
            time.sleep(delay)
            delay = min(0.5, max(0.01, delay * 2.0))


@dataclass(frozen=True)
class StageManifest:
    stage: str
    stage_version: str
    input_hash: str
    config_hash: str
    output_paths: tuple[str, ...]
    row_counts: dict[str, int]
    created_at: str
    status: str = "complete"
    notes: tuple[str, ...] = ()
    output_integrity: tuple[dict[str, Any], ...] = ()

    @classmethod
    def complete(
        cls,
        *,
        stage: str,
        stage_version: str,
        input_hash: str,
        config_hash: str,
        output_paths: tuple[str, ...],
        row_counts: Optional[dict[str, int]] = None,
        notes: tuple[str, ...] = (),
        output_integrity: tuple[dict[str, Any], ...] = (),
        status: str = "complete",
    ) -> "StageManifest":
        # ``status`` is deliberately settable: a stage whose declared source
        # was absent still writes its (empty) outputs, but must not claim
        # "complete" -- matches() only reuses a cached manifest whose status
        # is exactly "complete", so a source that appears later re-triggers
        # the stage instead of being masked by the empty artifact.
        return cls(
            stage=stage,
            stage_version=stage_version,
            input_hash=input_hash,
            config_hash=config_hash,
            output_paths=output_paths,
            row_counts=row_counts or {},
            created_at=datetime.now(timezone.utc).isoformat(),
            status=status,
            notes=notes,
            output_integrity=output_integrity,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> None:
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path | str) -> "StageManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["output_paths"] = tuple(data.get("output_paths", ()))
        data["notes"] = tuple(data.get("notes", ()))
        data["output_integrity"] = tuple(data.get("output_integrity", ()))
        return cls(**data)

    def reuse_decision(
        self, input_hash: str, config_hash: str, stage_version: str
    ) -> tuple[bool, str]:
        """Whether this artifact may be reused, and *why*.

        The boolean alone was the reason a walk-back over eleven stages could
        not be explained afterwards: it says a stage recomputed, never which
        of the four keys moved.  Returning the sentence costs nothing and is
        the difference between "the run restarted itself" and "the criteria
        module changed, so criteria-dependent stages recomputed".
        """

        if self.status != "complete":
            return False, f"vorheriger Lauf unvollstaendig (status={self.status})"
        if self.stage_version != stage_version:
            return (
                False,
                f"Stufenversion {self.stage_version} -> {stage_version}",
            )
        if self.input_hash != input_hash:
            return False, "Eingangsdaten geaendert (input_hash)"
        if self.config_hash != config_hash:
            return False, "wissenschaftliche Konfiguration geaendert (config_hash)"
        for path in self.output_paths:
            if not Path(path).exists():
                return False, f"Ausgabe fehlt: {Path(path).name}"
        if not self.output_integrity:
            # Legacy manifests did not record content integrity.  Keep them
            # readable; newly produced manifests can opt into strong checks.
            return True, "Schluessel identisch (Manifest ohne Integritaetsdaten)"
        recorded_paths = {
            str(Path(str(record.get("path", ""))).resolve())
            for record in self.output_integrity
        }
        file_outputs = {
            str(Path(path).resolve())
            for path in self.output_paths
            if Path(path).is_file()
        }
        if not file_outputs.issubset(recorded_paths):
            return False, "Ausgabedateien nicht vollstaendig protokolliert"
        if not verify_output_integrity(self.output_integrity):
            return False, "Ausgabedatei veraendert oder unlesbar"
        return True, "Schluessel identisch"

    def matches(self, input_hash: str, config_hash: str, stage_version: str) -> bool:
        reused, reason = self.reuse_decision(
            input_hash, config_hash, stage_version
        )
        # Reported rather than returned, so the twenty call sites keep their
        # ``if previous.matches(...)`` shape and still leave a trace.
        try:
            from .run_log import note_reuse

            note_reuse(self.stage, reused=reused, reason=reason)
        except Exception:
            pass
        return reused


def _own_grant_workers() -> Optional[int]:
    """Workers of a live grant owned by this process, else ``None``."""
    grant_path = os.environ.get("VCG_RESOURCE_GRANT_FILE")
    if not grant_path:
        return None
    from .resource_grants import read_grant

    grant = read_grant(Path(grant_path))
    if not grant or grant.get("owner_pid") != os.getpid():
        return None
    return int(grant["workers"])


@contextmanager
def isolated_analysis_connection(config) -> Iterator[Any]:
    """Use the same runtime caps and owned scratch space as catalog queries."""
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        yield connection


class DuckDBCatalog:
    def __init__(self, path: Path | str, memory_limit_gb: int = 20, threads: int = 8,
                 *, spill_limit_gb: float | None = None, follow_grant: bool = False):
        self.path = Path(path)
        # Opt-in (audit R12): a connection opened by a supervised job uses the
        # live grant's worker count instead of the fixed ``config.workers``
        # (7 on this machine, from a synthetic benchmark). Callers that cap
        # threads on purpose keep ``follow_grant=False``.
        self.follow_grant = bool(follow_grant)
        runtime_memory = _positive_environment_number(
            "VCG_JOB_MEMORY_GB"
        )
        runtime_threads = _positive_environment_number(
            "VCG_JOB_CPU_TOKENS"
        )
        self.memory_limit_gb = min(
            float(memory_limit_gb),
            runtime_memory if runtime_memory is not None else float(memory_limit_gb),
        )
        self.threads = min(
            int(threads),
            int(runtime_threads) if runtime_threads is not None else int(threads),
        )
        # Per-connection hard cap, not a claim of a global concurrent budget.
        # The scheduler/capacity proof must account for every live connection.
        self.spill_limit_gb = (spill_limit_gb if spill_limit_gb is not None else
                               _positive_environment_number("VCG_JOB_SPILL_GB") or 16.0)

    def effective_threads(self) -> int:
        """Grant-following thread count, evaluated per connection."""
        if self.follow_grant:
            granted = _own_grant_workers()
            if granted is not None:
                return max(1, granted)
        return max(1, int(self.threads))

    def effective_memory_limit_gb(self) -> float:
        """Memory limit for the next query block (audit R8).

        Memory pressure used to know two steps only -- one worker fewer, or
        the whole job paused and its running work discarded. A grant-following
        connection of a supervised job now opens with the limit the machine
        can actually give (commit-aware, minus the reserve and 1 GB for
        Python), so DuckDB spills instead of failing. Never above the
        configured limit; never below 2 GB, where operators that cannot spill
        would fail instead.
        """
        configured = float(self.memory_limit_gb)
        if not self.follow_grant or not os.environ.get("VCG_RESOURCE_GRANT_FILE"):
            return configured
        try:
            from .resources import available_memory_gb

            available = available_memory_gb()
        except Exception:
            return configured
        if available is None:
            return configured
        reserve = _positive_environment_number("VCG_JOB_OS_RESERVE_GB") or 2.0
        room = float(available) - float(reserve) - 1.0
        return max(2.0, min(configured, room))

    @contextmanager
    def connect(
        self,
        read_only: bool = False,
        *,
        isolated: bool = False,
    ) -> Iterator[Any]:
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError(
                "DuckDB is required for the MIMIC workflow. Install requirements with "
                "'python -m pip install -r vcg_biomarker/requirements.txt'."
            ) from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_context = (
            _catalog_writer_lock(self.path.with_suffix(f"{self.path.suffix}.lock"))
            if not read_only and not isolated
            else nullcontext()
        )
        # An in-memory DuckDB otherwise spills to `.tmp` below the caller's
        # working directory (potentially OneDrive or a different disk budget).
        # Each isolated connection owns a unique directory; close the DB before
        # TemporaryDirectory removes only that connection's scratch files.
        scratch_context = (
            tempfile.TemporaryDirectory(
                prefix="duckdb-isolated-", dir=self.path.resolve().parent,
            )
            if isolated else nullcontext(None)
        )
        from .spill_budget import reserve_spill
        with lock_context, reserve_spill(
            self.path.resolve().parent, self.spill_limit_gb,
        ), scratch_context as scratch_directory:
            connection = duckdb.connect(
                ":memory:" if isolated else str(self.path),
                read_only=read_only and not isolated,
            )
            try:
                if scratch_directory is not None:
                    scratch_sql = Path(scratch_directory).as_posix().replace("'", "''")
                    connection.execute(f"SET temp_directory='{scratch_sql}'")
                connection.execute(
                    f"SET memory_limit='{max(1, int(self.effective_memory_limit_gb()))}GB'"
                )
                connection.execute(f"SET threads={self.effective_threads()}")
                connection.execute(
                    f"SET max_temp_directory_size='{self.spill_limit_gb:g}GiB'"
                )
                if not read_only and not isolated:
                    self._ensure_metadata(connection)
                yield connection
            finally:
                connection.close()
    @staticmethod
    def _ensure_metadata(connection: Any) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vcg_stage_registry (
                stage VARCHAR PRIMARY KEY,
                stage_version VARCHAR NOT NULL,
                input_hash VARCHAR NOT NULL,
                config_hash VARCHAR NOT NULL,
                manifest_path VARCHAR NOT NULL,
                completed_at TIMESTAMP NOT NULL
            )
            """
        )

    def register_stage(self, manifest: StageManifest, manifest_path: Path | str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO vcg_stage_registry
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    manifest.stage,
                    manifest.stage_version,
                    manifest.input_hash,
                    manifest.config_hash,
                    str(Path(manifest_path).resolve()),
                    manifest.created_at,
                ],
            )

    def create_parquet_view(self, name: str, parquet_glob: Path | str) -> None:
        safe_name = _safe_identifier(name)
        path = str(parquet_glob).replace("\\", "/").replace("'", "''")
        with self.connect() as connection:
            connection.execute(
                f"CREATE OR REPLACE VIEW {safe_name} AS "
                f"SELECT * FROM read_parquet('{path}', union_by_name=true)"
            )


def _positive_environment_number(name: str) -> Optional[float]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) and number > 0 else None


def _fingerprint_mode(
    mode: Optional[str],
    *,
    include_sha256: Optional[bool],
) -> str:
    if include_sha256 is not None:
        return "sha256" if include_sha256 else "metadata"
    selected = str(
        mode
        if mode is not None
        else os.environ.get("VCG_FILE_FINGERPRINT_MODE", "metadata")
    ).strip().casefold()
    aliases = {
        "content": "sha256",
        "full": "sha256",
        "stat": "metadata",
        "legacy": "metadata",
    }
    selected = aliases.get(selected, selected)
    if selected not in {"metadata", "sha256"}:
        raise ValueError(
            "Fingerprint mode must be 'metadata' or 'sha256', got "
            f"{selected!r}"
        )
    return selected


def _output_integrity_mode(mode: Optional[str]) -> str:
    selected = str(
        mode
        if mode is not None
        else os.environ.get("VCG_OUTPUT_INTEGRITY_MODE", "sha256")
    ).strip().casefold()
    aliases = {
        "content": "sha256",
        "full": "sha256",
        "stat": "metadata",
        "existence": "off",
        "none": "off",
    }
    selected = aliases.get(selected, selected)
    if selected not in {"off", "metadata", "sha256"}:
        raise ValueError(
            "Output integrity mode must be 'off', 'metadata' or 'sha256', got "
            f"{selected!r}"
        )
    return selected


def _safe_identifier(value: str) -> str:
    if not value or not value.replace("_", "").isalnum() or value[0].isdigit():
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return value


@contextmanager
def _catalog_writer_lock(
    path: Path,
    *,
    timeout_seconds: float = 300.0,
) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    handle: Optional[int] = None
    while handle is None:
        try:
            handle = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            if _remove_stale_catalog_lock(path):
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for DuckDB catalog writer lock: "
                    f"{path}"
                )
            time.sleep(0.1)
    try:
        payload = canonical_json(
            {
                "pid": os.getpid(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ).encode("utf-8")
        os.write(handle, payload)
        os.fsync(handle)
        yield
    finally:
        os.close(handle)
        # Windows briefly denies deletion while a waiting writer reads the
        # ownership file. Do not strand the lock after releasing our handle.
        unlink_deadline = time.monotonic() + 2.0
        while True:
            try:
                path.unlink()
                break
            except FileNotFoundError:
                break
            except PermissionError:
                if time.monotonic() >= unlink_deadline:
                    raise
                time.sleep(0.01)


def _remove_stale_catalog_lock(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        try:
            stale_unreadable = time.time() - path.stat().st_mtime > 30.0
        except OSError:
            return True
        if stale_unreadable:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return True
        return False
    try:
        import psutil

        alive = pid > 0 and psutil.pid_exists(pid)
    except ImportError:
        alive = pid == os.getpid()
    if alive:
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return True


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _strict_json_default(value: Any) -> Any:
    return _strict_json_value(_json_default(value))


def _strict_json_value(value: Any) -> Any:
    """Return standards-compliant JSON values without NaN/Infinity tokens."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {
            str(key): _strict_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    return value
