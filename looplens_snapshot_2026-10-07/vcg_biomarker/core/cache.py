from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_HAS_PARQUET = False
try:
    import pyarrow  # noqa: F401
    _HAS_PARQUET = True
except ImportError:
    try:
        import fastparquet  # noqa: F401
        _HAS_PARQUET = True
    except ImportError:
        pass


class ParquetEngineUnavailableError(RuntimeError):
    """Raised when parquet cache files exist but no parquet reader is installed."""


def _parquet_engine_error(path: Path) -> ParquetEngineUnavailableError:
    return ParquetEngineUnavailableError(
        f"Parquet cache file '{path}' exists, but no parquet engine is installed. "
        "Install pyarrow or fastparquet, or rebuild the cache as CSV."
    )


class ResultCache:
    """
    Streaming-fähiger Ergebnis-Cache für große Datensätze.

    Schreibstrategie:
      - Während der Verarbeitung werden Batches in einzelne Chunk-Dateien
        unter chunks/ geschrieben (O(batch) pro Schreibvorgang, kein Rewrite).
      - Nach der Verarbeitung werden alle Chunks in batch_results.parquet
        konsolidiert (consolidate()).
      - load_all() liest batch_results, fällt aber auf einzelne Chunks zurück.
      - get_analysis_arrays() lädt nur die für die Analyse benötigten Spalten.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = cache_dir
        self._cached_ids: set[int] = set()
        self._config_hash: Optional[str] = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_index()

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    @property
    def cache_dir(self) -> Optional[Path]:
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value: Path):
        old_dir = self._cache_dir
        old_ids = set(self._cached_ids)
        self._cache_dir = value
        try:
            if value:
                value.mkdir(parents=True, exist_ok=True)
                self._load_index()
        except Exception:
            self._cache_dir = old_dir
            self._cached_ids = old_ids
            raise

    # ------------------------------------------------------------------
    # Config hash
    # ------------------------------------------------------------------

    def set_config_hash(self, config_dict: dict):
        raw = json.dumps(config_dict, sort_keys=True, default=str)
        self._config_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def check_config_valid(self) -> bool:
        if self._config_hash is None or self._cache_dir is None:
            return True
        hash_path = self._cache_dir / "config_hash.txt"
        if not hash_path.exists():
            return True
        return hash_path.read_text().strip() == self._config_hash

    def save_config_hash(self):
        if self._config_hash is not None and self._cache_dir is not None:
            (self._cache_dir / "config_hash.txt").write_text(self._config_hash)

    # ------------------------------------------------------------------
    # Low-level I/O helpers
    # ------------------------------------------------------------------

    def _atomic_write_text(self, path: Path, text: str):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def _atomic_pickle(self, path: Path, obj) -> bool:
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
            return True
        except Exception:
            try:
                tmp.unlink()
            except Exception:
                pass
            return False

    def _save_df(self, df: pd.DataFrame, path: Path):
        if _HAS_PARQUET:
            final_path = path.with_suffix(".parquet")
            tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
            df.to_parquet(tmp_path, index=False)
            tmp_path.replace(final_path)
        else:
            final_path = path.with_suffix(".csv")
            tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
            df.to_csv(tmp_path, index=False)
            tmp_path.replace(final_path)

    def _load_df(self, base_name: str) -> Optional[pd.DataFrame]:
        if self._cache_dir is None:
            return None
        parquet_path = self._cache_dir / f"{base_name}.parquet"
        csv_path = self._cache_dir / f"{base_name}.csv"
        if parquet_path.exists():
            if _HAS_PARQUET:
                try:
                    return pd.read_parquet(parquet_path)
                except Exception as exc:
                    if not csv_path.exists():
                        raise RuntimeError(f"Could not read parquet cache file '{parquet_path}': {exc}") from exc
            elif not csv_path.exists():
                raise _parquet_engine_error(parquet_path)
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path)
            except Exception:
                pass
        return None

    def _chunks_dir(self) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / "chunks"

    def _iter_chunk_files(self):
        """Yield all existing chunk files sorted by name."""
        d = self._chunks_dir()
        if d is None or not d.exists():
            return
        if _HAS_PARQUET:
            files = sorted(d.glob("chunk_*.parquet")) + sorted(d.glob("chunk_*.csv"))
        else:
            files = sorted(d.glob("chunk_*.csv")) + sorted(d.glob("chunk_*.parquet"))
        seen = set()
        for f in sorted(files, key=lambda x: x.stem):
            if f.stem not in seen:
                seen.add(f.stem)
                yield f

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_index(self):
        if self._cache_dir is None:
            return

        ids: set[int] = set()

        # Cache checkpoints may contain a saved index plus newer chunk files
        # from a run that stopped before consolidation.
        df = self._load_df("cache_index")
        if df is not None:
            ids.update(int(x) for x in df["ecg_id"].tolist())

        # Never decode the complete (potentially multi-GB) result matrix just
        # to rebuild its one-column membership index.
        batch_parquet = self._cache_dir / "batch_results.parquet"
        batch_csv = self._cache_dir / "batch_results.csv"
        compact_index_covers_batch = False
        if df is not None and batch_parquet.exists() and _HAS_PARQUET:
            try:
                import pyarrow.parquet as pq

                compact_index_covers_batch = (
                    int(pq.ParquetFile(batch_parquet).metadata.num_rows)
                    == len(ids)
                )
            except Exception:
                compact_index_covers_batch = False
        try:
            if compact_index_covers_batch:
                batch_ids = ()
            elif batch_parquet.exists() and _HAS_PARQUET:
                batch_ids = pd.read_parquet(
                    batch_parquet,
                    columns=["ecg_id"],
                )["ecg_id"]
            elif batch_csv.exists():
                batch_ids = pd.read_csv(
                    batch_csv,
                    usecols=["ecg_id"],
                )["ecg_id"]
            else:
                batch_ids = ()
            ids.update(int(x) for x in batch_ids)
        except Exception:
            pass

        for f in self._iter_chunk_files():
            try:
                if _HAS_PARQUET and f.suffix == ".parquet":
                    chunk_ids = pd.read_parquet(f, columns=["ecg_id"])["ecg_id"].tolist()
                else:
                    chunk_ids = pd.read_csv(f, usecols=["ecg_id"])["ecg_id"].tolist()
                ids.update(int(x) for x in chunk_ids)
            except Exception:
                pass
        self._cached_ids = ids

    def _save_index(self):
        if self._cache_dir is None or not self._cached_ids:
            return
        df = pd.DataFrame({"ecg_id": sorted(self._cached_ids)})
        self._save_df(df, self._cache_dir / "cache_index")

    def is_cached(self, ecg_id: int) -> bool:
        return ecg_id in self._cached_ids

    def get_uncached_ids(self, ecg_ids: list[int]) -> list[int]:
        return [eid for eid in ecg_ids if eid not in self._cached_ids]

    def get_n_cached(self) -> int:
        return len(self._cached_ids)

    # ------------------------------------------------------------------
    # Streaming chunk write (O(chunk) per write, no rewrite of existing data)
    # ------------------------------------------------------------------

    def get_next_chunk_id(self) -> int:
        """Return the next available chunk sequence number."""
        d = self._chunks_dir()
        if d is None or not d.exists():
            return 0
        ids = []
        for f in d.glob("chunk_*"):
            try:
                ids.append(int(f.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
        return max(ids) + 1 if ids else 0

    def store_chunk(self, rows: list[dict], chunk_id: int):
        """Write result rows to an individual chunk file — O(len(rows)), no global rewrite."""
        if self._cache_dir is None or not rows:
            return
        d = self._chunks_dir()
        d.mkdir(exist_ok=True)
        df = pd.DataFrame(rows)
        self._save_df(df, d / f"chunk_{chunk_id:08d}")
        for row in rows:
            self._cached_ids.add(int(row["ecg_id"]))
        # The chunk itself is the crash-safe incremental index.  Rewriting a
        # monotonically growing cache_index after every chunk was O(n^2).
        # A compact index is materialized once by consolidate().

    def consolidate(self):
        """
        Merge all chunk files into a single batch_results file and delete the chunks.
        Called once after all batches are processed.
        """
        if self._cache_dir is None:
            return
        chunk_files = list(self._iter_chunk_files())
        if not chunk_files:
            return

        existing_parquet = self._cache_dir / "batch_results.parquet"
        parquet_sources = (
            ([existing_parquet] if existing_parquet.exists() else [])
            + [path for path in chunk_files if path.suffix == ".parquet"]
        )
        if (
            parquet_sources
            and len(parquet_sources)
            == len(chunk_files) + int(existing_parquet.exists())
            and self._consolidate_parquet_streaming(parquet_sources)
        ):
            self._remove_consolidated_chunks(chunk_files)
            self._save_index()
            return

        dfs = []
        existing = self._load_df("batch_results")
        if existing is not None:
            dfs.append(existing)

        for f in chunk_files:
            try:
                if _HAS_PARQUET and f.suffix == ".parquet":
                    dfs.append(pd.read_parquet(f))
                else:
                    dfs.append(pd.read_csv(f))
            except Exception:
                pass

        if not dfs:
            return

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["ecg_id"], keep="last")
        self._save_df(combined, self._cache_dir / "batch_results")

        # Delete chunk files after successful consolidation
        self._remove_consolidated_chunks(chunk_files)
        self._save_index()

    def _consolidate_parquet_streaming(self, sources: list[Path]) -> bool:
        """Merge Parquet sources with DuckDB spill instead of pandas concat."""
        if self._cache_dir is None:
            return False
        try:
            import duckdb
        except ImportError:
            return False

        def _sql(value: Path) -> str:
            return str(value.resolve()).replace("\\", "/").replace("'", "''")

        target = self._cache_dir / "batch_results.parquet"
        temporary = self._cache_dir / ".batch_results.parquet.tmp"
        source_sql = ", ".join(f"'{_sql(path)}'" for path in sources)
        ordering_sql = "CASE filename " + " ".join(
            f"WHEN '{_sql(path)}' THEN {index}"
            for index, path in enumerate(sources)
        ) + " ELSE -1 END"
        try:
            import pyarrow.parquet as pq

            source_row_count = sum(
                int(pq.ParquetFile(path).metadata.num_rows)
                for path in sources
            )
        except Exception:
            source_row_count = -1
        keys_are_already_unique = (
            source_row_count >= 0
            and source_row_count == len(self._cached_ids)
        )
        connection = None
        try:
            if temporary.exists():
                temporary.unlink()
            connection = duckdb.connect()
            connection.execute(
                f"SET threads={max(1, min(8, int(os.cpu_count() or 1)))}"
            )
            try:
                import psutil

                memory_mb = max(
                    512,
                    min(
                        4096,
                        int(psutil.virtual_memory().available // (2 * 1024**2)),
                    ),
                )
                connection.execute(f"SET memory_limit='{memory_mb}MB'")
            except (ImportError, AttributeError, OSError, ValueError):
                pass
            temp_dir = self._cache_dir / "_duckdb_tmp"
            temp_dir.mkdir(exist_ok=True)
            connection.execute(
                f"SET temp_directory='{_sql(temp_dir)}'"
            )
            connection.execute("SET preserve_insertion_order=FALSE")
            if keys_are_already_unique:
                merge_sql = (
                    "SELECT * EXCLUDE (filename) "
                    f"FROM read_parquet([{source_sql}], "
                    "union_by_name=true, filename=true) ORDER BY ecg_id"
                )
            else:
                merge_sql = (
                    "SELECT * EXCLUDE (filename, __source_order, "
                    "__row_number) FROM ("
                    "SELECT *, row_number() OVER ("
                    "PARTITION BY ecg_id ORDER BY __source_order DESC"
                    ") AS __row_number FROM ("
                    f"SELECT *, {ordering_sql} AS __source_order "
                    f"FROM read_parquet([{source_sql}], "
                    "union_by_name=true, filename=true)"
                    ")) WHERE __row_number = 1 ORDER BY ecg_id"
                )
            connection.execute(
                f"COPY ({merge_sql}) TO '{_sql(temporary)}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
            )
            row_count, unique_count = connection.execute(
                "SELECT count(*), count(DISTINCT ecg_id) "
                f"FROM read_parquet('{_sql(temporary)}')"
            ).fetchone()
            if int(row_count) == 0 or int(row_count) != int(unique_count):
                raise RuntimeError("Streaming cache consolidation lost uniqueness")
            connection.close()
            connection = None
            temporary.replace(target)
            return True
        except Exception as exc:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise RuntimeError(
                "Streaming cache consolidation failed; chunks were kept "
                "intact and no in-memory fallback was attempted"
            ) from exc

    def _remove_consolidated_chunks(self, chunk_files: list[Path]) -> None:
        for f in chunk_files:
            try:
                f.unlink()
            except Exception:
                pass
        d = self._chunks_dir()
        if d and d.exists():
            try:
                d.rmdir()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Batch write (legacy — still used by single-threaded Pipeline)
    # ------------------------------------------------------------------

    def store(
        self,
        ecg_id: int,
        biomarker_dict: dict,
        pathology_labels: dict,
        quality_score: float = 0.0,
        age=None,
        sex=None,
    ):
        """Compatibility helper for storing a single processed row."""
        if self._cache_dir is None:
            return

        row = {
            "ecg_id": ecg_id,
            "quality_score": quality_score,
        }
        if age is not None:
            row["age"] = age
        if sex is not None:
            row["sex"] = sex
        for k, v in biomarker_dict.items():
            row[f"bm_{k}"] = v
        for k, v in pathology_labels.items():
            row[f"path_{k}"] = 1 if v else 0

        existing = self._load_df("batch_results")
        if existing is not None and "ecg_id" in existing.columns:
            existing = existing[existing["ecg_id"] != ecg_id]
            df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        self._save_df(df, self._cache_dir / "batch_results")
        self._cached_ids.add(int(ecg_id))
        self._save_index()

    def store_batch(self, records: list):
        from .pipeline import ProcessedRecord  # noqa: F401

        if self._cache_dir is None:
            return

        rows = []
        for pr in records:
            if pr.error is not None:
                continue
            row = {
                "ecg_id": pr.ecg_record.ecg_id,
                "quality_score": pr.delineation.quality_score if pr.delineation else 0,
            }
            if pr.ecg_record.age is not None:
                row["age"] = pr.ecg_record.age
            if pr.ecg_record.sex is not None:
                row["sex"] = pr.ecg_record.sex
            for k, v in pr.biomarker_dict.items():
                row[f"bm_{k}"] = v
            for k, v in pr.pathology_labels.items():
                row[f"path_{k}"] = 1 if v else 0
            rows.append(row)
            self._cached_ids.add(pr.ecg_record.ecg_id)

        if not rows:
            return

        df = pd.DataFrame(rows)
        existing = self._load_df("batch_results")
        if existing is not None:
            existing_ids = set(existing["ecg_id"].tolist())
            new_rows = [r for r in rows if r["ecg_id"] not in existing_ids]
            if new_rows:
                df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            else:
                df = existing
        self._save_df(df, self._cache_dir / "batch_results")
        self._save_index()

    # ------------------------------------------------------------------
    # Read — consolidated + chunk fallback
    # ------------------------------------------------------------------

    def load_all(self) -> Optional[pd.DataFrame]:
        """Load all cached results as a DataFrame (batch_results first, then chunks)."""
        df = self._load_df("batch_results")
        if df is not None:
            return df
        # Fall back to scanning chunk files
        dfs = []
        for f in self._iter_chunk_files():
            try:
                if _HAS_PARQUET and f.suffix == ".parquet":
                    dfs.append(pd.read_parquet(f))
                else:
                    dfs.append(pd.read_csv(f))
            except Exception:
                pass
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return None

    def load_columns(self, columns: list[str]) -> Optional[pd.DataFrame]:
        """Load selected cache columns from consolidated results or chunks."""
        if self._cache_dir is None:
            return None

        wanted = list(dict.fromkeys(columns))
        if not wanted:
            return pd.DataFrame()

        def _read_file(path: Path) -> Optional[pd.DataFrame]:
            try:
                if _HAS_PARQUET and path.suffix == ".parquet":
                    import pyarrow.parquet as pq
                    schema = pq.read_schema(path)
                    available = [c for c in wanted if c in schema.names]
                    if not available:
                        return None
                    return pd.read_parquet(path, columns=available)
                if path.suffix == ".csv":
                    header = pd.read_csv(path, nrows=0)
                    available = [c for c in wanted if c in header.columns]
                    if not available:
                        return None
                    return pd.read_csv(path, usecols=available)
            except Exception:
                return None
            return None

        parquet_path = self._cache_dir / "batch_results.parquet"
        csv_path = self._cache_dir / "batch_results.csv"
        if parquet_path.exists() and not _HAS_PARQUET and not csv_path.exists():
            raise _parquet_engine_error(parquet_path)

        for path in (parquet_path, csv_path):
            if path.exists():
                df = _read_file(path)
                if df is not None:
                    return df

        dfs = []
        for f in self._iter_chunk_files():
            df = _read_file(f)
            if df is not None:
                dfs.append(df)
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            if "ecg_id" in combined.columns:
                combined = combined.drop_duplicates(subset=["ecg_id"], keep="last")
            return combined
        return None

    def get_analysis_arrays(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Load only biomarker and pathology columns as numpy arrays.
        Much more memory-efficient than load_all() for analysis purposes.
        With parquet, only the requested columns are read from disk.
        """
        biomarker_parts: dict[str, list[np.ndarray]] = {}
        pathology_parts: dict[str, list[np.ndarray]] = {}

        if self._cache_dir is None:
            return {}, {}

        # Determine which columns we need
        def _extract(df: pd.DataFrame):
            for col in df.columns:
                if col.startswith("bm_"):
                    name = col[3:]
                    arr = df[col].to_numpy(dtype=np.float32)
                    biomarker_parts.setdefault(name, []).append(arr)
                elif col.startswith("path_"):
                    name = col[5:]
                    arr = df[col].fillna(0).to_numpy(dtype=int)
                    pathology_parts.setdefault(name, []).append(arr)

        def _build_arrays():
            return (
                {
                    name: parts[0]
                    if len(parts) == 1
                    else np.concatenate(parts)
                    for name, parts in biomarker_parts.items()
                },
                {
                    name: parts[0]
                    if len(parts) == 1
                    else np.concatenate(parts)
                    for name, parts in pathology_parts.items()
                },
            )

        # Try consolidated batch_results first
        parquet_path = self._cache_dir / "batch_results.parquet"
        csv_path = self._cache_dir / "batch_results.csv"
        if parquet_path.exists() and not _HAS_PARQUET and not csv_path.exists():
            raise _parquet_engine_error(parquet_path)
        if _HAS_PARQUET and parquet_path.exists():
            try:
                # Read only bm_/path_ columns — much faster than load_all()
                import pyarrow.parquet as pq
                schema = pq.read_schema(parquet_path)
                cols = [c for c in schema.names if c.startswith("bm_") or c.startswith("path_")]
                if cols:
                    df = pd.read_parquet(parquet_path, columns=cols)
                    _extract(df)
                    return _build_arrays()
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"Parquet-Cache konnte nicht gelesen werden ('{parquet_path}'): {exc}. "
                    "Fällt auf CSV-Fallback zurück.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        df = self._load_df("batch_results")
        if df is not None:
            _extract(df)
            return _build_arrays()

        # Fall back to chunks
        for f in self._iter_chunk_files():
            try:
                if _HAS_PARQUET and f.suffix == ".parquet":
                    chunk_df = pd.read_parquet(f)
                elif f.suffix == ".parquet":
                    raise _parquet_engine_error(f)
                else:
                    chunk_df = pd.read_csv(f)
                _extract(chunk_df)
            except ParquetEngineUnavailableError:
                raise
            except Exception:
                pass

        return _build_arrays()

    def get_analysis_arrays_full(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Like get_analysis_arrays() but also returns per-row metadata (age, sex).

        Returns:
            (biomarker_data, pathology_data, metadata)
            metadata keys:
              "age"  — float32 array, NaN where not recorded
              "sex"  — object (str) array, "" where not recorded ("M" / "F" otherwise)
        """
        biomarker_parts: dict[str, list[np.ndarray]] = {}
        pathology_parts: dict[str, list[np.ndarray]] = {}
        age_parts: list[np.ndarray] = []
        sex_parts: list[np.ndarray] = []

        if self._cache_dir is None:
            return {}, {}, {}

        def _extract(df: pd.DataFrame):
            n = len(df)
            for col in df.columns:
                if col.startswith("bm_"):
                    name = col[3:]
                    arr = df[col].to_numpy(dtype=np.float32)
                    biomarker_parts.setdefault(name, []).append(arr)
                elif col.startswith("path_"):
                    name = col[5:]
                    arr = df[col].fillna(0).to_numpy(dtype=int)
                    pathology_parts.setdefault(name, []).append(arr)
            age_parts.append(
                pd.to_numeric(df["age"], errors="coerce").to_numpy(dtype=np.float32)
                if "age" in df.columns
                else np.full(n, np.nan, dtype=np.float32)
            )
            sex_parts.append(
                df["sex"].fillna("").astype(str).to_numpy()
                if "sex" in df.columns
                else np.full(n, "", dtype=object)
            )

        def _build_meta() -> dict[str, np.ndarray]:
            return {
                "age": np.concatenate(age_parts) if age_parts else np.array([], dtype=np.float32),
                "sex": np.concatenate(sex_parts) if sex_parts else np.array([], dtype=object),
            }

        def _build_arrays():
            return (
                {
                    name: parts[0]
                    if len(parts) == 1
                    else np.concatenate(parts)
                    for name, parts in biomarker_parts.items()
                },
                {
                    name: parts[0]
                    if len(parts) == 1
                    else np.concatenate(parts)
                    for name, parts in pathology_parts.items()
                },
                _build_meta(),
            )

        parquet_path = self._cache_dir / "batch_results.parquet"
        csv_path = self._cache_dir / "batch_results.csv"
        if parquet_path.exists() and not _HAS_PARQUET and not csv_path.exists():
            raise _parquet_engine_error(parquet_path)
        if _HAS_PARQUET and parquet_path.exists():
            try:
                import pyarrow.parquet as pq
                schema = pq.read_schema(parquet_path)
                meta_cols = [c for c in ("age", "sex") if c in schema.names]
                cols = [c for c in schema.names if c.startswith("bm_") or c.startswith("path_")] + meta_cols
                if cols:
                    df = pd.read_parquet(parquet_path, columns=cols)
                    _extract(df)
                    return _build_arrays()
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"Parquet-Cache konnte nicht gelesen werden ('{parquet_path}'): {exc}. "
                    "Fällt auf CSV-Fallback zurück.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        df = self._load_df("batch_results")
        if df is not None:
            _extract(df)
            return _build_arrays()

        for f in self._iter_chunk_files():
            try:
                if _HAS_PARQUET and f.suffix == ".parquet":
                    chunk_df = pd.read_parquet(f)
                elif f.suffix == ".parquet":
                    raise _parquet_engine_error(f)
                else:
                    chunk_df = pd.read_csv(f)
                _extract(chunk_df)
            except ParquetEngineUnavailableError:
                raise
            except Exception:
                pass

        return _build_arrays()

    def load_cached_records(self, ecg_ids: list[int], data_loader=None) -> list:
        """Load minimal ProcessedRecord stubs for the given IDs (no signal data)."""
        from .pipeline import ProcessedRecord
        from .data_loader import ECGRecord

        df = self.load_all()
        if df is None:
            return []

        df = df[df["ecg_id"].isin(set(ecg_ids))]
        if df.empty:
            return []

        records = []
        for _, row in df.iterrows():
            ecg_id = int(row["ecg_id"])
            ecg_record = None
            if data_loader is not None:
                try:
                    ecg_record = data_loader.get_record(ecg_id, load_signal=False)
                except Exception:
                    pass
            if ecg_record is None:
                ecg_record = ECGRecord(
                    ecg_id=ecg_id, patient_id=0,
                    age=row.get("age"), sex=row.get("sex"),
                    height=None, weight=None, recording_date=None,
                    scp_codes={}, report="", heart_axis=None,
                    diagnostic_superclass=[], diagnostic_subclass=[],
                    validated_by_human=False, strat_fold=0,
                )
            biomarker_dict = {}
            pathology_labels = {}
            for col in row.index:
                if col.startswith("bm_"):
                    biomarker_dict[col[3:]] = float(row[col]) if pd.notna(row[col]) else None
                elif col.startswith("path_"):
                    pathology_labels[col[5:]] = bool(row[col])
            records.append(ProcessedRecord(
                ecg_record=ecg_record,
                preprocessed={},
                vcg_result=None,
                delineation=None,
                biomarker_results=[],
                biomarker_dict=biomarker_dict,
                pathology_labels=pathology_labels,
            ))
        return records

    # ------------------------------------------------------------------
    # Checkpoint manifest
    # ------------------------------------------------------------------

    def save_checkpoint_manifest(
        self,
        biomarker_names: list[str],
        pathology_names: list[str],
        vcg_method: str = "",
        n_records: int = 0,
    ):
        if self._cache_dir is None:
            return
        manifest = {
            "biomarker_names": sorted(biomarker_names),
            "pathology_names": sorted(pathology_names),
            "vcg_method": vcg_method,
            "n_records": n_records,
            "config_hash": self._config_hash or "",
            "updated_at": time.time(),
        }
        self._atomic_write_text(
            self._cache_dir / "checkpoint_manifest.json",
            json.dumps(manifest, indent=2),
        )

    def load_checkpoint_manifest(self) -> Optional[dict]:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / "checkpoint_manifest.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def check_checkpoint_valid(
        self,
        biomarker_names: list[str],
        pathology_names: list[str],
    ) -> tuple[bool, str]:
        """Return (is_valid, reason). Empty reason string means valid."""
        manifest = self.load_checkpoint_manifest()
        if manifest is None:
            return True, ""

        stored_bm = set(manifest.get("biomarker_names", []))
        stored_path = set(manifest.get("pathology_names", []))
        current_bm = set(biomarker_names)
        current_path = set(pathology_names)

        msgs = []
        added_bm = current_bm - stored_bm
        removed_bm = stored_bm - current_bm
        added_path = current_path - stored_path
        removed_path = stored_path - current_path

        if added_bm:
            msgs.append(f"Neue Biomarker: {sorted(added_bm)}")
        if removed_bm:
            msgs.append(f"Entfernte Biomarker: {sorted(removed_bm)}")
        if added_path:
            msgs.append(f"Neue Pathologien: {sorted(added_path)}")
        if removed_path:
            msgs.append(f"Entfernte Pathologien: {sorted(removed_path)}")

        if msgs:
            return False, "; ".join(msgs)
        return True, ""

    # ------------------------------------------------------------------
    # Analysis-Run persistence
    # ------------------------------------------------------------------

    def save_analysis_run(self, run) -> bool:
        if self._cache_dir is None:
            return False
        ok = self._atomic_pickle(self._cache_dir / "analysis_run.pkl", run)
        if ok and len(getattr(run, "results", []) or []) > 0:
            self.clear_analysis_checkpoint()
        return ok

    def load_analysis_run(self):
        if self._cache_dir is None:
            return None
        path = self._cache_dir / "analysis_run.pkl"
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def has_analysis_run(self) -> bool:
        if self._cache_dir is None:
            return False
        return (self._cache_dir / "analysis_run.pkl").exists()

    def clear_analysis(self):
        if self._cache_dir is None:
            return
        p = self._cache_dir / "analysis_run.pkl"
        if p.exists():
            p.unlink()
        self.clear_analysis_checkpoint()

    # ------------------------------------------------------------------
    # Normative-Range persistence
    # ------------------------------------------------------------------

    def save_normative_ranges(self, store) -> bool:
        if self._cache_dir is None:
            return False
        return self._atomic_pickle(self._cache_dir / "normative_ranges.pkl", store)

    def load_normative_ranges(self):
        if self._cache_dir is None:
            return None
        path = self._cache_dir / "normative_ranges.pkl"
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def has_normative_ranges(self) -> bool:
        if self._cache_dir is None:
            return False
        return (self._cache_dir / "normative_ranges.pkl").exists()

    def save_analysis_checkpoint(self, payload: dict) -> bool:
        if self._cache_dir is None:
            return False
        payload = dict(payload)
        payload["schema"] = "vcg_biomarker.analysis_checkpoint.v1"
        payload["updated_at"] = time.time()
        return self._atomic_pickle(self._cache_dir / "analysis_checkpoint.pkl", payload)

    def load_analysis_checkpoint(self) -> Optional[dict]:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / "analysis_checkpoint.pkl"
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
        return None

    def has_analysis_checkpoint(self) -> bool:
        if self._cache_dir is None:
            return False
        return (self._cache_dir / "analysis_checkpoint.pkl").exists()

    def clear_analysis_checkpoint(self):
        if self._cache_dir is None:
            return
        for name in ("analysis_checkpoint.pkl", "analysis_checkpoint.pkl.tmp"):
            p = self._cache_dir / name
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Clear everything
    # ------------------------------------------------------------------

    def clear(self):
        if self._cache_dir is None:
            return
        for pattern in ("*.parquet", "*.csv", "*.txt", "*.json", "*.pkl", "*.tmp"):
            for f in self._cache_dir.glob(pattern):
                try:
                    f.unlink()
                except Exception:
                    pass
        # Clear chunk subdirectory
        d = self._chunks_dir()
        if d and d.exists():
            for f in d.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            try:
                d.rmdir()
            except Exception:
                pass
        self._cached_ids.clear()
