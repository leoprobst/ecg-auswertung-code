from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


DEFAULT_DATABASE_ROOT = Path(r"E:\Database")
DEFAULT_WORK_ROOT = Path(r"E:\VCG\mimic_validation")


# These fields only control how a stage is executed.  They must be recorded in
# execution provenance, but changing them must neither alter the scientific
# identity of an artifact nor force a multi-day recomputation.
OPERATIONAL_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "workers",
        "duckdb_memory_limit_gb",
        "resource_mode",
        "os_memory_reserve_gb",
        "feature_micro_batch_size",
        "metadata",
    }
)


def _config_payload(config: "MimicProjectConfig | dict[str, Any]") -> dict[str, Any]:
    """Normalise anything config-shaped into a plain mapping.

    Duck-typed on ``to_dict`` rather than on the concrete class. The evidence
    attestation runs the whole build against a shadow root through a wrapper
    that carries ``to_dict`` but is not a ``MimicProjectConfig``; an isinstance
    check sent it to ``dict(config)``, which raised. The attestation therefore
    could never succeed, and its caller reported that as "exact event-layer
    equivalence could not be proven" -- blaming the data for a type error.
    """

    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    return dict(config)


def scientific_config_payload(
    config: "MimicProjectConfig | dict[str, Any]",
) -> dict[str, Any]:
    """Return the stable, result-defining part of a project configuration."""

    payload = _config_payload(config)
    return {
        key: value
        for key, value in payload.items()
        if key not in OPERATIONAL_CONFIG_FIELDS
    }


def operational_config_payload(
    config: "MimicProjectConfig | dict[str, Any]",
) -> dict[str, Any]:
    """Return resource/execution knobs without scientific settings."""

    payload = _config_payload(config)
    return {
        key: payload[key]
        for key in sorted(OPERATIONAL_CONFIG_FIELDS)
        if key in payload
    }


@dataclass
class MimicProjectConfig:
    mimic_iv_root: Path
    mimic_ed_root: Path
    mimic_ecg_root: Path
    echo_structured_path: Path
    echo_study_list_path: Path
    echo_record_list_path: Path
    work_root: Path = DEFAULT_WORK_ROOT
    mimic_note_root: Optional[Path] = None
    # PhysioNet/CinC Challenge 2021 training tree. External
    # cohorts are the only thing in this project that can speak
    # to generalisability beyond one hospital.
    external_ecg_root: Optional[Path] = None
    external_identity_manifest: Optional[Path] = None
    external_identity_manifest_sha256: Optional[str] = None
    code15_root: Optional[Path] = None
    sami_trop_root: Optional[Path] = None
    mimic_iv_release: str = "3.1"
    mimic_ed_release: str = "2.2"
    mimic_ecg_release: str = "1.0"
    mimic_echo_release: str = "1.0"
    split_salt: str = "vcg-mimic-validation-v1"
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    locked_test_fraction: float = 0.20
    workers: int = 6
    duckdb_memory_limit_gb: int = 20
    resource_mode: str = "auto"
    os_memory_reserve_gb: float = 6.0
    feature_micro_batch_size: int = 25
    random_seed: int = 20260725
    powerline_hz: float = 60.0
    primary_vcg_method: str = "kors"
    sensitivity_vcg_methods: tuple[str, ...] = ("inverse_dower",)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "mimic_iv_root",
            "mimic_ed_root",
            "mimic_ecg_root",
            "echo_structured_path",
            "echo_study_list_path",
            "echo_record_list_path",
            "work_root",
        ):
            setattr(self, name, Path(getattr(self, name)))
        if self.mimic_note_root is not None:
            self.mimic_note_root = Path(self.mimic_note_root)
        if self.external_ecg_root is not None:
            self.external_ecg_root = Path(self.external_ecg_root)
        if self.external_identity_manifest is not None:
            self.external_identity_manifest = Path(self.external_identity_manifest)
            digest = self.external_identity_manifest_sha256
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("External identity manifest requires its frozen lowercase SHA-256")
        elif self.external_identity_manifest_sha256 is not None:
            raise ValueError("External identity SHA-256 requires a manifest path")
        if self.code15_root is not None:
            self.code15_root = Path(self.code15_root)
        if self.sami_trop_root is not None:
            self.sami_trop_root = Path(self.sami_trop_root)
        total = self.train_fraction + self.validation_fraction + self.locked_test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")
        if min(self.train_fraction, self.validation_fraction, self.locked_test_fraction) <= 0:
            raise ValueError("Every split fraction must be positive")
        if self.workers < 1:
            raise ValueError("workers must be at least one")
        if self.resource_mode not in ("auto", "manual"):
            raise ValueError("resource_mode must be 'auto' or 'manual'")
        if self.os_memory_reserve_gb < 2:
            raise ValueError("At least 2 GB must remain reserved for the OS and GUI")
        if self.feature_micro_batch_size < 1:
            raise ValueError("feature_micro_batch_size must be at least one")

    @classmethod
    def discover(
        cls,
        database_root: Path | str = DEFAULT_DATABASE_ROOT,
        work_root: Path | str = DEFAULT_WORK_ROOT,
    ) -> "MimicProjectConfig":
        root = Path(database_root)
        echo_structured = _first_existing(
            root / "structured-measurement.csv.gz",
            root / "structured-measurement.csv",
        )
        return cls(
            mimic_iv_root=root / "mimic-iv-3.1",
            mimic_ed_root=root / "mimic-iv-ed-2.2",
            mimic_ecg_root=(
                root
                / "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
            ),
            mimic_note_root=_first_existing_optional(
                root / "mimic-iv-note",
                root / "mimic-iv-note-2.2",
                root / "mimic-iv-note-2.3",
            ),
            external_ecg_root=_first_existing_optional(
                root / "physionet-challenge-2021" / "training",
                root / "physionet-challenge-2021",
            ),
            code15_root=_first_existing_optional(root / "code15"),
            sami_trop_root=_first_existing_optional(root / "sami-trop"),
            echo_structured_path=echo_structured,
            echo_study_list_path=root / "echo-study-list.csv",
            echo_record_list_path=root / "echo-record-list.csv",
            work_root=Path(work_root),
        )

    @property
    def project_file(self) -> Path:
        return self.work_root / "project.json"

    @property
    def catalog_path(self) -> Path:
        return self.work_root / "catalog.duckdb"

    @property
    def standardized_root(self) -> Path:
        return self.work_root / "standardized"

    @property
    def evidence_root(self) -> Path:
        return self.work_root / "evidence"

    @property
    def master_root(self) -> Path:
        return self.work_root / "master"

    @property
    def features_root(self) -> Path:
        return self.work_root / "features"

    @property
    def labels_root(self) -> Path:
        return self.work_root / "labels"

    @property
    def runs_root(self) -> Path:
        return self.work_root / "runs"

    @property
    def quarantine_root(self) -> Path:
        return self.work_root / "quarantine"

    @property
    def jobs_root(self) -> Path:
        return self.work_root / "jobs"

    @property
    def jobs_database_path(self) -> Path:
        return self.jobs_root / "jobs.sqlite"

    @property
    def snapshots_root(self) -> Path:
        return self.work_root / "snapshots"

    @property
    def cases_root(self) -> Path:
        return self.work_root / "cases"

    @property
    def external_validation_root(self) -> Path:
        return self.work_root / "external"

    @property
    def resources_root(self) -> Path:
        return self.work_root / "resources"

    @property
    def delineation_benchmark_root(self) -> Path:
        return self.resources_root / "delineation" / "datasets"

    def ensure_directories(self) -> None:
        for path in (
            self.work_root,
            self.standardized_root,
            self.evidence_root,
            self.master_root,
            self.features_root,
            self.labels_root,
            self.runs_root,
            self.quarantine_root,
            self.jobs_root,
            self.snapshots_root,
            self.cases_root,
            self.resources_root,
            self.external_validation_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def validate_paths(self) -> list[str]:
        errors: list[str] = []
        required_dirs = {
            "MIMIC-IV": self.mimic_iv_root,
            "MIMIC-IV-ED": self.mimic_ed_root,
            "MIMIC-IV-ECG": self.mimic_ecg_root,
        }
        required_files = {
            "structured echo measurements": self.echo_structured_path,
            "echo study list": self.echo_study_list_path,
            "echo record list": self.echo_record_list_path,
        }
        for label, path in required_dirs.items():
            if not path.is_dir():
                errors.append(f"{label} directory not found: {path}")
        for label, path in required_files.items():
            if not path.is_file():
                errors.append(f"{label} file not found: {path}")
        try:
            self.work_root.mkdir(parents=True, exist_ok=True)
            probe = self.work_root / ".write_probe"
            probe.write_text("ok", encoding="ascii")
            probe.unlink()
        except OSError as exc:
            errors.append(f"Work directory is not writable: {self.work_root} ({exc})")
        return errors

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
            elif isinstance(value, tuple):
                data[key] = list(value)
        return data

    def save(self, path: Optional[Path | str] = None) -> Path:
        from .storage import atomic_write_json

        self.ensure_directories()
        target = Path(path) if path is not None else self.project_file
        atomic_write_json(target, self.to_dict())
        return target

    @classmethod
    def load(cls, path: Path | str) -> "MimicProjectConfig":
        source = Path(path)
        if source.is_dir():
            source = source / "project.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MimicProjectConfig":
        data = dict(data)
        data["sensitivity_vcg_methods"] = tuple(
            data.get("sensitivity_vcg_methods", ("inverse_dower",))
        )
        return cls(**data)


def load_or_discover_config(
    project: Optional[Path | str] = None,
) -> MimicProjectConfig:
    if project is not None:
        path = Path(project)
        if path.exists():
            return MimicProjectConfig.load(path)
    env_project = os.getenv("VCG_MIMIC_PROJECT")
    if env_project and Path(env_project).exists():
        return MimicProjectConfig.load(env_project)
    config = MimicProjectConfig.discover()
    if config.project_file.exists():
        return MimicProjectConfig.load(config.project_file)
    return config


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _first_existing_optional(*paths: Path) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None
