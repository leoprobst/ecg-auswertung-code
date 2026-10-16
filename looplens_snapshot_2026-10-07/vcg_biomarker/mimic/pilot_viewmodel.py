"""Read-only pilot result and clinical-context access, with a writable overlay."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import MimicProjectConfig
from .pilot_workspace import confined, WORKSPACE_VERSION
from .storage import file_sha256


def laboratory_segment_mask(n_samples: int, arrays: dict, segment: str) -> np.ndarray:
    """Map UI segment names to the laboratory's measured-segment contract."""
    if segment == "Gesamtsignal":
        return np.ones(n_samples, dtype=bool)
    backend_name = {"P": "p_wave", "QRS": "qrs_complex", "T": "t_wave"}.get(segment)
    if backend_name is None:
        raise ValueError(f"Unsupported laboratory segment: {segment}")
    names = arrays.get("segment_names", [])
    starts = arrays.get("segment_start_samples", [])
    ends = arrays.get("segment_end_samples", [])
    if len(names) != len(starts) or len(names) != len(ends):
        raise ValueError("Inconsistent laboratory segment arrays")
    selected = np.zeros(n_samples, dtype=bool)
    for name, start, end in zip(names, starts, ends):
        if str(name).casefold() in {backend_name, segment.casefold()}:
            # Clip both bounds: negative stop indices must not select almost
            # the entire recording for a segment preceding the signal.
            lower, upper = max(0, min(n_samples, int(start))), max(0, min(n_samples, int(end)+1))
            if upper > lower:
                selected[lower:upper] = True
    return selected


class PilotViewModel:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifest = json.loads((root / "workspace.json").read_text(encoding="utf-8"))
        if (self.manifest.get("version") != WORKSPACE_VERSION
                or self.manifest.get("production_resume_allowed") is not False
                or self.manifest.get("locked_test_read") is not False):
            raise ValueError("Unsupported or unsafe pilot workspace")
        self.run_id = str(self.manifest["run_id"])
        self._hashes = {x["path"]: x["sha256"] for x in self.manifest["files"]}
        self._frames: dict[tuple[str, tuple[str, ...] | None], pd.DataFrame] = {}
        self._verified: dict[str, tuple[int, int]] = {}
        self._feature_frames: dict[tuple, pd.DataFrame] = {}
        self._endpoint_labels: dict[str, pd.DataFrame] = {}
        self._case_frames: dict[str, tuple[tuple, pd.DataFrame]] = {}

    def _verified_path(self, relative: str) -> Path:
        path = confined(self.root, relative)
        if relative not in self._hashes:
            raise ValueError("Artifact not in immutable workspace inventory")
        info = path.stat()
        identity = (info.st_size, info.st_mtime_ns)
        if self._verified.get(relative) != identity:
            if file_sha256(path) != self._hashes[relative]:
                raise ValueError(f"Pilot artifact integrity failed: {relative}")
            self._verified[relative] = identity
        return path

    def read_frame(self, relative: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
        path = self._verified_path(relative)
        key = (relative, columns)
        if key not in self._frames:
            self._frames[key] = pd.read_parquet(path, columns=list(columns) if columns is not None else None)
        return self._frames[key].copy()

    def results(self, size: int) -> pd.DataFrame:
        if size not in self.manifest["pilot_sizes"]:
            raise ValueError("Unknown pilot size")
        frame = self.read_frame(f"original/runs/{self.run_id}-pilot-{size}/marker_pathology_explorer.parquet")
        # Preserve the original AUC; correct only its presentation. No source
        # files or scientific thresholds are rewritten by this adapter.
        frame["historical_evidence_grade"] = frame.get("evidence_grade", "")
        frame["evidence_grade"] = "D"
        frame["evidence_stage"] = "exploratory"
        frame["clinical_claim_allowed"] = False
        frame["analysis_scope"] = "pilot"
        frame["evidence_grade_de"] = "Pilot-Hypothese · keine klinische Freigabe"
        frame["reference_kind"] = np.select(
            [frame["endpoint_id"].str.contains("qtc_gt"),
             frame["endpoint_id"].str.contains("icd"),
             frame["endpoint_id"].str.contains("lvef|echo|kalemia|calcemia|magnesemia")],
            ["same_ecg_device", "administrative_or_text", "quantitative_reference_requires_case_review"],
            default="reference_requires_case_review")
        return frame

    def features(self, size: int, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
        if size not in self.manifest["pilot_sizes"]:
            raise ValueError("Unknown pilot size")
        prefix = f"original/features/{size}/split="
        # Recheck identities on cache hits too. Do not retain both all source
        # partitions and their concatenation in the generic frame cache.
        paths = [self._verified_path(name) for name in self._hashes
                 if name.startswith(prefix) and name.endswith(".parquet")]
        key = (size, columns)
        if key in self._feature_frames:
            return self._feature_frames[key].copy()
        parts = []
        for path in paths:
            names = pq.ParquetFile(path).schema_arrow.names
            required = {"subject_id", "study_id", "extraction_status"}
            if not required.issubset(names):
                raise ValueError("Pilot feature partition lacks record identity or extraction status")
            if columns is None:
                part = pd.read_parquet(path)
            else:
                part = pd.read_parquet(path, columns=[name for name in columns if name in names])
                # A historical marker may be absent in a partition. Preserve
                # its missingness rather than dropping those records.
                part = part.reindex(columns=list(columns))
            parts.append(part)
        if not parts:
            raise ValueError("Pilot feature partitions missing")
        result = pd.concat(parts, ignore_index=True)
        self._feature_frames = {key: result}
        return result.copy()

    def availability(self, size: int, endpoint: str, marker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        features = self.features(size, ("subject_id", "study_id", "extraction_status", marker))
        label_path = self._verified_path("original/labels/endpoint_index_cohorts.parquet")
        if endpoint not in self._endpoint_labels:
            labels = pd.read_parquet(label_path,
                columns=["subject_id", "study_id", "label_id", "label_state",
                         "eligibility_state", "evidence_summary", "time_delta_hours"],
                filters=[("label_id", "==", endpoint)])
            self._endpoint_labels = {endpoint: labels}
        labels = self._endpoint_labels[endpoint]
        if marker not in features:
            features = features.reindex(columns=[*features.columns, marker])
        joined = features[["subject_id", "study_id", "extraction_status", marker]].merge(
            labels[["subject_id", "study_id", "label_state", "eligibility_state", "evidence_summary", "time_delta_hours"]],
            on=["subject_id", "study_id"], how="inner", validate="one_to_one")
        qc = joined["extraction_status"].eq("ok")
        measured = qc & np.isfinite(pd.to_numeric(joined[marker], errors="coerce"))
        verified = joined["label_state"].isin(["positive", "negative"]) & joined["eligibility_state"].eq("eligible")
        joined["measurement_eligible"] = measured & verified
        counts = pd.DataFrame({"step": ["Ausgewählte Index-EKGs", "Extraktion/QC bestanden",
                              "Marker messbar", "Marker + Referenz verfügbar"],
                              "n": [len(joined), int(qc.sum()), int(measured.sum()), int((measured & verified).sum())]})
        return counts, joined

    def cases(self) -> pd.DataFrame:
        return self.read_frame("original/case_index.parquet")

    def _case_frame(self, relative: str, columns: tuple[str, ...], study_id: int) -> pd.DataFrame:
        """Push case selection into Parquet; retain only one case per source.

        Integrity is rechecked even on cache hits. Return independent frames so
        GUI annotations cannot mutate subsequent reads of archived context.
        """
        path = self._verified_path(relative)
        key = (columns, int(study_id))
        cached = self._case_frames.get(relative)
        if cached is None or cached[0] != key:
            frame = pd.read_parquet(path, columns=list(columns),
                                    filters=[("study_id", "==", int(study_id))])
            if len(frame) > 1:
                raise ValueError("Ambiguous archived study identity")
            self._case_frames[relative] = (key, frame)
        return self._case_frames[relative][1].copy()

    def waveform(self, study_id: int) -> Path:
        cases = self.cases()
        row = cases.loc[cases["study_id"].eq(study_id)].iloc[0]
        relative = str(row["waveform"])
        path = confined(self.root, relative)
        for suffix in (".hea", ".dat"):
            name = Path(relative).with_suffix(suffix).as_posix()
            if name not in self._hashes or file_sha256(path.with_suffix(suffix)) != self._hashes[name]:
                raise ValueError("Waveform integrity failed")
        return path

    def timeline(self, study_id: int) -> pd.DataFrame:
        relative = "original/evidence/clinical_context_features.parquet"
        names = pq.ParquetFile(self._verified_path(relative)).schema_arrow.names
        times = [name for name in names if name.endswith("_delta_hours")]
        values = [name.removesuffix("_delta_hours") + "_nearest" for name in times]
        selected = self._case_frame(relative,
            tuple(["study_id", *times, *[name for name in values if name in names]]), study_id)
        if selected.empty:
            return pd.DataFrame(columns=["event", "hours_from_ecg", "value"])
        row = selected.iloc[0]
        events = [{"event": "Index-EKG", "hours_from_ecg": 0.0, "value": "Zeitnull"}]
        for key, value in row.items():
            if not str(key).endswith("_delta_hours"):
                continue
            delta = pd.to_numeric(value, errors="coerce")
            if pd.isna(delta) or not np.isfinite(delta):
                continue
            stem = str(key).removesuffix("_delta_hours")
            measurement = row.get(stem + "_nearest", row.get(stem, "Referenzzeitpunkt"))
            events.append({"event": stem, "hours_from_ecg": float(delta), "value": str(measurement)})
        return pd.DataFrame(events).sort_values("hours_from_ecg")

    def acquisition(self, study_id: int) -> dict:
        """Read only hash-verified archived metadata, never production labels."""
        relative = "original/master/ecg_master_index.parquet"
        if relative not in self._hashes:
            return {"metadata_source": "archived_master_unavailable"}
        names = pq.ParquetFile(self._verified_path(relative)).schema_arrow.names
        selected_columns = tuple(name for name in ("study_id", "machine_bandwidth", "machine_report",
                                                   "original_sampling_rate") if name in names)
        if "study_id" not in selected_columns:
            raise ValueError("Archived master lacks study identity")
        rows = self._case_frame(relative, selected_columns, study_id)
        if rows.empty:
            return {"metadata_source": "archived_study_unavailable"}
        row = rows.iloc[0]
        from .feature_record import _acquisition_upper_bandwidth_hz
        from .report_parser import MachineReportParser, ReportMentionState
        report = row.get("machine_report")
        parsed = MachineReportParser().parse("" if pd.isna(report) else str(report))
        metadata = {"metadata_source": "hash_verified_archived_master",
            "upper_bandwidth_hz": _acquisition_upper_bandwidth_hz(row.get("machine_bandwidth")),
            "pacing_present": True if parsed["pacemaker"].state == ReportMentionState.POSITIVE else None,
            "pacing_source": "machine_report_positive_only_not_independent_adjudication"}
        original = pd.to_numeric(row.get("original_sampling_rate"), errors="coerce")
        if pd.notna(original) and np.isfinite(original):
            metadata["original_sampling_rate_hz"] = float(original)
        return metadata

    def overlay_config(self) -> MimicProjectConfig:
        # No production jobs, criteria, cases or catalog can be found here.
        original = self.root / "original"
        return MimicProjectConfig.discover(original, self.root / "overlay")
