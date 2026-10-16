"""Read-only, integrity-checked blinded signal packets; never coordinator files."""
from pathlib import Path
import hashlib
import io
import json
import re
import zipfile

import numpy as np
import pandas as pd

from ..core.data_loader import LEAD_NAMES_12
from .reviewer_export import REVIEW_EXPORT_VERSION


class ReviewerPacket:
    """Hashes detect changes relative to the packet, not trusted authorship.

    A trusted delivery channel is still required. No file is modified and no
    patient lookup, source path or coordinator mapping is followed.
    """
    MAX_SIGNAL_BYTES = 64 * 1024**2

    def __init__(self, folder: Path):
        self.folder = Path(folder).resolve()
        manifest_path = self.folder / "packet_complete.json"
        if manifest_path.is_symlink() or manifest_path.stat().st_size > 4*1024**2:
            raise ValueError("Invalid packet completion manifest")
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        if (manifest.get("version") != REVIEW_EXPORT_VERSION or manifest.get("status") != "complete"
                or manifest.get("clinical_use_allowed") is not False):
            raise ValueError("Incompatible or incomplete reviewer packet")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.hashes = manifest.get("files_sha256")
        if not isinstance(self.hashes, dict) or not {"cases.csv", "instructions.json"} <= set(self.hashes):
            raise ValueError("Missing reviewer packet files")
        for name, digest in self.hashes.items():
            if (name not in {"cases.csv", "instructions.json"} and not re.fullmatch(r"[0-9a-f]{32}\.npz", name)):
                raise ValueError("Unapproved reviewer packet file")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid packet file digest")
        actual = {p.name for p in self.folder.iterdir()}
        if actual != set(self.hashes) | {"packet_complete.json"}:
            raise ValueError("Unlisted files in blinded packet; do not include coordinator data")
        self.instructions = json.loads(self._read("instructions.json", 1024**2))
        if (self.instructions.get("pipeline_results_withheld") is not True
                or self.instructions.get("selection_strata_withheld") is not True
                or self.instructions.get("clinical_use_allowed") is not False
                or self.instructions.get("signal_unit") != "mV"
                or self.instructions.get("lead_order") != list(LEAD_NAMES_12)):
            raise ValueError("Incompatible blinding or signal instruction contract")
        self.cases = pd.read_csv(io.BytesIO(self._read("cases.csv", 8*1024**2)),
                                dtype={"review_case_id": str, "endpoint_id": str})
        required = {"review_case_id", "reviewer_slot", "endpoint_id", "blinded", "pipeline_label_withheld"}
        if set(self.cases) != required or self.cases.empty:
            raise ValueError("Invalid blinded case-list columns")
        if (self.cases.review_case_id.duplicated().any()
                or not self.cases.review_case_id.str.fullmatch(r"[0-9a-f]{32}").fillna(False).all()
                or self.cases.endpoint_id.isna().any()
                or not self.cases.blinded.eq(True).all()
                or not self.cases.pipeline_label_withheld.eq(True).all()):
            raise ValueError("Invalid case identity or blinding flags")
        slot = pd.to_numeric(self.cases.reviewer_slot, errors="coerce")
        if slot.nunique() != 1 or not (np.isfinite(slot) & slot.gt(0) & slot.mod(1).eq(0)).all():
            raise ValueError("Packet must belong to exactly one reviewer slot")
        expected = {f"{case}.npz" for case in self.cases.review_case_id} | {"cases.csv", "instructions.json"}
        if set(self.hashes) != expected:
            raise ValueError("Case-list and signal inventory do not match")

    def _read(self, name: str, limit: int) -> bytes:
        path = self.folder / name
        if name not in self.hashes or path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            raise ValueError("Missing, oversized or redirected packet file")
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Packet file exceeds memory budget")
        if hashlib.sha256(data).hexdigest() != self.hashes[name]:
            raise ValueError("Reviewer packet integrity failed")
        return data

    def signal(self, case_id: str) -> tuple[np.ndarray, float]:
        if case_id not in set(self.cases.review_case_id):
            raise ValueError("Case not assigned to this reviewer")
        data = self._read(f"{case_id}.npz", self.MAX_SIGNAL_BYTES)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(item.file_size for item in archive.infolist()) > self.MAX_SIGNAL_BYTES:
                raise ValueError("Decoded signal exceeds reviewer memory budget")
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            if set(arrays.files) != {"signal_mv", "sampling_rate_hz", "lead_names"}:
                raise ValueError("Unexpected signal metadata")
            signal = np.asarray(arrays["signal_mv"], dtype=float)
            rate_array = arrays["sampling_rate_hz"]
            if rate_array.shape != () or rate_array.dtype.kind not in "ifu":
                raise ValueError("Invalid signal sampling rate")
            rate = float(rate_array)
            if (signal.ndim != 2 or signal.shape[1] != 12 or len(signal) < 2
                    or not np.isfinite(signal).all() or not np.isfinite(rate) or rate <= 0
                    or arrays["lead_names"].tolist() != list(LEAD_NAMES_12)):
                raise ValueError("Invalid twelve-lead signal contract")
        return signal, rate
