"""Fail-closed approval and artifact binding for external PR error correction."""
import json

import numpy as np

from .storage import file_sha256

VERSION = "1.0.0"
REQUIRED_APPROVAL_GATES = ("approved_for_correction", "reference_error_identified",
                           "target_transportability_verified", "development_overlap_excluded")


def load_approved_calibration(certificate_path, beat_path):
    """Approval must identify reference error and demonstrate target transport.

    A filename in a row or qualification_verified=True is insufficient. This
    checks the declared evidence contract, not a substitute for expert review.
    """
    if not certificate_path.is_file():
        return None, "independent_calibration_certificate_missing"
    try:
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return None, "invalid_calibration_certificate"
    if not isinstance(certificate, dict):
        return None, "invalid_calibration_certificate"
    if any(certificate.get(name) is not True for name in REQUIRED_APPROVAL_GATES):
        return None, "independent_calibration_gates_unmet"
    if certificate.get("beat_artifact_sha256") != file_sha256(beat_path):
        return None, "qualified_beat_artifact_hash_mismatch"
    from .feature_extraction import feature_implementation_hash
    if certificate.get("feature_implementation_hash") != feature_implementation_hash():
        return None, "calibration_measurement_implementation_mismatch"
    for name in ("measurement_version", "transform", "review_report_sha256"):
        if not isinstance(certificate.get(name), str) or not certificate[name].strip():
            return None, "calibration_provenance_missing"
    report = certificate_path.with_name("pr_calibration_review_report.json")
    if not report.is_file() or certificate["review_report_sha256"] != file_sha256(report):
        return None, "independent_review_report_missing_or_changed"
    try:
        values = np.asarray([certificate["pr_error_variance_ms2"], certificate["error_covariance"]], float)
    except (KeyError, ValueError, TypeError):
        return None, "calibration_moments_missing"
    if not np.isfinite(values).all() or values[0] < 0:
        return None, "invalid_calibration_moments"
    return {**certificate, "certificate_sha256": file_sha256(certificate_path)}, ""
