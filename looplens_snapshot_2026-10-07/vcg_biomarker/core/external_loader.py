from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Mapping, Optional

import numpy as np

from .data_loader import ECGRecord, LEAD_NAMES_12


class ECGImportMappingRequired(ValueError):
    def __init__(
        self,
        message: str,
        *,
        missing_leads: tuple[str, ...] = (),
        missing_units: tuple[str, ...] = (),
        available_channels: tuple[str, ...] = (),
    ):
        super().__init__(message)
        self.missing_leads = missing_leads
        self.missing_units = missing_units
        self.available_channels = available_channels


@dataclass
class ExternalECGMeta:
    age: Optional[int] = None
    sex: Optional[str] = None
    sampling_rate: Optional[float] = None
    lead_names: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    lead_mapping: dict[str, str] = field(default_factory=dict)
    default_unit: Optional[str] = None
    source_format: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalECGLoadResult:
    signal_millivolts: np.ndarray
    sampling_rate_hz: float
    lead_names: tuple[str, ...]
    source_units: tuple[str, ...]
    source_format: str
    source_metadata: Mapping[str, Any] = field(default_factory=dict)


def load_wfdb(path: str | Path, meta: ExternalECGMeta) -> np.ndarray:
    result = _load_wfdb_result(path, meta)
    _update_meta(meta, result)
    return result.signal_millivolts


def load_edf(path: str | Path, meta: ExternalECGMeta) -> np.ndarray:
    result = _load_edf_result(path, meta)
    _update_meta(meta, result)
    return result.signal_millivolts


def load_csv(path: str | Path, meta: ExternalECGMeta) -> np.ndarray:
    result = _load_csv_result(path, meta)
    _update_meta(meta, result)
    return result.signal_millivolts


def load_dicom(path: str | Path, meta: ExternalECGMeta) -> np.ndarray:
    result = _load_dicom_result(path, meta)
    _update_meta(meta, result)
    return result.signal_millivolts


def load_npy(path: str | Path, meta: ExternalECGMeta) -> np.ndarray:
    result = _load_numpy_result(path, meta)
    _update_meta(meta, result)
    return result.signal_millivolts


def load_external_ecg_result(
    path: str | Path,
    meta: Optional[ExternalECGMeta] = None,
) -> ExternalECGLoadResult:
    metadata = meta or ExternalECGMeta()
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in (".hea", ".dat") or (
        not suffix and source.with_suffix(".hea").is_file()
    ):
        result = _load_wfdb_result(source, metadata)
    elif suffix == ".edf":
        result = _load_edf_result(source, metadata)
    elif suffix in (".csv", ".tsv", ".txt"):
        result = _load_csv_result(source, metadata)
    elif suffix in (".dcm", ".dicom"):
        result = _load_dicom_result(source, metadata)
    elif suffix in (".npy", ".npz"):
        result = _load_numpy_result(source, metadata)
    else:
        raise ValueError(f"Unsupported ECG file format: {suffix or '<none>'}")
    _validate_result(result)
    _update_meta(metadata, result)
    return result


def load_external_ecg(path: str | Path, meta: ExternalECGMeta) -> np.ndarray:
    return load_external_ecg_result(path, meta).signal_millivolts


def _load_wfdb_result(
    path: str | Path,
    meta: ExternalECGMeta,
) -> ExternalECGLoadResult:
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("Install wfdb to load WFDB records") from exc
    record_path = Path(path)
    if record_path.suffix.lower() in (".hea", ".dat"):
        record_path = record_path.with_suffix("")
    record = wfdb.rdrecord(str(record_path))
    if record.p_signal is None:
        raise ValueError("WFDB record contains no calibrated physical signal")
    signal = np.asarray(record.p_signal, dtype=float)
    names = tuple(str(item) for item in record.sig_name)
    indices = _strict_lead_indices(names, meta.lead_mapping)
    units = tuple(
        str(record.units[index] or meta.default_unit or "")
        for index in indices
    )
    ordered = _convert_units(signal[:, indices], units)
    sampling_rate = float(record.fs)
    return ExternalECGLoadResult(
        signal_millivolts=ordered,
        sampling_rate_hz=sampling_rate,
        lead_names=tuple(LEAD_NAMES_12),
        source_units=units,
        source_format="wfdb",
        source_metadata={
            "record_name": str(getattr(record, "record_name", "")),
            "comments": tuple(getattr(record, "comments", ()) or ()),
        },
    )


def _load_edf_result(
    path: str | Path,
    meta: ExternalECGMeta,
) -> ExternalECGLoadResult:
    try:
        import pyedflib
    except ImportError as exc:
        raise ImportError(
            "Install pyedflib to load EDF files. EDF import never substitutes "
            "missing leads."
        ) from exc
    reader = pyedflib.EdfReader(str(path))
    try:
        names = tuple(str(item).strip() for item in reader.getSignalLabels())
        indices = _strict_lead_indices(names, meta.lead_mapping)
        sample_rates = [float(reader.getSampleFrequency(index)) for index in indices]
        if max(sample_rates) - min(sample_rates) > 1e-6:
            raise ValueError(
                "Required EDF leads use different sampling rates; resample "
                "explicitly before import"
            )
        lengths = [int(reader.getNSamples()[index]) for index in indices]
        if len(set(lengths)) != 1:
            raise ValueError("Required EDF leads have different sample counts")
        signal = np.column_stack([reader.readSignal(index) for index in indices])
        units = tuple(
            str(reader.getPhysicalDimension(index) or meta.default_unit or "")
            for index in indices
        )
    finally:
        reader.close()
    return ExternalECGLoadResult(
        signal_millivolts=_convert_units(signal, units),
        sampling_rate_hz=sample_rates[0],
        lead_names=tuple(LEAD_NAMES_12),
        source_units=units,
        source_format="edf",
    )


def _load_csv_result(
    path: str | Path,
    meta: ExternalECGMeta,
) -> ExternalECGLoadResult:
    import pandas as pd

    source = Path(path)
    separator = "\t" if source.suffix.lower() in (".tsv", ".txt") else ","
    frame = pd.read_csv(source, sep=separator)
    columns = tuple(str(item) for item in frame.columns)
    indices = _strict_lead_indices(columns, meta.lead_mapping)
    ordered_columns = [frame.columns[index] for index in indices]
    signal = frame.loc[:, ordered_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    units = tuple(
        _unit_from_channel_name(name) or meta.default_unit or ""
        for name in ordered_columns
    )
    sampling_rate = _require_sampling_rate(meta)
    return ExternalECGLoadResult(
        signal_millivolts=_convert_units(signal, units),
        sampling_rate_hz=sampling_rate,
        lead_names=tuple(LEAD_NAMES_12),
        source_units=units,
        source_format="csv",
        source_metadata={"columns": columns},
    )


def _load_dicom_result(
    path: str | Path,
    meta: ExternalECGMeta,
) -> ExternalECGLoadResult:
    try:
        import pydicom
        from pydicom.waveforms import multiplex_array
    except ImportError as exc:
        raise ImportError("Install pydicom>=3.0 to load DICOM waveforms") from exc
    dataset = pydicom.dcmread(str(path))
    sequences = getattr(dataset, "WaveformSequence", None)
    if not sequences:
        raise ValueError("DICOM object contains no WaveformSequence")
    selected_index = None
    selected_names: tuple[str, ...] = ()
    for index, multiplex in enumerate(sequences):
        definitions = getattr(multiplex, "ChannelDefinitionSequence", ())
        names = tuple(_dicom_channel_name(item) for item in definitions)
        try:
            _strict_lead_indices(names, meta.lead_mapping)
        except ECGImportMappingRequired:
            continue
        selected_index = index
        selected_names = names
        break
    if selected_index is None:
        available = tuple(
            name
            for multiplex in sequences
            for name in (
                _dicom_channel_name(item)
                for item in getattr(multiplex, "ChannelDefinitionSequence", ())
            )
        )
        raise ECGImportMappingRequired(
            "No DICOM multiplex group contains all 12 required ECG leads",
            available_channels=available,
        )
    multiplex = sequences[selected_index]
    indices = _strict_lead_indices(selected_names, meta.lead_mapping)
    signal = np.asarray(
        multiplex_array(dataset, selected_index, as_raw=False),
        dtype=float,
    )
    units = tuple(
        _dicom_channel_unit(multiplex.ChannelDefinitionSequence[index])
        or meta.default_unit
        or ""
        for index in indices
    )
    return ExternalECGLoadResult(
        signal_millivolts=_convert_units(signal[:, indices], units),
        sampling_rate_hz=float(multiplex.SamplingFrequency),
        lead_names=tuple(LEAD_NAMES_12),
        source_units=units,
        source_format="dicom",
        source_metadata={
            "sop_instance_uid": str(
                getattr(dataset, "SOPInstanceUID", "")
            ),
            "multiplex_group": selected_index,
        },
    )


def _load_numpy_result(
    path: str | Path,
    meta: ExternalECGMeta,
) -> ExternalECGLoadResult:
    source = Path(path)
    embedded: dict[str, Any] = {}
    if source.suffix.lower() == ".npz":
        archive = np.load(str(source), allow_pickle=False)
        for key in ("ecg", "signal", "data"):
            if key in archive:
                array = archive[key]
                break
        else:
            array_keys = [
                key for key in archive.files if np.asarray(archive[key]).ndim >= 2
            ]
            if not array_keys:
                raise ValueError("NPZ contains no two-dimensional ECG array")
            array = archive[array_keys[0]]
        for key in ("sampling_rate", "sampling_rate_hz", "fs"):
            if key in archive:
                embedded["sampling_rate"] = float(np.asarray(archive[key]).item())
                break
        if "lead_names" in archive:
            embedded["lead_names"] = [
                str(item) for item in np.asarray(archive["lead_names"]).tolist()
            ]
        if "units" in archive:
            embedded["units"] = [
                str(item) for item in np.asarray(archive["units"]).tolist()
            ]
        if "simulation_provenance" in archive:
            provenance = json.loads(str(archive["simulation_provenance"]))
            if (not isinstance(provenance, dict) or provenance.get("schema") != "cardiac-simulation/1"
                    or provenance.get("synthetic") is not True
                    or provenance.get("clinical_use") is not False):
                raise ValueError("Invalid synthetic simulation provenance")
            embedded["synthetic"] = True
            embedded["simulation_provenance"] = provenance
        archive.close()
    else:
        array = np.load(str(source), allow_pickle=False)
    signal = np.asarray(array, dtype=float)
    if signal.ndim != 2:
        raise ValueError("NumPy ECG array must be two-dimensional")
    names = tuple(
        embedded.get("lead_names")
        or meta.lead_names
    )
    if not names:
        raise ECGImportMappingRequired(
            "NumPy ECG import requires explicit lead_names metadata"
        )
    if signal.shape[0] == len(names) and signal.shape[1] != len(names):
        signal = signal.T
    if signal.shape[1] != len(names):
        raise ValueError(
            f"NumPy channel count {signal.shape[1]} does not match "
            f"{len(names)} lead names"
        )
    indices = _strict_lead_indices(names, meta.lead_mapping)
    embedded_units = embedded.get("units") or meta.units
    if embedded_units:
        if len(embedded_units) != len(names):
            raise ValueError("NumPy units must match the channel count")
        units = tuple(str(embedded_units[index]) for index in indices)
    else:
        units = tuple(meta.default_unit or "" for _ in indices)
    sampling_rate = float(
        embedded.get("sampling_rate")
        or _require_sampling_rate(meta)
    )
    return ExternalECGLoadResult(
        signal_millivolts=_convert_units(signal[:, indices], units),
        sampling_rate_hz=sampling_rate,
        lead_names=tuple(LEAD_NAMES_12),
        source_units=units,
        source_format="numpy_research",
        source_metadata=embedded,
    )


def _strict_lead_indices(
    channel_names: tuple[str, ...] | list[str],
    explicit_mapping: Mapping[str, str],
) -> list[int]:
    available = tuple(str(item) for item in channel_names)
    normalized = [_normalize_lead_name(item) for item in available]
    indices: list[int] = []
    missing: list[str] = []
    for lead in LEAD_NAMES_12:
        configured = explicit_mapping.get(lead)
        if configured:
            candidates = [
                index
                for index, name in enumerate(available)
                if name == configured
                or _normalize_lead_name(name) == _normalize_lead_name(configured)
            ]
        else:
            canonical = _normalize_lead_name(lead)
            candidates = [
                index for index, value in enumerate(normalized) if value == canonical
            ]
        if len(candidates) != 1:
            missing.append(lead)
        else:
            indices.append(candidates[0])
    if missing:
        raise ECGImportMappingRequired(
            "Lead mapping is incomplete or ambiguous; no channel is silently "
            "substituted",
            missing_leads=tuple(missing),
            available_channels=available,
        )
    if len(set(indices)) != 12:
        raise ECGImportMappingRequired(
            "A source channel cannot represent more than one canonical ECG lead",
            available_channels=available,
        )
    return indices


def _normalize_lead_name(value: str) -> str:
    text = str(value).strip().upper()
    for token in (
        "ECG",
        "LEAD",
        "-REF",
        "_REF",
        "(MV)",
        "[MV]",
        "(UV)",
        "[UV]",
        " ",
        "_",
        "-",
    ):
        text = text.replace(token, "")
    aliases = {
        "AVR": "AVR",
        "AVL": "AVL",
        "AVF": "AVF",
        "VR": "AVR",
        "VL": "AVL",
        "VF": "AVF",
    }
    return aliases.get(text, text)


def _convert_units(signal: np.ndarray, units: tuple[str, ...]) -> np.ndarray:
    missing = tuple(
        LEAD_NAMES_12[index]
        for index, unit in enumerate(units)
        if _unit_factor_to_mv(unit) is None
    )
    if missing:
        raise ECGImportMappingRequired(
            "Physical units are missing or unsupported; specify mV, V or uV",
            missing_units=missing,
        )
    factors = np.asarray(
        [_unit_factor_to_mv(unit) for unit in units],
        dtype=float,
    )
    return np.asarray(signal, dtype=float) * factors[np.newaxis, :]


def _unit_factor_to_mv(value: str) -> Optional[float]:
    normalized = (
        str(value)
        .strip()
        .lower()
        .replace("microvolt", "uv")
        .replace("microvolts", "uv")
        .replace("µ", "u")
        .replace("μ", "u")
    )
    if normalized in ("mv", "millivolt", "millivolts"):
        return 1.0
    if normalized in ("v", "volt", "volts"):
        return 1000.0
    if normalized in ("uv", "microv"):
        return 0.001
    return None


def _unit_from_channel_name(value: str) -> str:
    upper = str(value).upper()
    if "(MV)" in upper or "[MV]" in upper or upper.endswith("_MV"):
        return "mV"
    if "(UV)" in upper or "[UV]" in upper or upper.endswith("_UV"):
        return "uV"
    if upper.endswith("_V"):
        return "V"
    return ""


def _require_sampling_rate(meta: ExternalECGMeta) -> float:
    if meta.sampling_rate is None or float(meta.sampling_rate) <= 0:
        raise ECGImportMappingRequired(
            "Sampling rate is required for this file format"
        )
    return float(meta.sampling_rate)


def _dicom_channel_name(definition) -> str:
    label = str(getattr(definition, "ChannelLabel", "") or "")
    if label:
        return label
    sequence = getattr(definition, "ChannelSourceSequence", ())
    if sequence:
        source = sequence[0]
        return str(
            getattr(source, "CodeMeaning", "")
            or getattr(source, "CodeValue", "")
        )
    return ""


def _dicom_channel_unit(definition) -> str:
    sequence = getattr(definition, "ChannelSensitivityUnitsSequence", ())
    if not sequence:
        return ""
    unit = sequence[0]
    return str(
        getattr(unit, "CodeValue", "")
        or getattr(unit, "CodeMeaning", "")
    )


def _validate_result(result: ExternalECGLoadResult) -> None:
    signal = np.asarray(result.signal_millivolts)
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError(f"Expected samples x 12 leads, got {signal.shape}")
    if signal.shape[0] < 2:
        raise ValueError("ECG contains fewer than two samples")
    if not np.isfinite(signal).all():
        raise ValueError("ECG contains non-finite samples")
    if result.sampling_rate_hz <= 0:
        raise ValueError("Sampling rate must be positive")
    if tuple(result.lead_names) != tuple(LEAD_NAMES_12):
        raise ValueError("External ECG leads are not in canonical order")


def _update_meta(meta: ExternalECGMeta, result: ExternalECGLoadResult) -> None:
    meta.sampling_rate = result.sampling_rate_hz
    meta.lead_names = list(result.lead_names)
    meta.units = list(result.source_units)
    meta.source_format = result.source_format
    meta.source_metadata = dict(result.source_metadata)


def make_external_record(signal: np.ndarray, meta: ExternalECGMeta) -> ECGRecord:
    sampling_rate = _require_sampling_rate(meta)
    return ECGRecord(
        ecg_id=0,
        patient_id=0,
        age=meta.age,
        sex=meta.sex,
        height=None,
        weight=None,
        recording_date=None,
        scp_codes={},
        report="",
        heart_axis=None,
        diagnostic_superclass=[],
        diagnostic_subclass=[],
        validated_by_human=False,
        strat_fold=0,
        signal_quality={},
        ecg_signal=np.asarray(signal, dtype=float),
        sampling_rate=int(round(sampling_rate)),
        r_peaks=None,
    )
