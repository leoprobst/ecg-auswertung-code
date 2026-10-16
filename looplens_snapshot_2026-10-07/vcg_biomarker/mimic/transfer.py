from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .config import MimicProjectConfig
from .label_registry import default_label_definitions
from .ptb_transfer import (
    DEFAULT_PTB_TRANSFER_RUN_ID,
    PTB_ENDPOINT_SPECS,
    load_ptb_frozen_transfer_scores,
)


TRANSFER_MATRIX_VERSION = "1.0.0"
TRANSFER_COLUMNS = (
    "transfer_type",
    "marker_or_score_id",
    "components",
    "panel_id",
    "ptb_endpoint_id",
    "ptb_endpoint_name",
    "ptb_reference_type",
    "ptb_reference_columns",
    "mimic_endpoint_id",
    "mimic_reference_type",
    "mimic_target_kind",
    "mimic_source_tables",
    "ptb_train_direction",
    "mimic_train_direction",
    "train_direction_agreement",
    "ptb_validation_direction",
    "mimic_validation_direction",
    "ptb_validation_direction_agreement",
    "mimic_validation_direction_agreement",
    "ptb_validation_rank_biserial",
    "mimic_validation_rank_biserial",
    "ptb_validation_roc_auc",
    "mimic_validation_roc_auc",
    "ptb_train_validation_auc_gap",
    "mimic_train_validation_auc_gap",
    "ptb_validation_missingness",
    "mimic_validation_missingness",
    "ptb_inference_status",
    "mimic_inference_status",
    "ptb_evaluation_split",
    "mimic_evaluation_split",
    "pooled_performance_permitted",
    "direct_auc_delta_permitted",
    "interpretation_note",
    "transfer_matrix_version",
)


def build_ptb_mimic_transfer_matrix(
    config: MimicProjectConfig,
    *,
    screening: pd.DataFrame,
    combinations: pd.DataFrame,
    stability: pd.DataFrame,
    ptb_run_id: str = DEFAULT_PTB_TRANSFER_RUN_ID,
    enabled: bool = True,
) -> pd.DataFrame:
    if not enabled:
        return _empty_transfer_matrix()
    load_ptb_frozen_transfer_scores(config, run_id=ptb_run_id)
    root = config.work_root / "ptb_revalidation" / ptb_run_id
    marker_path = root / "ptb_marker_results.parquet"
    score_path = root / "ptb_score_results.parquet"
    missing = [
        str(path)
        for path in (marker_path, score_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Frozen PTB development results are missing: {missing}"
        )
    return _build_transfer_matrix(
        pd.read_parquet(marker_path),
        pd.read_parquet(score_path),
        screening,
        combinations,
        stability,
    )


def _build_transfer_matrix(
    ptb_markers: pd.DataFrame,
    ptb_scores: pd.DataFrame,
    mimic_markers: pd.DataFrame,
    mimic_scores: pd.DataFrame,
    mimic_stability: pd.DataFrame,
) -> pd.DataFrame:
    definitions = {
        item.label_id: item for item in default_label_definitions()
    }
    rows: list[dict] = []
    for endpoint in PTB_ENDPOINT_SPECS:
        for mimic_endpoint_id in endpoint.mimic_endpoint_ids:
            definition = definitions.get(mimic_endpoint_id)
            if definition is None:
                continue
            reference = _mimic_reference(definition)
            for feature_id in endpoint.components:
                ptb_train = _find_row(
                    ptb_markers,
                    endpoint_id=endpoint.endpoint_id,
                    feature_id=feature_id,
                    evaluation_split="train",
                )
                ptb_validation = _find_row(
                    ptb_markers,
                    endpoint_id=endpoint.endpoint_id,
                    feature_id=feature_id,
                    evaluation_split="validation",
                )
                mimic_validation = _find_row(
                    mimic_markers,
                    endpoint_id=mimic_endpoint_id,
                    feature_id=feature_id,
                    evaluation_split="validation",
                )
                if ptb_validation is None or mimic_validation is None:
                    continue
                stability_row = _find_row(
                    mimic_stability,
                    endpoint_id=mimic_endpoint_id,
                    feature_id=feature_id,
                    analysis_type="single_marker",
                )
                rows.append(
                    _transfer_record(
                        transfer_type="shared_single_marker",
                        marker_or_score_id=feature_id,
                        components=feature_id,
                        panel_id=_text(mimic_validation.get("panel_id")),
                        endpoint=endpoint,
                        mimic_endpoint_id=mimic_endpoint_id,
                        reference=reference,
                        ptb_train=ptb_train,
                        ptb_validation=ptb_validation,
                        mimic_validation=mimic_validation,
                        mimic_stability=stability_row,
                    )
                )

            ptb_score_id = f"{endpoint.endpoint_id}__ptb_frozen_score"
            ptb_train_score = _find_row(
                ptb_scores,
                endpoint_id=endpoint.endpoint_id,
                feature_id=ptb_score_id,
                evaluation_split="train",
            )
            ptb_validation_score = _find_row(
                ptb_scores,
                endpoint_id=endpoint.endpoint_id,
                feature_id=ptb_score_id,
                evaluation_split="validation",
            )
            mimic_validation_score = _find_row(
                mimic_scores,
                endpoint_id=mimic_endpoint_id,
                ptb_endpoint_id=endpoint.endpoint_id,
                evaluation_split="validation",
                analysis_type="ptb_frozen_transfer",
            )
            if (
                ptb_validation_score is None
                or mimic_validation_score is None
            ):
                continue
            mimic_score_id = _text(
                mimic_validation_score.get("feature_id")
            )
            score_stability = _find_row(
                mimic_stability,
                endpoint_id=mimic_endpoint_id,
                feature_id=mimic_score_id,
                analysis_type="ptb_frozen_transfer",
            )
            rows.append(
                _transfer_record(
                    transfer_type="ptb_frozen_score",
                    marker_or_score_id=(
                        f"{endpoint.endpoint_id}__frozen_score"
                    ),
                    components=_text(
                        mimic_validation_score.get(
                            "components",
                            "|".join(endpoint.components),
                        )
                    ),
                    panel_id="ptb_frozen_score",
                    endpoint=endpoint,
                    mimic_endpoint_id=mimic_endpoint_id,
                    reference=reference,
                    ptb_train=ptb_train_score,
                    ptb_validation=ptb_validation_score,
                    mimic_validation=mimic_validation_score,
                    mimic_stability=score_stability,
                )
            )
    if not rows:
        return _empty_transfer_matrix()
    return (
        pd.DataFrame(rows, columns=TRANSFER_COLUMNS)
        .sort_values(
            [
                "ptb_endpoint_id",
                "mimic_endpoint_id",
                "transfer_type",
                "marker_or_score_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _transfer_record(
    *,
    transfer_type: str,
    marker_or_score_id: str,
    components: str,
    panel_id: str,
    endpoint,
    mimic_endpoint_id: str,
    reference: Mapping[str, str],
    ptb_train: Optional[Mapping],
    ptb_validation: Mapping,
    mimic_validation: Mapping,
    mimic_stability: Optional[Mapping],
) -> dict:
    ptb_train_direction = _integer(
        (ptb_train or {}).get(
            "direction",
            ptb_validation.get("direction"),
        )
    )
    mimic_train_direction = _integer(
        (mimic_stability or {}).get(
            "train_direction",
            mimic_validation.get("direction"),
        )
    )
    ptb_validation_direction = _direction_from_effect(
        ptb_validation.get("rank_biserial")
    )
    mimic_validation_direction = _integer(
        (mimic_stability or {}).get("validation_direction")
    )
    ptb_train_auc = _number((ptb_train or {}).get("roc_auc"))
    ptb_validation_auc = _number(ptb_validation.get("roc_auc"))
    return {
        "transfer_type": transfer_type,
        "marker_or_score_id": marker_or_score_id,
        "components": components,
        "panel_id": panel_id,
        "ptb_endpoint_id": endpoint.endpoint_id,
        "ptb_endpoint_name": endpoint.display_name,
        "ptb_reference_type": "PTB-XL structured ECG statement labels",
        "ptb_reference_columns": "|".join(endpoint.source_columns),
        "mimic_endpoint_id": mimic_endpoint_id,
        "mimic_reference_type": reference["reference_type"],
        "mimic_target_kind": reference["target_kind"],
        "mimic_source_tables": reference["source_tables"],
        "ptb_train_direction": ptb_train_direction,
        "mimic_train_direction": mimic_train_direction,
        "train_direction_agreement": _same_direction(
            ptb_train_direction,
            mimic_train_direction,
        ),
        "ptb_validation_direction": ptb_validation_direction,
        "mimic_validation_direction": mimic_validation_direction,
        "ptb_validation_direction_agreement": _same_direction(
            ptb_train_direction,
            ptb_validation_direction,
        ),
        "mimic_validation_direction_agreement": _optional_bool(
            (mimic_stability or {}).get("direction_agreement")
        ),
        "ptb_validation_rank_biserial": _number(
            ptb_validation.get("rank_biserial")
        ),
        "mimic_validation_rank_biserial": _number(
            mimic_validation.get("rank_biserial")
        ),
        "ptb_validation_roc_auc": ptb_validation_auc,
        "mimic_validation_roc_auc": _number(
            mimic_validation.get("roc_auc")
        ),
        "ptb_train_validation_auc_gap": (
            ptb_train_auc - ptb_validation_auc
            if ptb_train_auc is not None
            and ptb_validation_auc is not None
            else None
        ),
        "mimic_train_validation_auc_gap": _number(
            (mimic_stability or {}).get(
                "train_validation_auc_gap"
            )
        ),
        "ptb_validation_missingness": _number(
            ptb_validation.get("missing_fraction")
        ),
        "mimic_validation_missingness": _number(
            mimic_validation.get("missingness")
        ),
        "ptb_inference_status": _text(
            ptb_validation.get("inference_status")
        ),
        "mimic_inference_status": _text(
            mimic_validation.get("inference_status")
        ),
        "ptb_evaluation_split": "fold_9_validation",
        "mimic_evaluation_split": "validation",
        "pooled_performance_permitted": False,
        "direct_auc_delta_permitted": False,
        "interpretation_note": (
            "Reference endpoints differ. Compare direction, effect and "
            "stability only; AUCs are shown side by side and must not be "
            "pooled or subtracted."
        ),
        "transfer_matrix_version": TRANSFER_MATRIX_VERSION,
    }


def _mimic_reference(definition) -> dict[str, str]:
    target_kind = definition.target_kind.value
    return {
        "reference_type": f"MIMIC clinical {target_kind} endpoint",
        "target_kind": target_kind,
        "source_tables": "|".join(definition.source_tables),
    }


def _find_row(
    frame: pd.DataFrame,
    **conditions: str,
) -> Optional[dict]:
    if frame.empty or any(column not in frame for column in conditions):
        return None
    mask = np.ones(len(frame), dtype=bool)
    for column, value in conditions.items():
        mask &= frame[column].astype(str).eq(str(value)).to_numpy()
    matches = frame.loc[mask]
    return matches.iloc[0].to_dict() if not matches.empty else None


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _integer(value) -> Optional[int]:
    number = _number(value)
    return int(number) if number is not None else None


def _direction_from_effect(value) -> Optional[int]:
    effect = _number(value)
    if effect is None:
        return None
    return 1 if effect >= 0 else -1


def _same_direction(
    first: Optional[int],
    second: Optional[int],
) -> Optional[bool]:
    if first is None or second is None:
        return None
    return bool(first == second)


def _optional_bool(value) -> Optional[bool]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return bool(value)


def _text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _empty_transfer_matrix() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSFER_COLUMNS)
