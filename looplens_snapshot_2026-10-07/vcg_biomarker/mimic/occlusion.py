"""What the ST-elevation criterion misses, and what still sees it.

The OMI argument is that the threshold which decides who goes to the
catheterisation laboratory -- ST elevation in two contiguous leads -- was
chosen to be specific, and pays for it in occlusions that never meet it. That
claim is usually tested against a cardiologist's reading of the same ECG, which
cannot settle it: the reading and the criterion look at one tracing.

This project can do better, because ``omi_revascularization_proxy`` is built
from a troponin rise plus a PCI within a day plus a coronary presentation --
none of which is the ECG. The criterion and the reference are independent, so
the question "how much does the criterion miss" has an answer here.

**Historical warning.** The numerical examples retained below came from the
aborted legacy run before one ECG per acute encounter was enforced. They are
implementation clues only, not reportable estimates; the next pilot must
replace every number under the episode-index design implemented here.

**Measured**, over 21.462 ECGs with an adjudicable endpoint, of which 2.120
positive:

========================  =========  =========  ========  ========
Kriterium                 n positiv  Sens. %    Spez. %   PPW %
========================  =========  =========  ========  ========
st_elevation_contiguous       2.071       18,7      91,3      19,1
st_depression_contiguous      7.290       33,1      65,9       9,6
t_inversion_contiguous        4.727       21,2      77,9       9,5
wellens_ecg_pattern           1.317        7,9      94,1      12,8
posterior_mi_pattern            401        2,1      98,2      11,2
de_winter_ecg_pattern            87        0,8      99,6      18,4
pathological_q_waves         17.434       87,2      19,4      10,6
========================  =========  =========  ========  ========

**1.724 of the 2.120 events -- 81,3 % -- carry no ST-elevation criterion.** The
three OMI-equivalent patterns recover 202 of them, 11,7 %: real, and small
against what is left.

*This number is not "STEMI criteria miss 81 % of occlusions", and must never be
reported as though it were.* The endpoint is a revascularization proxy, and its
own summary says so: "no angiographic occlusion data". Many of these patients
had a culprit lesion opened without a total occlusion, which is exactly the
population a STEMI criterion is not built to flag. The defensible statement is
narrower and still worth making: among patients with an acute troponin rise who
reached the catheterisation laboratory within a day, fewer than one in five met
the criterion on the ECG recorded around that event.

**The headline does not depend on how missingness is counted.** 11,2 % of ECGs
cannot be assessed for the ST criteria and 20,7 % for the pattern criteria, and
the two defensible readings -- count them as "criterion not met", or drop them
from the denominator -- give:

============================  =========  =======  =======  ========
Lesart                        bewertet   Sens.    Spez.    verfehlt
============================  =========  =======  =======  ========
nicht beurteilbar = negativ   21.462     18,7 %   91,3 %   81,3 %
nicht beurteilbar ausgelassen 19.051     19,6 %   90,2 %   80,4 %
============================  =========  =======  =======  ========

Removing 2.411 recordings moves the answer by 0,9 points. The choice is
therefore not carrying the finding, which is the only reason it is safe to have
a default at all.

**A second finding, about this project rather than about cardiology.**
``classical_pathological_q_waves`` is positive in 17.434 of 21.462 ECGs -- 81 %
of an unselected hospital population -- for a specificity of 19,4 %. A rule
that fires in four patients out of five is not implementing the pathological Q
wave; it is measuring something else. It is left in place here and reported as
measured, because that is what the comparison needs, but it should not be read
as the classical criterion until it has been checked.

**What the VCG adds.** Discrimination among the ST-elevation-negative, against
the same endpoint:

==================================  ===========  ==================
Marker                              alle         ST-Hebung negativ
==================================  ===========  ==================
spatial_qrs_t_angle_deg             0,600        0,585
st_vector_to_tpeak_angle_deg        0,555        0,578
qrs_t_hausdorff_distance            0,582        0,568
st_vector_magnitude                 0,517        0,551
==================================  ===========  ==================

The point is not the height of these numbers, which is modest. It is that
removing every ST-elevation-positive ECG barely moves them, and raises two of
them. A marker that was merely re-detecting ST elevation would collapse here.
These do not, so whatever they measure is not the criterion -- which is the
only property that makes a marker useful in the patients the criterion misses.

The same holds on the validation split by itself (5.572 ECGs, 557 events):
0,610 to 0,598 for the spatial QRS-T angle, 0,592 to 0,582 for the Hausdorff
distance, and 0,520 to 0,570 for the ST vector magnitude. So the pattern is not
an artefact of having looked at everything at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from .panels import STUDY_PANEL_IDS


OCCLUSION_VERSION = "1.3.0"

# The criterion that decides the catheterisation laboratory today.
STEMI_CRITERION = "classical_st_elevation_contiguous_positive"

# Patterns taught as "STEMI equivalents": an occlusion that does not raise ST.
OMI_EQUIVALENT_PATTERNS: tuple[str, ...] = (
    "classical_de_winter_ecg_pattern_positive",
    "classical_wellens_ecg_pattern_positive",
    "classical_posterior_mi_pattern_positive",
)

# Everything classical that speaks to an acute coronary event, for the table.
CLASSICAL_CRITERIA: tuple[str, ...] = (
    STEMI_CRITERION,
    "classical_st_depression_contiguous_positive",
    "classical_t_inversion_contiguous_positive",
    *OMI_EQUIVALENT_PATTERNS,
    "classical_pathological_q_waves_positive",
)

OCCLUSION_ENDPOINT = "omi_revascularization_proxy"


@dataclass(frozen=True)
class OcclusionPolicy:
    """Everything that decides an occlusion-yield number."""

    endpoint: str = OCCLUSION_ENDPOINT
    criterion: str = STEMI_CRITERION
    minimum_events: int = 50
    treat_unassessable_as_negative: bool = True

    def to_dict(self) -> dict:
        return {
            "endpoint": str(self.endpoint),
            "criterion": str(self.criterion),
            "minimum_events": int(self.minimum_events),
            "treat_unassessable_as_negative": bool(
                self.treat_unassessable_as_negative
            ),
            "version": OCCLUSION_VERSION,
        }


def _flag(frame: pd.DataFrame, column: str, *, unassessable_negative: bool):
    """A criterion column, its assessable mask, and its missing share.

    "The criterion is not met" and "the criterion could not be assessed" are
    different statements, and 11,2 % of ECGs here fall in the second for the
    ST criteria, 20,7 % for the pattern criteria. Folding them into negative
    is what a clinician effectively does at the bedside -- no elevation was
    seen -- so it is the default.

    **The mask is the whole point of the parameter.** The first version
    returned only a flag, and both branches computed the same thing:
    ``raw.fillna(False) & raw.notna()`` is ``raw.fillna(False)``, because a
    missing value is already ``False`` by the time the second term is applied.
    Setting ``treat_unassessable_as_negative=False`` therefore changed
    nothing, while the setting was written into the policy and the summary as
    though it were a choice. Excluding an unassessable ECG means removing it
    from the *denominator*, which a flag alone cannot express.
    """

    if column not in frame.columns:
        empty = pd.Series(False, index=frame.index)
        return empty, empty.copy(), 0.0
    raw = frame[column]
    assessable = raw.notna()
    missing = float((~assessable).mean())
    flag = raw.astype("boolean").fillna(False).astype(bool)
    if unassessable_negative:
        # Every ECG stays in the denominator; a missing criterion counts as
        # not met.
        return flag, pd.Series(True, index=frame.index), missing
    return flag, assessable, missing


def criterion_yield(
    frame: pd.DataFrame,
    *,
    criteria: Sequence[str] = CLASSICAL_CRITERIA,
    endpoint_column: str = "occlusion",
    policy: OcclusionPolicy = OcclusionPolicy(),
) -> pd.DataFrame:
    """Sensitivity, specificity and yield of each classical criterion.

    Positive predictive value is reported because it is the number that
    justifies a criterion in practice, and it is the one that exposes a rule
    firing on four patients in five: such a rule has a PPV equal to the
    prevalence, which is how ``pathological_q_waves`` gives itself away here.
    """

    columns = [
        "criterion", "n_assessed", "n_positive", "n_unassessable_pct",
        "sensitivity", "specificity", "positive_predictive_value",
    ]
    if frame is None or len(frame) == 0 or endpoint_column not in frame:
        return pd.DataFrame(columns=columns)
    outcome = pd.to_numeric(frame[endpoint_column], errors="coerce").fillna(0).astype(bool)
    if int(outcome.sum()) < int(policy.minimum_events):
        return pd.DataFrame(columns=columns)

    records: list[dict] = []
    for name in criteria:
        flag, assessable, missing = _flag(
            frame, name,
            unassessable_negative=policy.treat_unassessable_as_negative,
        )
        # Everything below is computed inside the assessable set. With the
        # default that set is everyone, so the numbers are unchanged; with the
        # strict setting the unassessable leave the denominator instead of
        # padding the specificity.
        flag = flag & assessable
        positives = outcome & assessable
        negatives = (~outcome) & assessable
        true_positive = int((flag & positives).sum())
        records.append(
            {
                "criterion": str(name),
                "n_assessed": int(assessable.sum()),
                "n_positive": int(flag.sum()),
                "n_unassessable_pct": 100.0 * missing,
                "sensitivity": 100.0 * true_positive / max(int(positives.sum()), 1),
                "specificity": 100.0 * int((~flag & negatives).sum())
                / max(int(negatives.sum()), 1),
                "positive_predictive_value": 100.0 * true_positive
                / max(int(flag.sum()), 1),
            }
        )
    return pd.DataFrame(records)


def missed_by_criterion(
    frame: pd.DataFrame,
    *,
    endpoint_column: str = "occlusion",
    equivalents: Sequence[str] = OMI_EQUIVALENT_PATTERNS,
    policy: OcclusionPolicy = OcclusionPolicy(),
) -> dict:
    """How many events the criterion misses, and how many patterns recover."""

    if frame is None or len(frame) == 0 or endpoint_column not in frame:
        return {"n_events": 0, "n_missed": 0, "version": OCCLUSION_VERSION}
    outcome = pd.to_numeric(frame[endpoint_column], errors="coerce").fillna(0).astype(bool)
    criterion, assessable, _ = _flag(
        frame, policy.criterion,
        unassessable_negative=policy.treat_unassessable_as_negative,
    )
    outcome = outcome & assessable
    criterion = criterion & assessable
    missed = outcome & ~criterion

    rescued = pd.Series(False, index=frame.index)
    for name in equivalents:
        pattern, pattern_assessable, _ = _flag(
            frame, name,
            unassessable_negative=policy.treat_unassessable_as_negative,
        )
        rescued = rescued | (pattern & pattern_assessable)
    recovered = int((missed & rescued).sum())
    n_missed = int(missed.sum())
    return {
        "version": OCCLUSION_VERSION,
        "endpoint": str(policy.endpoint),
        "criterion": str(policy.criterion),
        "n_evaluable": int(assessable.sum()),
        "n_events": int(outcome.sum()),
        "n_missed": n_missed,
        "missed_share_pct": 100.0 * n_missed / max(int(outcome.sum()), 1),
        "n_recovered_by_equivalents": recovered,
        "recovered_share_pct": 100.0 * recovered / max(n_missed, 1),
        # The caveat travels with the number, so it cannot be quoted without.
        "reference_note": (
            "Revaskularisations-Surrogat ohne angiographischen "
            "Verschlussnachweis; nicht als versaeumte Verschluesse lesbar"
        ),
    }


def discrimination_without_the_criterion(
    frame: pd.DataFrame,
    markers: Sequence[str],
    *,
    endpoint_column: str = "occlusion",
    policy: OcclusionPolicy = OcclusionPolicy(),
) -> pd.DataFrame:
    """Marker discrimination with, and then without, the criterion-positive.

    The second column is the one that matters. A marker that merely
    re-expresses ST elevation loses its discrimination once every
    criterion-positive ECG is removed; one that carries independent
    information does not. ``retained`` is the ratio of the two margins over
    chance, so 1,0 means the criterion contributed nothing to it.
    """

    columns = [
        "marker", "n_all", "auc_all", "n_criterion_negative",
        "n_events_criterion_negative", "auc_criterion_negative",
        "retained", "direction",
    ]
    if frame is None or len(frame) == 0 or endpoint_column not in frame:
        return pd.DataFrame(columns=columns)
    from sklearn.metrics import roc_auc_score

    outcome = pd.to_numeric(frame[endpoint_column], errors="coerce").fillna(0).astype(bool)
    criterion, assessable, _ = _flag(
        frame, policy.criterion,
        unassessable_negative=policy.treat_unassessable_as_negative,
    )
    criterion_negative = (~criterion) & assessable
    negative = frame.loc[criterion_negative]
    negative_outcome = outcome.loc[criterion_negative]

    records: list[dict] = []
    for marker in markers:
        if marker not in frame.columns:
            continue
        full = pd.to_numeric(frame[marker], errors="coerce")
        usable = full.notna()
        if int(outcome[usable].sum()) < int(policy.minimum_events):
            continue
        auc_all = float(roc_auc_score(outcome[usable], full[usable]))

        restricted = pd.to_numeric(negative[marker], errors="coerce")
        usable_restricted = restricted.notna()
        events = int(negative_outcome[usable_restricted].sum())
        if events < int(policy.minimum_events):
            continue
        auc_negative = float(
            roc_auc_score(
                negative_outcome[usable_restricted], restricted[usable_restricted]
            )
        )
        # Orientation is a property of the marker, not of the population, so
        # it is fixed once on the full cohort and carried over. Flipping it
        # per population would let a marker "keep" its discrimination by
        # reversing direction, which is not the same finding at all.
        direction = 1.0 if auc_all >= 0.5 else -1.0
        margin_all = abs(auc_all - 0.5)
        margin_negative = (auc_negative - 0.5) * direction
        records.append(
            {
                "marker": str(marker),
                "n_all": int(usable.sum()),
                "auc_all": auc_all if direction > 0 else 1.0 - auc_all,
                "n_criterion_negative": int(usable_restricted.sum()),
                "n_events_criterion_negative": events,
                "auc_criterion_negative": (
                    auc_negative if direction > 0 else 1.0 - auc_negative
                ),
                "retained": (
                    margin_negative / margin_all if margin_all > 1e-12 else np.nan
                ),
                "direction": "+" if direction > 0 else "-",
            }
        )
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records).sort_values(
        "auc_criterion_negative", ascending=False, ignore_index=True
    )


def occlusion_frame_query(
    master_sql: str,
    labels_sql: str,
    feature_clause: str,
    *,
    marker_columns: str,
    policy: OcclusionPolicy = OcclusionPolicy(),
    splits: Sequence[str] = ("train", "validation"),
) -> str:
    """First eligible ECG per acute clinical episode with an adjudicated endpoint.

    ``indeterminate`` and ``unavailable`` are excluded rather than folded into
    the negatives: the endpoint marks partial evidence as indeterminate on
    purpose, and counting those as "no occlusion" would put the uncertain cases
    on the side that inflates every specificity in the table.
    """

    split_list = ", ".join(f"'{value}'" for value in splits)
    state = f"{policy.endpoint}_state"
    indexed_marker_columns = marker_columns.replace("f.", "i.")
    return f"""
        WITH eligible AS (
            SELECT
                m.subject_id, m.study_id, m.split, m.ecg_time,
                COALESCE(
                    'hadm:' || CAST(m.hadm_id AS VARCHAR),
                    'ed:' || CAST(m.ed_stay_id AS VARCHAR),
                    'date:' || CAST(CAST(m.ecg_time AS DATE) AS VARCHAR)
                ) AS clinical_episode_id,
                {marker_columns}
            FROM read_parquet('{master_sql}') m
            JOIN {feature_clause} f
              ON f.subject_id = m.subject_id AND f.study_id = m.study_id
            WHERE f.extraction_status = 'ok'
              AND m.split IN ({split_list})
        ),
        indexed AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY subject_id, clinical_episode_id
                    ORDER BY ecg_time, study_id
                ) AS episode_ecg_rank
            FROM eligible
        )
        SELECT
            i.subject_id, i.study_id, i.split, i.ecg_time,
            i.clinical_episode_id,
            l.{state} AS endpoint_state,
            CASE WHEN l.{state} = 'positive' THEN 1 ELSE 0 END AS occlusion,
            {indexed_marker_columns}
        FROM indexed i
        JOIN read_parquet('{labels_sql}') l
          ON l.subject_id = i.subject_id AND l.study_id = i.study_id
        WHERE l.{state} IN ('positive', 'negative')
          AND i.episode_ecg_rank = 1
    """


# The markers carried into the occlusion arm. Chosen for what they measure --
# the spatial relationship between depolarisation and repolarisation, and the
# ST vector itself -- not for how they screened, since the whole question here
# is whether they see something the ST criterion does not.
OCCLUSION_MARKERS: tuple[str, ...] = (
    "spatial_qrs_t_angle_deg",
    "st_vector_to_tpeak_angle_deg",
    "qrs_t_hausdorff_distance",
    "st_vector_magnitude",
    "st_loop_area_best_fit_plane",
    "svg_magnitude",
)


def run_occlusion_yield(
    config,
    *,
    run_id: str,
    panel_ids: Sequence[str] = STUDY_PANEL_IDS,
    markers: Sequence[str] = OCCLUSION_MARKERS,
    policy: OcclusionPolicy = OcclusionPolicy(),
    feature_glob: Optional[str] = None,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
):
    """What the ST criterion catches, what it misses, and what still sees it."""

    import duckdb

    from .storage import (
        StageManifest,
        atomic_write_json,
        atomic_write_parquet,
        capture_output_integrity,
        content_hash,
        development_feature_glob,
        feature_manifest_path,
        feature_parquet_clause,
        input_artifact_fingerprints,
        scientific_config_hash,
    )

    output_root = config.runs_root / run_id / "occlusion"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "occlusion_yield.manifest.json"

    marker_ids = tuple(dict.fromkeys(str(item) for item in markers))
    resolved_glob = feature_glob or development_feature_glob(
        config.features_root, panel_ids
    )
    feature_manifest = feature_manifest_path(resolved_glob)
    upstream = input_artifact_fingerprints(
        {
            "feature_manifest": feature_manifest,
            "feature_partition_index": feature_manifest.parent
            / "partition_index.json",
            "master_index_manifest": config.master_root
            / "master_index.manifest.json",
            "labels_manifest": config.labels_root / "labels.manifest.json",
        }
    )
    input_hash = content_hash(
        {
            "run_id": run_id,
            "feature_glob": resolved_glob,
            "markers": list(marker_ids),
            "criteria": list(CLASSICAL_CRITERIA),
            "policy": policy.to_dict(),
            "upstream": upstream,
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, OCCLUSION_VERSION):
            return previous

    labels = config.labels_root / "clinical_labels_wide.parquet"
    if not labels.exists():
        raise FileNotFoundError(
            f"Fuer die Verschluss-Auswertung fehlen die Labels: {labels}"
        )

    def sql_path(value) -> str:
        return str(value).replace(chr(92), "/")

    if progress:
        progress("Verschluss: Kohorte mit entschiedenem Endpunkt bilden")

    columns = list(CLASSICAL_CRITERIA) + list(marker_ids)
    from .storage import isolated_analysis_connection

    with isolated_analysis_connection(config) as connection:
        clause = feature_parquet_clause(connection, sql_path(resolved_glob))
        available = set(
            connection.execute(f"SELECT * FROM {clause} LIMIT 0").df().columns
        )
        present = [name for name in columns if name in available]
        frame = connection.execute(
            occlusion_frame_query(
                sql_path(config.master_root / "ecg_master_index.parquet"),
                sql_path(labels),
                clause,
                marker_columns=", ".join(f"f.{name}" for name in present),
                policy=policy,
            )
        ).df()

    if progress:
        progress(
            f"Verschluss: {len(frame):,} EKGs, "
            f"{int(frame['occlusion'].sum()) if len(frame) else 0:,} Ereignisse"
        )

    yield_table = criterion_yield(frame, policy=policy)
    summary = missed_by_criterion(frame, policy=policy)
    summary.update(
        {
            "run_id": str(run_id),
            "policy": policy.to_dict(),
            "n_markers": len([m for m in marker_ids if m in present]),
            "n_patients": (
                int(frame["subject_id"].nunique()) if len(frame) else 0
            ),
            "n_clinical_episodes": (
                int(
                    frame[["subject_id", "clinical_episode_id"]]
                    .drop_duplicates()
                    .shape[0]
                )
                if len(frame)
                else 0
            ),
            "observational_unit": (
                "first_eligible_ecg_per_admission_ed_visit_or_patient_day"
            ),
        }
    )
    if len(yield_table):
        yield_table["observational_unit"] = "clinical_episode_index_ecg"
    discrimination = discrimination_without_the_criterion(
        frame, [m for m in marker_ids if m in present], policy=policy
    )
    if len(discrimination):
        discrimination["observational_unit"] = "clinical_episode_index_ecg"

    yield_path = output_root / "criterion_yield.parquet"
    discrimination_path = output_root / "criterion_negative_discrimination.parquet"
    summary_path = output_root / "occlusion_yield_summary.json"
    atomic_write_parquet(yield_table, yield_path)
    atomic_write_parquet(discrimination, discrimination_path)
    atomic_write_json(summary_path, summary)

    manifest = StageManifest.complete(
        stage="occlusion-yield",
        stage_version=OCCLUSION_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(yield_path.resolve()),
            str(discrimination_path.resolve()),
            str(summary_path.resolve()),
        ),
        row_counts={
            "criteria": int(len(yield_table)),
            "markers": int(len(discrimination)),
            "events": int(summary.get("n_events", 0)),
        },
        output_integrity=capture_output_integrity(
            (yield_path, discrimination_path, summary_path)
        ),
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "CLASSICAL_CRITERIA",
    "OCCLUSION_ENDPOINT",
    "OCCLUSION_VERSION",
    "OMI_EQUIVALENT_PATTERNS",
    "OCCLUSION_MARKERS",
    "OcclusionPolicy",
    "STEMI_CRITERION",
    "criterion_yield",
    "discrimination_without_the_criterion",
    "missed_by_criterion",
    "occlusion_frame_query",
    "run_occlusion_yield",
]
