"""Known physiology, declared in advance, checked in every run.

The screen tests roughly 1.9 million associations per axis. Nothing tested
**the screen**. That gap is why a null result could not be read: on the
2026-08-28 run ``severe_hyperkalemia_near_ecg`` came back at +0.0006
incremental AUC over 1,026 cases, which would be a strong finding -- except
the marker paired with it was ``t_loop_width_major`` while
``classical_peaked_t_wave_heuristic`` sat unused in the same catalog. The
number said nothing about the ECG and everything about the pairing.

A control panel closes that. Two kinds, and they answer different questions:

**Positive controls** are associations physiology says must exist. Hyperkalemia
peaks the T wave; hypocalcaemia prolongs the QT; digoxin scoops the ST segment.
If the pipeline cannot recover these, its negative findings carry no weight --
and that has to be known before a paper is written, not after. If it can, that
is the most persuasive line in the results.

**Negative controls** are pairs with no described mechanism. They estimate the
*real* false-positive rate, which an FDR calculation cannot: Benjamini-Hochberg
controls the error rate under the null, but confounding by illness severity is
not the null. In an intensive-care population almost everything correlates with
being sick, and a negative control is the only instrument here that measures
how much of that leaks through.

**Every entry is fixed before the run and carries its mechanism.** A control
chosen after seeing results is not a control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


CONTROL_PANEL_VERSION = "1.1.0"

# A positive control asks whether the machinery *sees* known physiology, not
# whether the effect is large. Measured 2026-08-30 on the real screen: QTc is
# 3.4 ms higher in patients exposed to QT-prolonging drugs (436.4 vs 433.0,
# p = 2.4e-05, n = 2,912 vs 6,727) against a population SD of 44.5 ms. That is
# a correct, highly significant, tiny effect -- exactly what a mixed "any
# exposure" label should produce, and it fails any threshold on effect size.
# Judging a control by magnitude would have declared a working pipeline broken.
CONTROL_ALPHA = 0.01

# Effect floors, used to describe magnitude and to judge negative controls --
# never to judge whether a positive control was recovered.
CONTROL_MINIMUM_ABS_RHO = 0.10
CONTROL_MINIMUM_AUC_MARGIN = 0.05

DIRECTION_UP = 1
DIRECTION_DOWN = -1
DIRECTION_NONE = 0


@dataclass(frozen=True)
class Control:
    """One prespecified expectation, with the reason it is expected."""

    control_id: str
    kind: str  # "positive" or "negative"
    endpoint_id: str
    feature_id: str
    expected_direction: int
    mechanism: str
    reference: str = ""

    def to_dict(self) -> dict:
        return {
            "control_id": self.control_id,
            "kind": self.kind,
            "endpoint_id": self.endpoint_id,
            "feature_id": self.feature_id,
            "expected_direction": int(self.expected_direction),
            "mechanism": self.mechanism,
            "reference": self.reference,
        }


# --------------------------------------------------------------- the panel

POSITIVE_CONTROLS: tuple[Control, ...] = (
    Control(
        "hyperkalemia_peaked_t",
        "positive",
        "severe_hyperkalemia_near_ecg",
        "classical_peaked_t_wave_heuristic",
        DIRECTION_UP,
        "Potassium raises the resting membrane potential and steepens phase 3 "
        "repolarisation, which narrows and peaks the T wave. It is the first "
        "and most reliable ECG change of hyperkalaemia.",
        "Surawicz B. Relationship between electrocardiogram and electrolytes. "
        "Am Heart J 1967;73:814-834.",
    ),
    Control(
        "hyperkalemia_qrs_widening",
        "positive",
        "severe_hyperkalemia_near_ecg",
        "qrs_duration_ms",
        DIRECTION_UP,
        "Above roughly 6.5 mmol/l sodium-channel availability falls and "
        "conduction slows, widening the QRS. Later than the T-wave change, so "
        "a weaker control -- but it must not point the other way.",
        "Surawicz B. Am Heart J 1967;73:814-834.",
    ),
    Control(
        "hypokalemia_tpeak_tend",
        "positive",
        "severe_hypokalemia_near_ecg",
        "tpeak_tend_ms",
        DIRECTION_UP,
        "Potassium depletion prolongs repolarisation and disperses it across "
        "the ventricular wall, which widens the interval from T peak to T end "
        "and brings out the U wave.",
        "Diercks DB et al. Electrocardiographic manifestations: electrolyte "
        "abnormalities. J Emerg Med 2004;27:153-160.",
    ),
    Control(
        "hypocalcemia_qt",
        "positive",
        "hypocalcemia_near_ecg",
        "qtc_fridericia_ms",
        DIRECTION_UP,
        "Calcium sets the duration of the plateau phase. Lowering it lengthens "
        "phase 2 and therefore the QT interval, with the ST segment stretched "
        "and the T wave largely unchanged.",
        "Diercks DB et al. J Emerg Med 2004;27:153-160.",
    ),
    Control(
        "qt_prolonging_drug_qt",
        "positive",
        "qt_prolonging_drug_exposure",
        "qtc_fridericia_ms",
        DIRECTION_UP,
        "Drugs on the CredibleMeds list block the rapid delayed-rectifier "
        "current. Prolonged QT is the definition of the exposure, so this is "
        "the strongest control in the panel -- if it fails, the QT measurement "
        "itself is broken.",
        "Woosley RL et al. www.crediblemeds.org",
    ),
    Control(
        "digoxin_st_depression",
        "positive",
        "digoxin_exposure_near_ecg",
        "classical_st_depression_contiguous_mm",
        DIRECTION_UP,
        "Digoxin shortens the plateau and produces the down-sloping, scooped "
        "ST depression that carries its name. Present at therapeutic levels, "
        "so exposure alone should show it.",
        "Ma G et al. Electrocardiographic manifestations: digitalis toxicity. "
        "J Emerg Med 2001;20:145-152.",
    ),
)

# Pairs with no described mechanism. A hit here is confounding, not chance --
# which is exactly what makes them worth measuring, because an FDR correction
# cannot see confounding at all.
NEGATIVE_CONTROLS: tuple[Control, ...] = (
    Control(
        "hypocalcemia_p_width",
        "negative",
        "hypocalcemia_near_ecg",
        "p_loop_width_major",
        DIRECTION_NONE,
        "Calcium acts on the ventricular plateau. No mechanism connects it to "
        "the width of the atrial depolarisation loop.",
    ),
    Control(
        "digoxin_p_duration",
        "negative",
        "digoxin_exposure_near_ecg",
        "p_duration_ms",
        DIRECTION_NONE,
        "Digitalis acts on repolarisation and on the AV node. Atrial "
        "depolarisation duration is not among its described effects.",
    ),
    Control(
        "hypomagnesemia_qrs_axis",
        "negative",
        "hypomagnesemia_near_ecg",
        "qrs_loop_frontal_xy_signed_area",
        DIRECTION_NONE,
        "Magnesium affects repolarisation and arrhythmic threshold. The "
        "orientation of the depolarisation loop is anatomy, not electrolyte.",
    ),
    Control(
        "qt_drug_p_width",
        "negative",
        "qt_prolonging_drug_exposure",
        "p_loop_width_major",
        DIRECTION_NONE,
        "Potassium-channel blockade lengthens ventricular repolarisation. It "
        "has no described effect on atrial loop width.",
    ),
)

CONTROL_PANEL: tuple[Control, ...] = POSITIVE_CONTROLS + NEGATIVE_CONTROLS


def neutral_effect(target_kind: object) -> float:
    """The value an effect takes when there is nothing to find."""

    return 0.5 if str(target_kind) == "binary" else 0.0


def minimum_effect(target_kind: object) -> float:
    return (
        CONTROL_MINIMUM_AUC_MARGIN
        if str(target_kind) == "binary"
        else CONTROL_MINIMUM_ABS_RHO
    )


def evaluate_controls(
    results: pd.DataFrame,
    *,
    panel: Sequence[Control] = CONTROL_PANEL,
    endpoint_column: str = "endpoint_id",
    feature_column: str = "feature_id",
    effect_column: str = "effect",
    kind_column: Optional[str] = "target_kind",
    default_target_kind: str = "binary",
    p_value_column: Optional[str] = "p_value",
    split_column: Optional[str] = "evaluation_split",
    alpha: float = CONTROL_ALPHA,
) -> pd.DataFrame:
    """Check each declared control against what a screen actually produced.

    Accepts any result table that names an endpoint, a marker and an effect,
    so the same panel can be read against the broad-axis screen, the phenotype
    screen or the adjusted models without a second copy of the expectations.

    **One row per control and split.** A screening table holds the same
    endpoint-marker pair twice, once for training and once for validation, and
    the first version of this function took ``match.iloc[0]`` -- which is the
    training row. It reported ``hyperkalemia_peaked_t`` at AUC 0,568 with no
    column saying where that came from, while the validation value was 0,543.
    A reader would have taken the certificate of the run's honesty for a
    held-out result when it was measured on the data the thresholds were
    chosen on.

    The shrinkage between the two is worth having in its own right: a control
    that survives the move from training to validation is a different kind of
    evidence from one that halves.
    """

    has_split = bool(
        split_column
        and results is not None
        and not results.empty
        and split_column in results.columns
    )
    splits: tuple[Optional[str], ...] = (None,)
    if has_split:
        found_splits = [
            str(value)
            for value in results[split_column].dropna().unique()
        ]
        # Validation last so a reader scanning the table ends on the number
        # that carries the weight.
        splits = tuple(
            sorted(found_splits, key=lambda name: (name == "validation", name))
        ) or (None,)

    records: list[dict] = []
    for control in panel:
        for split in splits:
            row = {
                **control.to_dict(),
                "evaluation_split": split if split is not None else "all",
                "observed_effect": float("nan"),
                "observed_direction": 0,
                "observed_p_value": float("nan"),
                "effect_above_floor": False,
                "target_kind": default_target_kind,
                "found": False,
                "meets_expectation": False,
                "status": "not_tested",
            }
            if results is not None and not results.empty:
                selector = (
                    results[endpoint_column].astype(str).eq(control.endpoint_id)
                    & results[feature_column].astype(str).eq(control.feature_id)
                )
                if split is not None:
                    selector = selector & results[split_column].astype(str).eq(
                        split
                    )
                match = results.loc[selector]
                if not match.empty:
                    item = match.iloc[0]
                    kind = (
                        str(item[kind_column])
                        if kind_column and kind_column in match.columns
                        else default_target_kind
                    )
                    effect = pd.to_numeric(item[effect_column], errors="coerce")
                    row["target_kind"] = kind
                    row["observed_effect"] = float(effect)
                    row["found"] = True
                    if p_value_column and p_value_column in match.columns:
                        row["observed_p_value"] = float(
                            pd.to_numeric(item[p_value_column], errors="coerce")
                        )
                    if np.isfinite(effect):
                        delta = float(effect) - neutral_effect(kind)
                        # Direction is the sign, whatever the size. Magnitude
                        # is reported separately so a small real effect is not
                        # mistaken for no effect.
                        row["observed_direction"] = int(np.sign(delta))
                        row["effect_above_floor"] = bool(
                            abs(delta) >= minimum_effect(kind) - 1e-12
                        )
            records.append(row)

    table = pd.DataFrame(records)
    if table.empty:
        return table

    def verdict(row: Mapping) -> tuple[bool, str]:
        if not row["found"]:
            return False, "not_tested"
        if row["kind"] == "positive":
            # Direction and significance. Not magnitude: see CONTROL_ALPHA.
            if row["observed_direction"] != row["expected_direction"]:
                if row["observed_direction"] == 0:
                    return False, "missed"
                return False, "wrong_direction"
            significant = (
                np.isfinite(row["observed_p_value"])
                and float(row["observed_p_value"]) <= float(alpha)
            )
            if significant:
                return True, "recovered"
            if not np.isfinite(row["observed_p_value"]):
                return False, "no_uncertainty_reported"
            return False, "direction_only"
        # A negative control has no mechanism, so whatever it shows is
        # confounding. Judged by magnitude, because at these sample sizes
        # confounding is always significant.
        if not bool(row["effect_above_floor"]):
            return True, "silent_as_expected"
        return False, "false_positive"

    verdicts = [verdict(row) for _, row in table.iterrows()]
    table["meets_expectation"] = [item[0] for item in verdicts]
    table["status"] = [item[1] for item in verdicts]
    return table


# The split the verdict is taken on. Training is where the thresholds and
# directions were chosen, so a control recovered there says the screen can
# fit; recovered on validation, it says the screen found something.
VERDICT_SPLIT = "validation"


def _verdict_rows(table: pd.DataFrame, split: str = VERDICT_SPLIT) -> pd.DataFrame:
    """The rows the verdict is taken on, falling back when there is one split.

    A table produced from a source without an ``evaluation_split`` column
    carries the sentinel ``"all"``; judging that is the only thing available
    and is correct. What must never happen again is judging the training rows
    while presenting the number as though it were held out.
    """

    if table is None or table.empty or "evaluation_split" not in table.columns:
        return table if table is not None else pd.DataFrame()
    preferred = table.loc[table["evaluation_split"].astype(str).eq(split)]
    if not preferred.empty:
        return preferred
    remaining = {
        str(value) for value in table["evaluation_split"].dropna().unique()
    }
    if remaining <= {"all"}:
        return table
    # There are splits, but not the one the verdict wants. Judging the rest
    # would be exactly the defect this function exists to prevent.
    return table.iloc[0:0]


def effect_shrinkage(table: pd.DataFrame) -> pd.DataFrame:
    """How far each control's effect falls from training to validation.

    Worth its own table: a control that keeps its effect across the split is
    evidence the screen generalises, and one that loses most of it is evidence
    the training number was partly fit. Neither is visible from one split.
    """

    columns = [
        "control_id",
        "kind",
        "train_effect",
        "validation_effect",
        "shrinkage",
        "train_margin",
        "shrinkage_defined",
    ]
    if (
        table is None
        or table.empty
        or "evaluation_split" not in table.columns
    ):
        return pd.DataFrame(columns=columns)
    split = table["evaluation_split"].astype(str)
    train = table.loc[split.eq("train")].set_index("control_id")
    held = table.loc[split.eq(VERDICT_SPLIT)].set_index("control_id")
    shared = [name for name in train.index if name in held.index]
    if not shared:
        return pd.DataFrame(columns=columns)
    records: list[dict] = []
    for name in shared:
        kind = str(train.loc[name, "target_kind"])
        neutral = neutral_effect(kind)
        train_effect = float(train.loc[name, "observed_effect"])
        held_effect = float(held.loc[name, "observed_effect"])
        train_delta = abs(train_effect - neutral)
        held_delta = abs(held_effect - neutral)
        records.append(
            {
                "control_id": str(name),
                "kind": str(train.loc[name, "kind"]),
                "train_effect": train_effect,
                "validation_effect": held_effect,
                # Positive means the effect shrank on held-out data, which is
                # the expected direction; negative means it grew.
                # Only defined when there was an effect to lose. Guarding on
                # a bare epsilon is not enough: ``qt_drug_p_width`` scored
                # 0,5002 on training, a margin of 0,0002, and the ratio came
                # out at -32,02 -- a number that says nothing except that the
                # denominator was noise. The floor is the same one the panel
                # already uses to decide whether an effect exists at all.
                "shrinkage": (
                    float(1.0 - held_delta / train_delta)
                    if np.isfinite(train_delta)
                    and train_delta >= minimum_effect(kind)
                    else float("nan")
                ),
                "train_margin": train_delta,
                "shrinkage_defined": bool(
                    np.isfinite(train_delta)
                    and train_delta >= minimum_effect(kind)
                ),
            }
        )
    return pd.DataFrame(records)


def control_panel_verdict(table: pd.DataFrame) -> dict:
    """What the panel says about the run that produced it.

    Taken on the validation rows. The first version took whichever row came
    first, which was training -- see :func:`evaluate_controls`.
    """

    if table is None or table.empty:
        return {
            "version": CONTROL_PANEL_VERSION,
            "n_controls": 0,
            "trustworthy": False,
            "reason": "keine Kontrollen ausgewertet",
        }
    full = table
    shrinkage = effect_shrinkage(full)
    table = _verdict_rows(full)
    if table.empty:
        return {
            "version": CONTROL_PANEL_VERSION,
            "n_controls": int(len(full)),
            "verdict_split": VERDICT_SPLIT,
            "trustworthy": False,
            "reason": (
                f"keine Kontrollzeilen auf dem Split {VERDICT_SPLIT!r}; ein "
                "Urteil auf den Trainingszeilen waere kein Urteil"
            ),
        }
    positive = table.loc[table["kind"].eq("positive")]
    negative = table.loc[table["kind"].eq("negative")]
    tested_positive = positive.loc[positive["found"].astype(bool)]
    tested_negative = negative.loc[negative["found"].astype(bool)]
    recovered = int((tested_positive["status"] == "recovered").sum())
    false_positive = int((tested_negative["status"] == "false_positive").sum())

    # The most useful number the panel produces. A pair with no mechanism
    # still shows an effect, because in an intensive-care population almost
    # everything tracks being ill. Whatever the largest of those is, a finding
    # below it cannot be told apart from confounding -- measured 2026-08-30,
    # digoxin exposure against P-wave duration reached AUC 0.616, larger than
    # any mechanism-backed control in the same run.
    noise_floor = float("nan")
    if not tested_negative.empty:
        deltas = [
            abs(float(row["observed_effect"]) - neutral_effect(row["target_kind"]))
            for _, row in tested_negative.iterrows()
            if np.isfinite(row["observed_effect"])
        ]
        if deltas:
            noise_floor = float(max(deltas))

    # A screen that recovers none of the physiology it was pointed at cannot
    # be read either way -- its null results are as uninformative as its hits.
    trustworthy = bool(
        len(tested_positive) > 0
        and recovered >= max(1, (len(tested_positive) + 1) // 2)
    )
    result = {
        "version": CONTROL_PANEL_VERSION,
        "n_controls": int(len(full)),
        "verdict_split": VERDICT_SPLIT,
        "n_controls_on_verdict_split": int(len(table)),
        "n_positive_declared": int(len(positive)),
        "n_positive_tested": int(len(tested_positive)),
        "n_positive_recovered": recovered,
        "n_positive_wrong_direction": int(
            (tested_positive["status"] == "wrong_direction").sum()
        ),
        "n_negative_declared": int(len(negative)),
        "n_negative_tested": int(len(tested_negative)),
        "n_negative_false_positive": false_positive,
        "empirical_false_positive_rate": (
            float(false_positive / len(tested_negative))
            if len(tested_negative)
            else float("nan")
        ),
        "confounding_noise_floor": noise_floor,
        # How many controls produce an effect large enough to be told from
        # the floor at all. The verdict deliberately judges direction and
        # significance rather than magnitude -- at these sample sizes a real
        # mechanism can be highly significant and tiny -- but a reader has to
        # see how small. Measured on the development run: 2 of 10.
        "n_above_effect_floor": int(
            sum(
                1
                for _, row in table.iterrows()
                if bool(row.get("effect_above_floor", False))
            )
        ),
        "trustworthy": trustworthy,
        "reason": (
            "Mehrheit der Positivkontrollen wiedergefunden"
            if trustworthy
            else "zu wenige Positivkontrollen wiedergefunden -- Negativbefunde "
            "dieses Laufs sind nicht als Abwesenheit lesbar"
        ),
    }
    if not shrinkage.empty:
        usable = shrinkage["shrinkage"].dropna()
        if not usable.empty:
            result["median_effect_shrinkage"] = float(usable.median())
            result["max_effect_shrinkage"] = float(usable.max())
    return result


def render_control_panel(table: pd.DataFrame, verdict: Mapping) -> str:
    lines = [
        f"{'Kontrolle':28s} {'Art':9s} {'erwartet':9s} {'gemessen':>10s}  Befund",
    ]
    for _, row in table.iterrows():
        expected = {1: "hoeher", -1: "niedriger", 0: "kein Effekt"}[
            int(row["expected_direction"])
        ]
        observed = (
            f"{float(row['observed_effect']):.3f}"
            if np.isfinite(row["observed_effect"])
            else "-"
        )
        lines.append(
            f"{str(row['control_id']):28s} {str(row['kind']):9s} "
            f"{expected:9s} {observed:>10s}  {row['status']}"
        )
    lines.append("")
    lines.append(
        f"Positivkontrollen wiedergefunden: "
        f"{verdict.get('n_positive_recovered', 0)}"
        f"/{verdict.get('n_positive_tested', 0)}"
    )
    rate = verdict.get("empirical_false_positive_rate")
    if rate is not None and np.isfinite(rate):
        lines.append(
            f"Empirische Falsch-Positiv-Rate der Negativkontrollen: {rate:.0%}"
        )
    floor = verdict.get("confounding_noise_floor")
    if floor is not None and np.isfinite(floor):
        lines.append(
            f"Konfundierungs-Rauschgrenze: {floor:.3f} -- ein Befund unterhalb "
            "davon ist von Konfundierung nicht zu unterscheiden"
        )
    lines.append(f"Belastbar: {'ja' if verdict.get('trustworthy') else 'nein'}"
                 f" -- {verdict.get('reason', '')}")
    return "\n".join(lines)


def run_control_panel(
    config,
    *,
    run_id: str,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
):
    """Check the declared controls against what the screen actually produced.

    Reads ``screening_results.parquet`` -- the same table every other reader
    of this run sees -- so the panel can never be checking a private copy of
    the numbers. It computes nothing new: its whole job is to ask whether the
    findings a run already made point where physiology says they must.

    A failing positive control does not stop the run. It is written down, and
    the verdict travels with the results, because the appropriate response to
    "potassium does not move the T wave here" is to look at the pairing, not
    to discard the run.
    """

    import pandas as pd

    from .storage import (
        StageManifest,
        atomic_write_json,
        atomic_write_parquet,
        content_hash,
        scientific_config_hash,
    )

    run_root = config.runs_root / run_id
    output_root = run_root / "controls"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "control_panel.manifest.json"

    source = run_root / "screening_results.parquet"
    if not source.exists():
        raise FileNotFoundError(
            f"Die Kontrolltafel braucht die Screening-Ergebnisse: {source}"
        )

    input_hash = content_hash(
        {
            "run_id": run_id,
            "source": str(source.resolve()),
            "source_mtime": source.stat().st_mtime_ns,
            "panel": [control.control_id for control in CONTROL_PANEL],
            "alpha": float(CONTROL_ALPHA),
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, CONTROL_PANEL_VERSION):
            return previous

    if progress:
        progress("Kontrolltafel: Screening-Ergebnisse lesen")
    results = pd.read_parquet(source)

    table = evaluate_controls(results, effect_column="roc_auc")
    verdict = control_panel_verdict(table)
    verdict["run_id"] = str(run_id)
    shrinkage = effect_shrinkage(table)
    report = render_control_panel(table, verdict)

    if progress:
        floor = verdict.get("confounding_noise_floor")
        progress(
            "Kontrolltafel: "
            f"{verdict.get('n_positive_recovered', 0)}/"
            f"{verdict.get('n_positive_tested', 0)} Positivkontrollen "
            "wiedergefunden, "
            f"{verdict.get('n_positive_wrong_direction', 0)} mit falscher "
            "Richtung, "
            f"{verdict.get('n_negative_false_positive', 0)}/"
            f"{verdict.get('n_negative_tested', 0)} Negativkontrollen "
            "auffaellig"
            + f" (Urteil auf {verdict.get('verdict_split', '?')})"
            + (f", Rauschboden {float(floor):.3f}" if floor is not None else "")
        )

    table_path = output_root / "control_panel.parquet"
    shrinkage_path = output_root / "control_panel_shrinkage.parquet"
    verdict_path = output_root / "control_panel_verdict.json"
    report_path = output_root / "control_panel.md"
    atomic_write_parquet(table, table_path)
    atomic_write_parquet(shrinkage, shrinkage_path)
    atomic_write_json(verdict_path, verdict)
    report_path.write_text(report, encoding="utf-8")

    manifest = StageManifest.complete(
        stage="control-panel",
        stage_version=CONTROL_PANEL_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(table_path.resolve()),
            str(shrinkage_path.resolve()),
            str(verdict_path.resolve()),
            str(report_path.resolve()),
        ),
        row_counts={
            "controls": int(len(table)),
            "shrinkage_pairs": int(len(shrinkage)),
        },
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "run_control_panel",
    "CONTROL_PANEL",
    "CONTROL_PANEL_VERSION",
    "Control",
    "NEGATIVE_CONTROLS",
    "POSITIVE_CONTROLS",
    "VERDICT_SPLIT",
    "control_panel_verdict",
    "effect_shrinkage",
    "evaluate_controls",
    "minimum_effect",
    "neutral_effect",
    "render_control_panel",
]
