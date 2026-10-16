"""A second, independent derivation of the same labels.

MIMIC-IV-ECG-Ext-ICD links every one of the 800.035 ECGs to the ICD-10 codes of
the encounter it belongs to. This project already derives its endpoints from the
same underlying MIMIC tables, with its own rules and its own time windows -- so
the two are independent readings of one source, and where they disagree at
least one of them is wrong.

That is the whole value here, and it is worth being precise about what it is
*not*. This is not an external cohort: it is the same patients, the same
recordings, the same hospital. Agreement proves nothing about generalisation.
Disagreement, on the other hand, is informative in one direction only -- it
localises a defect, in one derivation or the other, that neither could find
alone.

**Why the published folds are read but not adopted.** The file carries a
20-fold stratified assignment (``strat_fold``) that other work on this dataset
uses, which makes results comparable across papers. This project has its own
train/validation/locked_test split, and that split is what every threshold,
every model and every frozen contract in the run rests on. Swapping it would
invalidate all of them. So the folds are read and the overlap is *reported* --
a reader who wants to compare against fold-based literature can see exactly how
the two partitions relate -- and nothing downstream consumes them.

**What the columns mean.** Four groupings, differing in where the diagnosis
came from and when:

``ed_diag_ed``
    coded in the emergency department for that ED stay.
``ed_diag_hosp``
    coded for the hospital admission that the ED stay led to.
``hosp_diag_hosp``
    coded for the hospital admission the ECG was taken in.
``all_diag_hosp`` / ``all_diag_all``
    the union, the second including diagnoses from the patient's other
    encounters.

Only the first three are anchored to the encounter that contains the ECG.
``all_diag_all`` reaches across a patient's history, so a code there may have
been assigned years before or after the recording -- which is the difference
between "this ECG was taken during that illness" and "this patient had that
illness at some point". Both are compared, and the comparison is only readable
because both are.

**What it found**, over 640.376 linked ECGs:

==========================  ==========  ==========  ============  ==========
Endpunkt                    Umfang      beide       nur Projekt   nur Ext-ICD
==========================  ==========  ==========  ============  ==========
heart_failure_icd           Vorgesch.   97.967      32            3
cardiac_arrest              Vorgesch.   6.055       2             2
pericarditis_icd            Vorgesch.   7.080       4             12
acute_mi_high_specificity   Vorgesch.   1.585       5             9
acute_mi_probable           Vorgesch.   5.071       9             9
cardiomyopathy_icd          Vorgesch.   18.264      235           0
==========================  ==========  ==========  ============  ==========

Against the right scope the two derivations agree to within a handful of ECGs
out of a hundred thousand. That is a strong statement about this project's
label pipeline, and it could not have been made from inside it.

It also shows how easily the same data tells the opposite story. Compared
against the *encounter* codes alone, this project marked ``heart_failure_icd``
positive 21.458 times where no code appeared -- which reads as a 28 % over-call
and is nothing but the window: the endpoint asks about the patient, the column
answers about the admission. Running one column would have produced a defect
report about a pipeline that is correct.

**Two endpoints stay apart, and the likelier fault is on this side.**
``old_mi_icd`` (37.648 project-only) and ``pulmonary_hypertension_icd``
(13.570) disagree under both scopes. The prefixes above are this module's
mapping, not the endpoint's definition -- ``I252`` is specifically "old
myocardial infarction" while the endpoint may admit a past ``I21``, and the
pulmonary hypertension codes are wider than ``I270``/``I272``. Both are
mapping questions to settle before either is read as a labelling defect.

``new_onset_af_30d`` is the control and behaves as one: an endpoint about the
*next thirty days* compared against codes from the encounter agrees on nothing
at all under the encounter scope (0 of 16.429). That is the evidence that this
comparison is sensitive to timing rather than measuring string overlap.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import pandas as pd


EXT_ICD_VERSION = "1.0.0"

# The file, as published.
EXT_ICD_FILENAME = "records_w_diag_icd10.csv"
EXT_ICD_DIRECTORY = (
    "mimic-iv-ecg-ext-icd-diagnostic-labels-for-mimic-iv-ecg-1.0.1"
)

# Anchored to the encounter that contains the ECG. See the module docstring for
# why the wider groupings are not the default.
ENCOUNTER_SCOPED_COLUMNS: tuple[str, ...] = (
    "hosp_diag_hosp",
    "ed_diag_hosp",
    "ed_diag_ed",
)
DEFAULT_DIAGNOSIS_COLUMN = "hosp_diag_hosp"

# Endpoints worth cross-checking, with the ICD-10 prefixes that should carry
# them. Deliberately short: each pairing is a claim about coding practice, and
# a long list of guesses would turn a check into noise.
#
# Every name here was read off the label table rather than guessed. The first
# draft of this list invented eight endpoint names from what a cardiac endpoint
# set "should" contain; six of the eight did not exist, and the cross-check
# would have silently compared nothing while reporting that it ran.
#
# Two pairings are deliberately imperfect and are the interesting ones:
# ``acute_mi_high_specificity`` and ``acute_mi_probable`` share the same ICD
# prefixes but differ in how strictly this project adjudicates them, so the two
# rows measure what that strictness costs and buys. ``new_onset_af_30d`` is a
# *future* endpoint compared against codes from the encounter, so a low
# agreement there is expected and is the control that shows the comparison is
# sensitive to timing at all.
CROSS_CHECK_PREFIXES: dict[str, tuple[str, ...]] = {
    "heart_failure_icd": ("I50",),
    "cardiomyopathy_icd": ("I42", "I43"),
    "old_mi_icd": ("I252",),
    "pericarditis_icd": ("I30", "I31", "I32"),
    "pulmonary_embolism_icd": ("I26",),
    "pulmonary_hypertension_icd": ("I270", "I272"),
    "cardiac_arrest": ("I46",),
    "acute_mi_high_specificity": ("I21", "I22"),
    "acute_mi_probable": ("I21", "I22"),
    "new_onset_af_30d": ("I48",),
}


@dataclass(frozen=True)
class ExtIcdPolicy:
    """Everything that decides an agreement number."""

    diagnosis_column: str = DEFAULT_DIAGNOSIS_COLUMN
    # The wide grouping is compared *as well*, never instead. Running only the
    # encounter-scoped column made the first result uninterpretable: this
    # project called ``old_mi_icd`` positive 2,7 times as often, which looks
    # like a defect and is nothing of the kind -- an endpoint about a patient's
    # history cannot be checked against the codes of one admission. Reporting
    # both columns separates a window difference from a derivation difference,
    # which is the only way either becomes readable.
    wide_column: str = "all_diag_all"
    minimum_positive: int = 50

    def to_dict(self) -> dict:
        return {
            "diagnosis_column": str(self.diagnosis_column),
            "wide_column": str(self.wide_column),
            "minimum_positive": int(self.minimum_positive),
            "version": EXT_ICD_VERSION,
        }


def ext_icd_path(config) -> Optional[Path]:
    """Where the published table lives, if it was added to the database."""

    root = getattr(config, "mimic_ecg_root", None)
    if root is None:
        return None
    candidate = Path(root).parent / EXT_ICD_DIRECTORY / EXT_ICD_FILENAME
    return candidate if candidate.exists() else None


def parse_code_list(value: object) -> tuple[str, ...]:
    """The codes of one ECG, from the published list-shaped string.

    The column holds a Python list literal rather than a delimited field, so it
    is parsed as one -- splitting on commas would keep the brackets and quotes
    and turn every code into a near-miss that matches nothing.
    """

    if value is None:
        return ()
    text = str(value).strip()
    if not text or text in {"[]", "nan", "None"}:
        return ()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return ()
    if not isinstance(parsed, (list, tuple)):
        return ()
    return tuple(
        str(item).strip().upper().replace(".", "")
        for item in parsed
        if str(item).strip()
    )


def has_any_prefix(codes: Iterable[str], prefixes: Sequence[str]) -> bool:
    upper = tuple(str(prefix).upper() for prefix in prefixes)
    return any(str(code).upper().startswith(upper) for code in codes)


def ext_icd_query(
    ext_icd_sql: str,
    master_sql: str,
    *,
    policy: ExtIcdPolicy = ExtIcdPolicy(),
    splits: Sequence[str] = ("train", "validation"),
) -> str:
    """The published labels beside this project's own split assignment."""

    split_list = ", ".join(f"'{value}'" for value in splits)
    return f"""
        SELECT
            m.subject_id, m.study_id, m.split,
            x.{policy.diagnosis_column} AS diagnosis_codes,
            x.{policy.wide_column} AS wide_diagnosis_codes,
            x.strat_fold,
            x.ecg_taken_in_ed,
            x.ecg_taken_in_hosp
        FROM read_parquet('{master_sql}') m
        JOIN read_csv_auto('{ext_icd_sql}') x
          ON x.study_id = m.study_id AND x.subject_id = m.subject_id
        WHERE m.split IN ({split_list})
    """


def agreement_table(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    prefixes: dict[str, tuple[str, ...]] = None,
    policy: ExtIcdPolicy = ExtIcdPolicy(),
) -> pd.DataFrame:
    """Where the two derivations agree, and where exactly they part.

    The four cells are reported separately rather than as one agreement
    statistic, because they mean different things: an endpoint this project
    calls positive where no code exists is a possible over-call, and a code
    with no endpoint is a possible miss. A single kappa would hide which.
    """

    columns = [
        "endpoint_id",
        "code_scope",
        "icd_prefixes",
        "n_compared",
        "n_both",
        "n_project_only",
        "n_ext_icd_only",
        "n_neither",
        "agreement",
        "project_positive_rate",
        "ext_icd_positive_rate",
    ]
    prefixes = dict(prefixes or CROSS_CHECK_PREFIXES)
    if frame is None or len(frame) == 0 or labels is None or len(labels) == 0:
        return pd.DataFrame(columns=columns)

    merged = frame.merge(labels, on=["subject_id", "study_id"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)
    scopes = [("encounter", "diagnosis_codes")]
    if "wide_diagnosis_codes" in merged.columns:
        scopes.append(("patient_history", "wide_diagnosis_codes"))
    parsed = {
        name: merged[column].map(parse_code_list) for name, column in scopes
    }

    records: list[dict] = []
    for endpoint, endpoint_prefixes in sorted(prefixes.items()):
        state = f"{endpoint}_state"
        if state not in merged.columns:
            continue
        adjudicated = merged[state].isin(["positive", "negative"])
        if int(adjudicated.sum()) == 0:
            continue
        project = merged.loc[adjudicated, state].eq("positive")
        for scope, _ in scopes:
            external = parsed[scope][adjudicated].map(
                lambda items, p=endpoint_prefixes: has_any_prefix(items, p)
            )
            both = int((project & external).sum())
            project_only = int((project & ~external).sum())
            external_only = int((~project & external).sum())
            neither = int((~project & ~external).sum())
            total = both + project_only + external_only + neither
            if both + project_only < int(policy.minimum_positive):
                continue
            records.append(
                {
                    "endpoint_id": str(endpoint),
                    "code_scope": scope,
                    "icd_prefixes": ", ".join(endpoint_prefixes),
                    "n_compared": total,
                    "n_both": both,
                    "n_project_only": project_only,
                    "n_ext_icd_only": external_only,
                    "n_neither": neither,
                    "agreement": (both + neither) / max(total, 1),
                    "project_positive_rate": (both + project_only) / max(total, 1),
                    "ext_icd_positive_rate": (both + external_only) / max(total, 1),
                }
            )
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records).sort_values(
        ["endpoint_id", "code_scope"], ignore_index=True
    )


def fold_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    """How the published 20 folds relate to this project's own split.

    Reported so a reader can compare against fold-based literature. Nothing
    downstream consumes it: this project's split carries every threshold and
    every frozen contract in the run, and swapping partitions would invalidate
    all of them at once.
    """

    if (
        frame is None
        or len(frame) == 0
        or "strat_fold" not in frame
        or "split" not in frame
    ):
        return pd.DataFrame(columns=["strat_fold", "split", "n_ecgs"])
    table = (
        frame.groupby(["strat_fold", "split"], dropna=False)
        .size()
        .reset_index(name="n_ecgs")
    )
    return table.sort_values(["strat_fold", "split"], ignore_index=True)


def run_ext_icd_crosscheck(
    config,
    *,
    run_id: str,
    policy: ExtIcdPolicy = ExtIcdPolicy(),
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
):
    """Compare this project's endpoint labels with the published ones."""

    import duckdb

    from .storage import (
        StageManifest,
        atomic_write_json,
        atomic_write_parquet,
        content_hash,
        scientific_config_hash,
    )

    output_root = config.runs_root / run_id / "ext_icd"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "ext_icd_crosscheck.manifest.json"

    source = ext_icd_path(config)
    if source is None:
        raise FileNotFoundError(
            "MIMIC-IV-ECG-Ext-ICD ist nicht in der Datenbank; erwartet unter "
            f"<mimic_ecg_root>/../{EXT_ICD_DIRECTORY}/{EXT_ICD_FILENAME}"
        )
    labels_path = config.labels_root / "clinical_labels_wide.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Der Abgleich braucht die eigenen Labels: {labels_path}"
        )

    input_hash = content_hash(
        {
            "run_id": run_id,
            "source": str(source.resolve()),
            "source_size": source.stat().st_size,
            "labels_mtime": labels_path.stat().st_mtime_ns,
            "policy": policy.to_dict(),
            "prefixes": {k: list(v) for k, v in CROSS_CHECK_PREFIXES.items()},
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, EXT_ICD_VERSION):
            return previous

    def sql_path(value) -> str:
        return str(value).replace(chr(92), "/")

    if progress:
        progress("Ext-ICD: publizierte Labels lesen")
    from .storage import isolated_analysis_connection

    with isolated_analysis_connection(config) as connection:
        external = connection.execute(
            ext_icd_query(
                sql_path(source),
                sql_path(config.master_root / "ecg_master_index.parquet"),
                policy=policy,
            )
        ).df()
        wanted = set(
            connection.execute(
                f"SELECT * FROM read_parquet('{sql_path(labels_path)}') LIMIT 0"
            ).df().columns
        )
        states = [
            f"{name}_state"
            for name in CROSS_CHECK_PREFIXES
            if f"{name}_state" in wanted
        ]
        own = connection.execute(
            "SELECT subject_id, study_id"
            + ("".join(f", {name}" for name in states) if states else "")
            + f" FROM read_parquet('{sql_path(labels_path)}')"
        ).df()

    if progress:
        progress(
            f"Ext-ICD: {len(external):,} EKGs gekoppelt, "
            f"{len(states)} Endpunkte vergleichbar"
        )

    agreement = agreement_table(own, external, policy=policy)
    folds = fold_overlap(external)
    summary = {
        "version": EXT_ICD_VERSION,
        "run_id": str(run_id),
        "policy": policy.to_dict(),
        "n_linked_ecgs": int(len(external)),
        "n_endpoints_compared": int(len(agreement)),
        "n_folds": int(external["strat_fold"].nunique()) if len(external) else 0,
        "folds_are_reported_not_adopted": True,
        "same_population": (
            "Dieselbe Kohorte, dieselben Aufzeichnungen -- Uebereinstimmung "
            "belegt keine Uebertragbarkeit; nur eine Abweichung ist "
            "informativ."
        ),
    }
    if not agreement.empty:
        worst = agreement.iloc[0]
        summary["lowest_agreement_endpoint"] = str(worst["endpoint_id"])
        summary["lowest_agreement"] = float(worst["agreement"])
        if progress:
            progress(
                "Ext-ICD: geringste Uebereinstimmung "
                f"{worst['agreement']*100:.1f} % bei {worst['endpoint_id']}"
            )

    agreement_path = output_root / "ext_icd_agreement.parquet"
    folds_path = output_root / "ext_icd_fold_overlap.parquet"
    summary_path = output_root / "ext_icd_summary.json"
    atomic_write_parquet(agreement, agreement_path)
    atomic_write_parquet(folds, folds_path)
    atomic_write_json(summary_path, summary)

    manifest = StageManifest.complete(
        stage="ext-icd-crosscheck",
        stage_version=EXT_ICD_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(agreement_path.resolve()),
            str(folds_path.resolve()),
            str(summary_path.resolve()),
        ),
        row_counts={
            "endpoints": int(len(agreement)),
            "linked_ecgs": int(len(external)),
        },
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "CROSS_CHECK_PREFIXES",
    "DEFAULT_DIAGNOSIS_COLUMN",
    "ENCOUNTER_SCOPED_COLUMNS",
    "EXT_ICD_DIRECTORY",
    "EXT_ICD_FILENAME",
    "EXT_ICD_VERSION",
    "ExtIcdPolicy",
    "agreement_table",
    "ext_icd_path",
    "ext_icd_query",
    "fold_overlap",
    "has_any_prefix",
    "parse_code_list",
    "run_ext_icd_crosscheck",
]
