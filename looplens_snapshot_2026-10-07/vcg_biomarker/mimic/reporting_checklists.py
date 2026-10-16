"""Reporting checklists carried with every run (audit 5.3.8).

STARD 2015 (Bossuyt et al., BMJ 2015, PMID 26511519) for diagnostic
accuracy, TRIPOD+AI (Collins et al., BMJ 2024, PMID 38626948) for prediction
models and RECORD (Benchimol et al., PLoS Med 2015, PMID 26440803) for
routinely collected data. Item numbers follow the published checklists; the
topics are paraphrased. Each item names the run artifact that answers it or
says that it is the author's to write (also where the pipeline produces no
dedicated table, e.g. sample-size justification or baseline
characteristics). ``present`` checks only that the artifact exists --
whether it answers the item well is the reviewer's call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

REPORTING_CHECKLIST_VERSION = "1.0.0"
AUTHOR = "author_responsibility"

# (item, topic, artifact relative to the run root or AUTHOR)
STARD_2015: tuple[tuple[str, str, str], ...] = (
    ("1", "identification as a diagnostic accuracy study", AUTHOR),
    ("2", "structured abstract", AUTHOR),
    ("3", "scientific and clinical background, intended use", AUTHOR),
    ("4", "objectives and hypotheses", "validation_spec.json"),
    ("5", "prospective or retrospective data collection", "validation_spec.json"),
    ("6", "eligibility criteria", "endpoint_index_cohorts.parquet"),
    ("7", "basis for identifying eligible participants", "endpoint_index_cohorts.parquet"),
    ("8", "where and when participants were identified", "endpoint_index_cohorts.parquet"),
    ("9", "consecutive, random or convenience series", "endpoint_index_cohorts.parquet"),
    ("10a", "index test in detail", "feature_panels.json"),
    ("10b", "reference standard in detail", "label_definitions.json"),
    ("11", "rationale for the reference standard", "label_definitions.json"),
    ("12a", "index test positivity cut-offs, prespecified or not", "operating_point_results.parquet"),
    ("12b", "reference standard positivity cut-offs", "label_definitions.json"),
    ("13a", "blinding of index test to reference and clinical information", "validation_spec.json"),
    ("13b", "blinding of reference standard to index test", "validation_spec.json"),
    ("14", "methods for estimating or comparing accuracy", "screening_results.parquet"),
    ("15", "handling of indeterminate results", "label_summary.json"),
    ("16", "handling of missing data", "screening_results.parquet"),
    ("17", "analyses of variability, prespecified or exploratory", "overfitting_stability_report.parquet"),
    ("18", "intended sample size and how it was determined", AUTHOR),
    ("19", "flow of participants", "endpoint_index_cohorts.parquet"),
    ("20", "baseline characteristics", AUTHOR),
    ("21a", "severity distribution in those with the target condition", AUTHOR),
    ("21b", "alternative diagnoses in those without", AUTHOR),
    ("22", "interval and interventions between index test and reference", "label_definitions.json"),
    ("23", "cross tabulation of index test by reference standard", "operating_point_results.parquet"),
    ("24", "accuracy estimates with precision", "screening_results.parquet"),
    ("25", "adverse events from the tests", AUTHOR),
    ("26", "limitations", AUTHOR),
    ("27", "implications for practice", AUTHOR),
    ("28", "registration", AUTHOR),
    ("29", "access to the protocol", AUTHOR),
    ("30", "funding", AUTHOR),
)

TRIPOD_AI: tuple[tuple[str, str, str], ...] = (
    ("1", "title identifies development or evaluation of a prediction model", AUTHOR),
    ("2", "abstract per TRIPOD+AI for Abstracts", AUTHOR),
    ("3a", "healthcare context and rationale", AUTHOR),
    ("3b", "target population and intended purpose", AUTHOR),
    ("3c", "known health inequalities between groups", AUTHOR),
    ("4", "objectives: development, evaluation or both", "validation_spec.json"),
    ("5a", "data sources for development and evaluation", "validation_spec.json"),
    ("5b", "dates of participant data", "validation_spec.json"),
    ("6a", "setting and centres", "validation_spec.json"),
    ("6b", "eligibility criteria", "endpoint_index_cohorts.parquet"),
    ("6c", "treatments received and their handling", AUTHOR),
    ("7", "data pre-processing and quality checks", "feature_panels.json"),
    ("8a", "outcome definition and time horizon", "label_definitions.json"),
    ("8b", "subjective outcome assessment and assessors", "label_definitions.json"),
    ("8c", "blinding of outcome assessment", "validation_spec.json"),
    ("9a", "choice of initial predictors", "feature_panels.json"),
    ("9b", "predictor definitions, timing and measurement", "feature_panels.json"),
    ("9c", "subjective predictor measurement", "feature_panels.json"),
    ("10", "study size", AUTHOR),
    ("11", "missing data handling", "screening_results.parquet"),
    ("12a", "use of the data for development and evaluation", "validation_spec.json"),
    ("12b", "handling of predictors in the analysis", "adjusted_model_results.parquet"),
    ("12c", "model type and building steps", "adjusted_model_results.parquet"),
    ("12d", "heterogeneity across clusters", "overfitting_stability_report.parquet"),
    ("12e", "performance measures and plots", "screening_results.parquet"),
    ("12f", "model updating", AUTHOR),
    ("12g", "calculation of predictions in evaluation", "adjusted_model_results.parquet"),
    ("13", "class imbalance methods", AUTHOR),
    ("14", "fairness approaches", "overfitting_stability_report.parquet"),
    ("15", "model output and decision thresholds", "operating_point_results.parquet"),
    ("16", "differences between development and evaluation data", "overfitting_stability_report.parquet"),
    ("17", "ethics approval", AUTHOR),
    ("18a", "funding", AUTHOR),
    ("18b", "conflicts of interest", AUTHOR),
    ("18c", "protocol access", AUTHOR),
    ("18d", "registration", AUTHOR),
    ("18e", "data availability", AUTHOR),
    ("18f", "code availability", AUTHOR),
    ("19", "patient and public involvement", AUTHOR),
    ("20a", "participant flow and outcome counts", "endpoint_index_cohorts.parquet"),
    ("20b", "characteristics overall and per data source", AUTHOR),
    ("20c", "evaluation data compared with development data", "overfitting_stability_report.parquet"),
    ("21", "participants and events per analysis", "screening_results.parquet"),
    ("22", "full model specification", "adjusted_model_results.parquet"),
    ("23a", "performance estimates with intervals", "screening_results.parquet"),
    ("23b", "heterogeneity in performance across groups", "overfitting_stability_report.parquet"),
    ("24", "results of model updating", AUTHOR),
    ("25", "overall interpretation including fairness", AUTHOR),
    ("26", "limitations", AUTHOR),
    ("27a", "handling of poor-quality or unavailable inputs in use", AUTHOR),
    ("27b", "user interaction with input data", AUTHOR),
    ("27c", "next steps and generalisability", AUTHOR),
)

RECORD: tuple[tuple[str, str, str], ...] = (
    ("1.1", "type of routinely collected data in title or abstract", AUTHOR),
    ("1.2", "geographic region and time frame", "validation_spec.json"),
    ("1.3", "statement of database linkage", "validation_spec.json"),
    ("6.1", "population selection, codes and algorithms", "label_definitions.json"),
    ("6.2", "validation studies of the codes", "label_definitions.json"),
    ("6.3", "linkage flow diagram", AUTHOR),
    ("7.1", "complete list of codes and algorithms", "label_definitions.json"),
    ("12.1", "extent of database access", AUTHOR),
    ("12.2", "data cleaning methods", "feature_panels.json"),
    ("12.3", "linkage methods and quality", "validation_spec.json"),
    ("13.1", "selection of persons included", "endpoint_index_cohorts.parquet"),
    ("19.1", "implications of data not collected for research", AUTHOR),
    ("22.1", "access to protocol, raw data and code", AUTHOR),
)

CHECKLISTS = {
    "STARD_2015": ("PMID 26511519", STARD_2015),
    "TRIPOD_AI": ("PMID 38626948", TRIPOD_AI),
    "RECORD": ("PMID 26440803", RECORD),
}


def build_reporting_checklists(roots: Sequence[Path]) -> dict:
    """Checklist status for one run: which artifact answers each item.

    ``roots`` are the run directory and the shared label, evidence and
    feature roots, where the cohort, label and panel artifacts live.
    """

    available: set[str] = set()
    for root in roots:
        root = Path(root)
        if root.is_dir():
            available.update(path.name for path in root.rglob("*") if path.is_file())
    output: dict = {"version": REPORTING_CHECKLIST_VERSION, "checklists": {}}
    for name, (source, items) in CHECKLISTS.items():
        rows = []
        for item, topic, artifact in items:
            if artifact == AUTHOR:
                status = AUTHOR
            else:
                status = "artifact_present" if artifact in available else "artifact_missing"
            rows.append({"item": item, "topic": topic, "answered_by": artifact, "status": status})
        output["checklists"][name] = {
            "source": source,
            "items": rows,
            "n_items": len(rows),
            "n_artifact_present": sum(row["status"] == "artifact_present" for row in rows),
            "n_artifact_missing": sum(row["status"] == "artifact_missing" for row in rows),
            "n_author_responsibility": sum(row["status"] == AUTHOR for row in rows),
        }
    return output
