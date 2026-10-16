"""A small, fail-closed entrance to the result atlas.

The existing research views stay available in detailed mode.  This module
deliberately does not compute clinical evidence from a displayed AUC.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import duckdb
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QTabWidget,
    QTableWidget, QTableWidgetItem,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ...mimic.models import ResultQuery
from ...mimic.result_query import ResultRepository, ResultPage
from .map_view import AtlasMapWidget, MapLoad
from .export import choose_plot_export, choose_table_export
from .discovery_view import AtlasDiscoveryWidget, DiscoverySnapshot
from .marker_view import AtlasMarkerWidget
from .model_quality import AtlasModelQualityWidget, ModelQualitySnapshot, load_model_quality
from .availability_view import AtlasAvailabilityWidget, AvailabilitySnapshot, load_availability
from .row_scope import compatible_validation_clauses
from .transport_view import AtlasTransportWidget, TransportSnapshot
from .paired_view import (
    AtlasPairedWidget, PairedSnapshot, load_paired_snapshot, paired_summary,
    paired_outcome_summary,
)
from .rule_audit import AtlasRuleAuditWidget, RuleAuditSnapshot
from .broad_axis_view import AtlasBroadAxisWidget, BroadAxisLoad, BroadAxisSnapshot
from .prognosis_view import AtlasPrognosisWidget, PrognosisSnapshot, load_prognosis
from .response_view import AtlasResponseWidget, ResponseSnapshot, load_response
from .curve_view import AtlasCurveWidget, CurveSnapshot, load_marker_curves
from .score_view import AtlasScoreWidget, ScoreSnapshot, load_score_snapshot


def _string(value: object, fallback: str = "nicht angegeben") -> str:
    if value is None or pd.isna(value):
        return fallback
    result = str(value).strip()
    return result if result and result.lower() not in {"nan", "none"} else fallback


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _eligible_research_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """No technical self-comparisons or unsupported clinical promotion."""
    if rows.empty:
        return rows
    frame = rows.copy()
    if "reference_independence" in frame:
        frame = frame.loc[~frame["reference_independence"].astype(str).eq(
            "same_ecg_technical_comparison"
        )]
    if "claim_ceiling" in frame:
        frame = frame.loc[~frame["claim_ceiling"].astype(str).eq("agreement_only")]
    if "candidate_type" in frame:
        frame = frame.loc[~frame["candidate_type"].astype(str).eq("classical_rule")]
    if "selection_eligible" in frame:
        frame = frame.loc[frame["selection_eligible"].fillna(False).eq(True)]
    if "historical_pre_audit_result" in frame:
        frame = frame.loc[~frame["historical_pre_audit_result"].astype(str).str.casefold().eq("true")]
    elif "historical_implementation" in frame:
        frame = frame.loc[~frame["historical_implementation"].astype(str).str.casefold().eq("true")]
    if "roc_auc" in frame:
        frame = frame.loc[pd.to_numeric(frame["roc_auc"], errors="coerce").notna()]
    return frame


def _stage(row: pd.Series) -> str:
    return _string(row.get("evidence_stage"), "exploratory")


def _stage_label(stage: str) -> str:
    return {
        "clinically_validated": "Klinisch validiert",
        "guideline_candidate": "Kriterienkandidat",
        "replicated_association": "Replizierte Assoziation",
        "exploratory": "Forschungshypothese",
    }.get(stage, "Forschungshypothese")


def _reference_label(value: object) -> str:
    code = _string(value, "unverified_or_unknown")
    return {
        "timed_quantitative": "Echo/Labor im definierten Zuordnungsfenster",
        "blinded_adjudication": "verblindete unabhängige Beurteilung",
        "independent_outcome": "unabhängig erfasstes Ereignis",
        "administrative_or_text": "administrativer Code oder Text",
        "same_ecg_device": "Gerätebefund desselben ECGs",
        "echo_documentation_proxy_not_disease_exclusion": "Echo-Dokumentation, kein Ausschlussbefund",
        "mixed_or_unknown": "gemischte oder unklare Referenz",
        "unverified_or_unknown": "unbekannte Referenzqualität",
    }.get(code, code.replace("_", " "))


def _ceiling_label(value: object) -> str:
    code = _string(value, "unknown")
    return {
        "agreement_only": "nur Übereinstimmung mit demselben ECG",
        "association_only": "nur Assoziation",
        "independent_reference_comparison": "Vergleich mit unabhängiger Referenz",
        "documentation_association_only": "nur Dokumentations-Assoziation",
    }.get(code, "Aussagegrenze nicht belegt")


def _timing_label(row: pd.Series) -> str:
    if _string(row.get("reference_kind"), "") != "timed_quantitative":
        return ""
    note = _string(row.get("timing_evidence_note"), "not_reported")
    count = _number(row.get("clock_alignment_confirmed_n"))
    if note == "no_clock_alignment_confirmed":
        return "Geräteuhr nicht unabhängig abgeglichen; zeitliche Zuordnung bleibt unsicher."
    if note == "some_clock_alignments_confirmed_not_necessarily_entire_cohort":
        return (
            f"Uhrabgleich bei {int(count)} Referenzen belegt, nicht zwingend für alle Fälle."
            if count is not None else
            "Uhrabgleich nur für einen Teil der Referenzen belegt."
        )
    return "Unabhängiger Uhrabgleich für diese Zuordnung nicht berichtet."


def _auc_label(row: pd.Series) -> str:
    auc = _number(row.get("roc_auc"))
    if auc is None:
        return "AUC nicht berechnet"
    low = _number(row.get("roc_auc_ci_low"))
    high = _number(row.get("roc_auc_ci_high"))
    if low is not None and high is not None and low <= auc <= high:
        return f"AUC {auc:.2f} (95-%-CI {low:.2f}–{high:.2f})"
    return f"AUC {auc:.2f} (CI nicht berechnet)"


def _case_label(row: pd.Series) -> str:
    positive = _number(row.get("n_positive"))
    negative = _number(row.get("n_negative"))
    if positive is None or negative is None:
        return "Fallzahl nicht angegeben"
    return f"{int(positive)} Fälle · {int(negative)} Kontrollen"


def _availability_label(row: pd.Series) -> str:
    eligible = _number(row.get("n_reference_eligible"))
    observed = _number(row.get("n_marker_observed"))
    if eligible is not None and observed is not None and eligible > 0 and 0 <= observed <= eligible:
        return f"messbar bei {int(observed)} von {int(eligible)} Referenzdatensätzen"
    missing = _number(row.get("missingness"))
    if missing is not None and 0 <= missing <= 1:
        return f"{missing:.0%} Marker-Missingness"
    return "Messverfügbarkeit nicht angegeben"


def _paired_comparison_text(rows: pd.DataFrame) -> str:
    """Explorer rows alone lack the paired comparison design and inference status."""
    if rows.empty:
        return "Klassischer ECG-Vergleich: noch keine auswertbare neue Markerzeile."
    return (
        "Klassischer ECG-Vergleich: aus der Markerübersicht allein nicht belegt. "
        "Gepaarte Vergleiche samt Messdefinition, Fallzahl und Inferenzstatus "
        "stehen im eigenen Detailreiter; getrennte AUCs auf unterschiedlichen "
        "Patienten sind nicht fair vergleichbar."
    )


def _limits_text(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "Kalibration, Entscheidungsnutzen und externe Bestätigung: nicht beurteilbar."
    gates = set()
    if "failed_gates" in rows:
        for value in rows["failed_gates"].dropna():
            gates.update(part for part in str(value).split("|") if part)
    labels = [
        ("calibration", "Kalibration"),
        ("net_benefit", "Entscheidungsnutzen"),
        ("external_validation", "externe Bestätigung"),
    ]
    if not gates:
        return (
            "Kalibration, Entscheidungsnutzen und externe Bestätigung werden hier "
            "nicht aus einer AUC abgeleitet; vollständige Gate-Belege im Detail prüfen."
        )
    missing = [label for gate, label in labels if gate in gates]
    if missing:
        return "Noch nicht belegt laut Ergebnisvertrag: " + ", ".join(missing) + "."
    return "Kalibration, Entscheidungsnutzen und externe Bestätigung: Gate-Belege im Detail prüfen."


@dataclass(frozen=True)
class AtlasCard:
    title: str
    body: str
    detail: str
    comparison: str
    open_questions: str
    endpoint_id: str
    takeaway: str = ""
    evidence_label: str = "Forschungshypothese"


@dataclass(frozen=True)
class AtlasPayload:
    page: ResultPage
    stage_counts: dict[str, int]
    gate_counts: tuple[tuple[str, int], ...]
    gate_patterns: pd.DataFrame
    scope_counts: dict[str, int]
    model_quality: ModelQualitySnapshot
    availability: AvailabilitySnapshot
    paired: PairedSnapshot
    prognosis: PrognosisSnapshot
    response: ResponseSnapshot
    scores: ScoreSnapshot


def cards_from_page(
    page: ResultPage, paired: PairedSnapshot | None = None,
) -> tuple[str, list[AtlasCard]]:
    if page.status not in {"ok", "empty"}:
        return (page.message_de or "Für diesen Lauf liegen noch keine kompatiblen Ergebnisse vor.", [])
    if page.rows.empty:
        return ("Für die aktuelle Auswahl liegen noch keine auswertbaren Ergebnisse vor.", [])
    eligible = _eligible_research_rows(page.rows)
    if eligible.empty:
        return (
            "In den geladenen Ergebnissen liegt kein auswahlfähiger neuer Markerbefund vor. "
            "Technische Selbstvergleiche werden nicht als Entdeckung gewertet.",
            [],
        )
    cards: list[AtlasCard] = []
    seen: set[str] = set()
    for _, row in eligible.iterrows():
        endpoint = _string(row.get("endpoint_id"), "")
        if endpoint in seen:
            continue
        seen.add(endpoint)
        stage = _stage(row)
        marker = _string(row.get("feature_name_de"), _string(row.get("feature_id")))
        target = _string(row.get("endpoint_name_de"), endpoint)
        reference = _reference_label(row.get("reference_kind"))
        ceiling = _ceiling_label(row.get("claim_ceiling"))
        post_outcome = str(row.get("post_outcome_hypothesis", "")).lower() == "true"
        cards.append(AtlasCard(
            title=target,
            body=(f"Messung im Validation-Anteil: {_auc_label(row)} · "
                  f"{_case_label(row)} · {_availability_label(row)}."),
            detail=(
                f"{_stage_label(stage)} · Referenz: {reference} · Aussagegrenze: {ceiling}."
                + (" Nach Pilotansicht formulierte Hypothese." if post_outcome else "")
                + (" " + _timing_label(row) if _timing_label(row) else "")
            ),
            comparison=(paired_summary(paired, endpoint, _string(row.get("feature_id"), ""))
                        if paired is not None else _paired_comparison_text(pd.DataFrame([row]))),
            open_questions=_limits_text(pd.DataFrame([row])),
            endpoint_id=endpoint,
            takeaway=(
                f"{marker} ist für diese klinische Frage extern validiert. "
                "Gültigkeitsbereich und Einzelnachweise vor Anwendung prüfen."
                if stage == "clinically_validated" else
                f"{marker} ist ein Kriterienkandidat, noch keine freigegebene Befundungsregel."
                if stage == "guideline_candidate" else
                f"{marker} zeigte eine replizierte Assoziation; ein klinischer Zusatznutzen ist separat zu prüfen."
                if stage == "replicated_association" else
                f"{marker} ist ein Forschungshinweis zu dieser Frage; ein besserer klinischer Befund ist nicht belegt."
            ),
            evidence_label=_stage_label(stage),
        ))
        if len(cards) == 3:
            break
    if not cards:
        return ("Noch kein auswertbarer neuer Markerbefund in dieser Auswahl.", [])
    if any(_stage(row) == "clinically_validated" for _, row in eligible.iterrows()):
        headline = "Die geladenen Ergebnisse enthalten klinisch validierte Marker; Details und Gates prüfen."
    elif any(_stage(row) in {"guideline_candidate", "replicated_association"} for _, row in eligible.iterrows()):
        headline = "Die geladenen Ergebnisse enthalten bestätigte Forschungsassoziationen, aber keine freigegebene Befundungsregel."
    else:
        headline = "Noch kein belastbar bestätigter neuer Befund. Die folgenden Ergebnisse sind Forschungshinweise."
    return headline, cards


def _atlas_state_text(page: ResultPage, cards: list[AtlasCard]) -> str:
    """Distinguish unavailable analyses from completed negative-looking views."""
    if page.status in {"no_compatible_run", "artifact_not_available"}:
        return "Noch nicht berechnet: Es liegt kein abgeschlossener, kompatibler Validation-Ergebnisstand vor."
    if page.status not in {"ok", "empty"}:
        return "Nicht verwendbar: Dieser Ergebnisstand ist unvollständig oder nicht kompatibel."
    if page.status == "empty":
        return "Abgeschlossen, aber für die aktuelle Auswahl sind keine Ergebniszeilen vorhanden."
    if not cards:
        return (
            "Ausgewertet, aber kein auswahlfähiger neuer Markerbefund in dieser Auswahl. "
            "Das bedeutet weder einen negativen klinischen Befund noch vollständige Markerabdeckung."
        )
    if ("evidence_stage" in page.rows
            and page.rows["evidence_stage"].astype(str).eq("clinically_validated").any()):
        return "Klinische Evidenzstufe im Ergebnisvertrag vorhanden; Einzelnachweise und Gültigkeitsbereich prüfen."
    return "Forschungshinweise vorhanden; noch keine freigegebene neue Befundungsregel."


def endpoint_evidence_frame(page: ResultPage, endpoint_id: str = "") -> pd.DataFrame:
    """Bounded displayed sample; missing combinations are not neutral values."""
    frame = _eligible_research_rows(page.rows)
    if endpoint_id and "endpoint_id" in frame:
        frame = frame.loc[frame["endpoint_id"].astype(str).eq(endpoint_id)]
    if frame.empty:
        return frame
    if not endpoint_id and "endpoint_id" in frame:
        first = str(frame.iloc[0]["endpoint_id"])
        frame = frame.loc[frame["endpoint_id"].astype(str).eq(first)]
    return frame.head(25).copy()


_ATLAS_COLUMNS = (
    "endpoint_id", "endpoint_name_de", "feature_id", "feature_name_de",
    "evidence_stage", "reference_kind", "claim_ceiling", "failed_gates",
    "reference_independence", "candidate_type", "selection_eligible",
    "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "n_positive",
    "n_negative", "missingness", "delta_auc", "delta_auc_ci_low",
    "delta_auc_ci_high", "baseline_display_name", "n_reference_eligible",
    "n_marker_observed", "n_marker_raw_observed", "n_analysis_restriction_excluded",
    "post_outcome_hypothesis", "baseline_feature_id",
    "independent_modality_observed_n", "clock_alignment_confirmed_n",
    "timing_evidence_note",
)
_ATLAS_REQUIRED = {
    "endpoint_id", "feature_id", "evidence_stage", "reference_kind",
    "claim_ceiling", "reference_independence", "candidate_type",
    "selection_eligible", "roc_auc",
}
_GATE_NAMES_DE = {
    "analytical_validity": "Analytische Gültigkeit",
    "availability": "Messverfügbarkeit",
    "repeatability": "Wiederholbarkeit",
    "replication": "Replikation",
    "adjustment": "Adjustierte Bestätigung",
    "incremental_classical_value": "Zusatznutzen gegenüber klassischem ECG",
    "transform_robustness": "Robustheit gegenüber Transformationen",
    "subgroup_stability": "Stabilität in Untergruppen",
    "effect_ci": "Effekt mit Konfidenzintervall",
    "independent_reference": "Unabhängige Referenz",
    "prespecified_threshold": "Vorab festgelegter Grenzwert",
    "calibration": "Kalibration",
    "net_benefit": "Klinischer Entscheidungsnutzen",
    "external_validation": "Externe Validierung",
    "preview_or_unknown_scope": "Vorschau oder unbekannter Analyseumfang",
}


def marker_gate_rows(row: pd.Series) -> list[tuple[str, str]]:
    """Present only explicit failures; silence in an old contract is not proof."""
    raw = row.get("failed_gates")
    if raw is None or pd.isna(raw):
        return [("Gate-Vertrag", "Nicht vorhanden: kein Gate darf als bestanden gelten")]
    failed = {part.strip() for part in str(raw).split("|") if part.strip()}
    if not failed:
        return [("Gate-Vertrag", "Keine offenen Gates gemeldet; Einzelnachweise im Ergebnisvertrag prüfen")]
    return [(_GATE_NAMES_DE.get(gate, gate.replace("_", " ")), "Nicht belegt")
            for gate in sorted(failed)]


def load_atlas_page(repository: ResultRepository, run_id: str, endpoint_id: str) -> ResultPage:
    """Query a verified artifact, then take a bounded overview over all rows.

    This avoids declaring 'no finding' because the first 200 raw rows happened
    to contain only technical comparisons.  The query never touches a run
    without a complete, compatible validation manifest.
    """
    page = repository.query(ResultQuery(
        perspective="pathologies", run_id=run_id,
        validation_only=True, complete_runs_only=True,
        sort_by="evidence_rank", limit=1,
    ))
    if page.status not in {"ok", "empty"} or page.artifact_path is None:
        return page
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        path = str(page.artifact_path)
        cursor = connection.execute("SELECT * FROM read_parquet(?) LIMIT 0", [path])
        available = {description[0] for description in cursor.description}
        if not _ATLAS_REQUIRED.issubset(available):
            return ResultPage(
                **{**page.__dict__, "rows": pd.DataFrame(), "total_count": 0,
                   "status": "incomplete_contract",
                   "message_de": "Der Ergebnisstand enthält nicht alle für eine sichere Kurzfassung nötigen Vertragsfelder."}
            )
        where = [
            "CAST(selection_eligible AS BOOLEAN) = true",
            "COALESCE(CAST(candidate_type AS VARCHAR), '') <> 'classical_rule'",
            "COALESCE(CAST(reference_independence AS VARCHAR), '') <> 'same_ecg_technical_comparison'",
            "COALESCE(CAST(claim_ceiling AS VARCHAR), '') <> 'agreement_only'",
            "TRY_CAST(roc_auc AS DOUBLE) IS NOT NULL",
        ]
        where.extend(compatible_validation_clauses(available))
        parameters: list[object] = [path]
        if endpoint_id:
            where.append("CAST(endpoint_id AS VARCHAR) = ?")
            parameters.append(endpoint_id)
        where_sql = " AND ".join(where)
        count = int(connection.execute(
            f"SELECT COUNT(*) FROM read_parquet(?) WHERE {where_sql}", parameters,
        ).fetchone()[0])
        projected = ", ".join(
            f'"{name}"' if name in available else f'NULL AS "{name}"'
            for name in _ATLAS_COLUMNS
        )
        per_question_limit = 200 if endpoint_id else 5
        # A global top-200 by AUC can be monopolized by near-aliases from one
        # endpoint. Keep a bounded set per clinical question before the global
        # cap; retain the full bounded detail set for an explicit endpoint.
        rows = connection.execute(
            f"WITH candidates AS (SELECT {projected}, "
            "ROW_NUMBER() OVER (PARTITION BY endpoint_id ORDER BY "
            "CASE evidence_stage WHEN 'clinically_validated' THEN 0 "
            "WHEN 'guideline_candidate' THEN 1 "
            "WHEN 'replicated_association' THEN 2 ELSE 3 END, "
            "TRY_CAST(roc_auc AS DOUBLE) DESC NULLS LAST, feature_id) AS atlas_rank "
            f"FROM read_parquet(?) WHERE {where_sql}) "
            f"SELECT {projected} FROM candidates WHERE atlas_rank <= {per_question_limit} "
            "ORDER BY CASE evidence_stage "
            "WHEN 'clinically_validated' THEN 0 "
            "WHEN 'guideline_candidate' THEN 1 "
            "WHEN 'replicated_association' THEN 2 ELSE 3 END, "
            "TRY_CAST(roc_auc AS DOUBLE) DESC NULLS LAST, endpoint_id, feature_id "
            "LIMIT 200",
            parameters,
        ).fetchdf()
        return ResultPage(**{**page.__dict__, "rows": rows, "total_count": count})
    finally:
        connection.close()


def load_atlas_quality(page: ResultPage) -> tuple[dict[str, int], tuple[tuple[str, int], ...]]:
    """Count overlapping gate failures without implying a sequential funnel."""
    if page.status not in {"ok", "empty"} or page.artifact_path is None:
        return {}, ()
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        path = str(page.artifact_path)
        schema = {item[0] for item in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [path],
        ).description}
        if not {"evidence_stage", "candidate_type", "reference_independence"}.issubset(schema):
            return {}, ()
        conditions = [
            "COALESCE(CAST(candidate_type AS VARCHAR), '') <> 'classical_rule'",
            "COALESCE(CAST(reference_independence AS VARCHAR), '') <> 'same_ecg_technical_comparison'",
        ]
        conditions.extend(compatible_validation_clauses(schema))
        where_sql = " AND ".join(conditions)
        stages = {
            str(stage): int(count)
            for stage, count in connection.execute(
                f"SELECT CAST(evidence_stage AS VARCHAR), COUNT(*) "
                f"FROM read_parquet(?) WHERE {where_sql} "
                "GROUP BY evidence_stage", [path],
            ).fetchall()
            if stage is not None
        }
        if "failed_gates" not in schema:
            return stages, ()
        gates = tuple(
            (str(gate), int(count))
            for gate, count in connection.execute(
                "SELECT gate, COUNT(*) AS n FROM ("
                "SELECT UNNEST(string_split(COALESCE(CAST(failed_gates AS VARCHAR), ''), '|')) AS gate "
                f"FROM read_parquet(?) WHERE {where_sql}"
                ") WHERE gate <> '' GROUP BY gate ORDER BY n DESC, gate LIMIT 12",
                [path],
            ).fetchall()
        )
        return stages, gates
    finally:
        connection.close()


def load_gate_patterns(page: ResultPage) -> pd.DataFrame:
    """Count exact co-failure combinations over the full Validation resultset."""
    if page.status not in {"ok", "empty"} or page.artifact_path is None:
        return pd.DataFrame()
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        path = str(page.artifact_path)
        schema = {item[0] for item in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [path],
        ).description}
        needed = {"failed_gates", "candidate_type", "reference_independence"}
        if not needed.issubset(schema):
            return pd.DataFrame()
        conditions = [
            "COALESCE(CAST(candidate_type AS VARCHAR), '') <> 'classical_rule'",
            "COALESCE(CAST(reference_independence AS VARCHAR), '') <> 'same_ecg_technical_comparison'",
        ]
        conditions.extend(compatible_validation_clauses(schema))
        where_sql = " AND ".join(conditions)
        return connection.execute(
            "WITH filtered AS ("
            "SELECT CASE WHEN TRIM(COALESCE(CAST(failed_gates AS VARCHAR), '')) = '' "
            "THEN '(kein Gate gemeldet)' ELSE "
            "array_to_string(list_sort(string_split(CAST(failed_gates AS VARCHAR), '|')), '|') "
            "END AS pattern FROM read_parquet(?) "
            f"WHERE {where_sql}"
            "), grouped AS (SELECT pattern, COUNT(*) AS n FROM filtered GROUP BY pattern) "
            "SELECT pattern, n, SUM(n) OVER () AS total_pairs, "
            "COUNT(*) OVER () AS distinct_patterns FROM grouped "
            "ORDER BY n DESC, pattern LIMIT 12",
            [path],
        ).fetchdf()
    finally:
        connection.close()


def load_atlas_scope(page: ResultPage) -> dict[str, int]:
    """Count evaluated questions and eligible pairs, never sum overlapping patients."""
    if page.status not in {"ok", "empty"} or page.artifact_path is None:
        return {}
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        path = str(page.artifact_path)
        schema = {column[0] for column in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [path],
        ).description}
        if not {"endpoint_id", "selection_eligible", "candidate_type",
                "reference_independence", "claim_ceiling", "roc_auc"}.issubset(schema):
            return {}
        clauses = compatible_validation_clauses(schema)
        where_sql = " AND ".join(clauses) if clauses else "true"
        eligible_sql = (
            "TRY_CAST(selection_eligible AS BOOLEAN) = true "
            "AND COALESCE(CAST(candidate_type AS VARCHAR), '') <> 'classical_rule' "
            "AND COALESCE(CAST(reference_independence AS VARCHAR), '') "
            "<> 'same_ecg_technical_comparison' "
            "AND COALESCE(CAST(claim_ceiling AS VARCHAR), '') <> 'agreement_only' "
            "AND TRY_CAST(roc_auc AS DOUBLE) IS NOT NULL"
        )
        evaluated, eligible_endpoints, eligible_pairs = connection.execute(
            "SELECT COUNT(DISTINCT endpoint_id), "
            f"COUNT(DISTINCT CASE WHEN {eligible_sql} THEN endpoint_id END), "
            f"COUNT(*) FILTER (WHERE {eligible_sql}) "
            f"FROM read_parquet(?) WHERE {where_sql}", [path],
        ).fetchone()
        return {
            "evaluated_endpoints": int(evaluated),
            "eligible_endpoints": int(eligible_endpoints),
            "eligible_pairs": int(eligible_pairs),
        }
    finally:
        connection.close()


class _AtlasLoad(QThread):
    ready = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, generation: int, repository: ResultRepository, run_id: str, endpoint_id: str):
        super().__init__()
        self.generation = generation
        self.repository = repository
        self.run_id = run_id
        self.endpoint_id = endpoint_id

    def run(self) -> None:
        try:
            repository = ResultRepository(
                self.repository.config,
                expected_feature_code_hash=self.repository.expected_feature_code_hash,
                expected_validation_code_hash=self.repository.expected_validation_code_hash,
            )
            page = load_atlas_page(repository, self.run_id, self.endpoint_id)
            stages, gates = load_atlas_quality(page)
            patterns = load_gate_patterns(page)
            scope = load_atlas_scope(page)
            endpoint = self.endpoint_id or (
                str(page.rows.iloc[0]["endpoint_id"])
                if not page.rows.empty and "endpoint_id" in page.rows else ""
            )
            quality = load_model_quality(page, endpoint)
            availability = load_availability(page, endpoint)
            focus = _eligible_research_rows(page.rows)
            if not focus.empty:
                focus = focus.drop_duplicates("endpoint_id").head(3)
            focus_pairs = tuple(
                (str(row.endpoint_id), str(row.feature_id))
                for row in focus.itertuples()
            )
            paired = load_paired_snapshot(page, self.endpoint_id, focus_pairs=focus_pairs)
            prognosis = load_prognosis(page)
            response = load_response(page)
            scores = load_score_snapshot(repository, self.run_id)
            self.ready.emit(self.generation, AtlasPayload(
                page, stages, gates, patterns, scope, quality, availability, paired,
                prognosis, response, scores,
            ))
        except Exception as exc:
            self.failed.emit(self.generation, f"Ergebnisübersicht konnte nicht geladen werden: {type(exc).__name__}: {exc}")


class _AvailabilityLoad(QThread):
    ready = pyqtSignal(int, str, object)
    failed = pyqtSignal(int, str, str)

    def __init__(self, generation: int, page: ResultPage, endpoint_id: str, marker_id: str):
        super().__init__()
        self.generation = generation
        self.page = page
        self.endpoint_id = endpoint_id
        self.marker_id = marker_id

    def run(self) -> None:
        try:
            snapshot = load_availability(self.page, self.endpoint_id, self.marker_id)
            curves = load_marker_curves(self.page, self.endpoint_id, self.marker_id)
            self.ready.emit(self.generation, self.marker_id, (snapshot, curves))
        except Exception as exc:
            self.failed.emit(
                self.generation, self.marker_id,
                f"Verfügbarkeit konnte nicht geladen werden: {type(exc).__name__}: {exc}",
            )


class ResultsAtlasWidget(QWidget):
    """Simple answer-first page; detailed legacy pages are managed by parent."""

    endpoint_requested = pyqtSignal(str)
    spatial_requested = pyqtSignal(str)
    designs_requested = pyqtSignal()
    scores_requested = pyqtSignal()
    leads_requested = pyqtSignal()
    review_requested = pyqtSignal(str)
    detail_requested = pyqtSignal(int)
    simple_requested = pyqtSignal()
    idle = pyqtSignal()

    def __init__(self, repository: ResultRepository, parent: QWidget | None = None):
        super().__init__(parent)
        self.repository = repository
        self._generation = 0
        self._loaded_generation = -1
        self._workers: list[_AtlasLoad] = []
        self._shutting_down = False
        self._last_request: tuple[str, str, int] | None = None
        self._detailed = False
        self._has_answer_cards = False
        self._map_generation = 0
        self._map_worker: MapLoad | None = None
        self._map_run_id = ""
        self._map_revision = 0
        self._broad_generation = 0
        self._broad_worker: BroadAxisLoad | None = None
        self._broad_run_id = ""
        self._broad_revision = 0
        self._last_page: ResultPage | None = None
        self._detail_frame = pd.DataFrame()
        self._availability_generation = 0
        self._availability_worker: _AvailabilityLoad | None = None
        self._selected_marker_id = ""
        self._selected_endpoint_id = ""
        self._detail_history: list[int] = [0]
        self._detail_history_position = 0
        self._history_navigation = False
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        self.hero = QFrame()
        self.hero.setObjectName("resultsAtlasHero")
        hero_layout = QVBoxLayout(self.hero)
        hero_layout.setContentsMargins(18, 15, 18, 15)
        hero_layout.setSpacing(7)
        self.hero_kicker = QLabel("ERGEBNISATLAS  /  KLINIK ZUERST")
        self.hero_kicker.setObjectName("resultsAtlasKicker")
        hero_layout.addWidget(self.hero_kicker)
        self.headline = QLabel("Ergebnisstand wird geprüft …")
        self.headline.setWordWrap(True)
        self.headline.setObjectName("resultsAtlasHeadline")
        hero_layout.addWidget(self.headline)
        self.result_state = QLabel("Ergebnisstatus wird geprüft …")
        self.result_state.setWordWrap(True)
        self.result_state.setObjectName("resultsAtlasState")
        self.result_state.setProperty("tone", "pending")
        hero_layout.addWidget(self.result_state)
        layout.addWidget(self.hero)
        # The answer cards belong directly below the verdict. Run metadata and
        # evidence accounting must not bury the findings.
        self.cards_host = QWidget()
        self.cards_layout = QGridLayout(self.cards_host)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setHorizontalSpacing(14)
        self.cards_layout.setVerticalSpacing(14)
        layout.addWidget(self.cards_host)
        self.empty_state = QFrame()
        self.empty_state.setObjectName("resultsAtlasEmpty")
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setContentsMargins(18, 17, 18, 17)
        empty_layout.setSpacing(12)
        self.empty_title = QLabel("Ergebnisse folgen nach abgeschlossener Validation")
        self.empty_title.setObjectName("resultsAtlasEmptyTitle")
        empty_layout.addWidget(self.empty_title)
        self.empty_message = QLabel(
            "Bis dahin wird aus fehlenden Ergebnisdateien weder ein negativer Befund "
            "noch eine neue klinische Regel abgeleitet."
        )
        self.empty_message.setWordWrap(True)
        empty_layout.addWidget(self.empty_message)
        empty_guide = QHBoxLayout()
        for index, (title, explanation) in enumerate((
            ("Neue Hinweise", "Welche Fragen zeigen ein Signal?"),
            ("Fairer Vergleich", "Trägt es über klassisches EKG hinaus?"),
            ("Offene Belege", "Was fehlt bis zur klinischen Aussage?"),
        ), start=1):
            step = QFrame()
            step.setObjectName("resultsAtlasEmptyStep")
            step_layout = QVBoxLayout(step)
            number = QLabel(f"{index:02d}  /  {title}")
            number.setObjectName("resultsAtlasEmptyStepTitle")
            step_layout.addWidget(number)
            description = QLabel(explanation)
            description.setWordWrap(True)
            step_layout.addWidget(description)
            empty_guide.addWidget(step, 1)
        empty_layout.addLayout(empty_guide)
        layout.addWidget(self.empty_state)
        self.context = QLabel("Validation · Forschungsergebnisse · keine Einzelfalldiagnose")
        self.context.setWordWrap(True)
        self.context.setObjectName("mimicTextMuted")
        layout.addWidget(self.context)
        self.evidence_summary = QLabel("Evidenzzähler werden geladen …")
        self.evidence_summary.setWordWrap(True)
        self.evidence_summary.setObjectName("mimicInfoBoxNeutral")
        layout.addWidget(self.evidence_summary)
        self.scope_summary = QLabel("Untersuchungsumfang wird geprüft …")
        self.scope_summary.setWordWrap(True)
        self.scope_summary.setObjectName("mimicTextMuted")
        layout.addWidget(self.scope_summary)
        self.research_host = QFrame()
        self.research_host.setFrameShape(QFrame.Shape.StyledPanel)
        research_layout = QVBoxLayout(self.research_host)
        self.research_summary = QLabel("Weitere Forschungsanalysen werden geprüft.")
        self.research_summary.setWordWrap(True)
        research_layout.addWidget(self.research_summary)
        research_buttons = QHBoxLayout()
        self.score_summary_button = QPushButton("Scores prüfen")
        self.score_summary_button.setToolTip(
            "Öffnet die OOF-Forschungsscores; keine extern validierten Diagnoseregeln."
        )
        self.score_summary_button.clicked.connect(
            lambda _checked=False: self.detail_requested.emit(
                self.detail_tabs.indexOf(self.score_view)
            )
        )
        research_buttons.addWidget(self.score_summary_button)
        self.prognosis_summary_button = QPushButton("Prognose prüfen")
        self.prognosis_summary_button.setToolTip(
            "Öffnet beobachtende Zeit-bis-Ereignis-Analysen, keine Behandlungseffekte."
        )
        self.prognosis_summary_button.clicked.connect(
            lambda _checked=False: self.detail_requested.emit(
                self.detail_tabs.indexOf(self.prognosis_view)
            )
        )
        research_buttons.addWidget(self.prognosis_summary_button)
        self.response_summary_button = QPushButton("Response prüfen")
        self.response_summary_button.setToolTip(
            "Öffnet beobachtete Vorher-nachher-Änderungen, keine Therapieempfehlungen."
        )
        self.response_summary_button.clicked.connect(
            lambda _checked=False: self.detail_requested.emit(
                self.detail_tabs.indexOf(self.response_view)
            )
        )
        research_buttons.addWidget(self.response_summary_button)
        research_buttons.addStretch(1)
        research_layout.addLayout(research_buttons)
        layout.addWidget(self.research_host)
        self.research_host.hide()
        self.comparison_summary = QLabel("Klassischer ECG-Vergleich wird geprüft …")
        self.comparison_summary.setWordWrap(True)
        self.comparison_summary.setObjectName("mimicInfoBoxNeutral")
        layout.addWidget(self.comparison_summary)
        self.limits_summary = QLabel("Aussagegrenzen werden geprüft …")
        self.limits_summary.setWordWrap(True)
        self.limits_summary.setObjectName("mimicTextMuted")
        layout.addWidget(self.limits_summary)
        self.nonfindings_summary = QLabel(
            "Nicht-Befunde und klassische Vergleiche werden geprüft …"
        )
        self.nonfindings_summary.setWordWrap(True)
        self.nonfindings_summary.setObjectName("mimicInfoBoxNeutral")
        layout.addWidget(self.nonfindings_summary)
        self.detail_panel = QWidget()
        detail_layout = QVBoxLayout(self.detail_panel)
        self.detail_heading = QLabel("Belege zu einer klinischen Frage")
        self.detail_heading.setStyleSheet("font-weight: 650; font-size: 16px;")
        detail_layout.addWidget(self.detail_heading)
        self.detail_note = QLabel("Für die ausgewählte Frage liegen noch keine Vergleichsdaten vor.")
        self.detail_note.setWordWrap(True)
        detail_layout.addWidget(self.detail_note)
        self.availability_note = QLabel("Messverfügbarkeit noch nicht bestimmt.")
        self.availability_note.setWordWrap(True)
        detail_layout.addWidget(self.availability_note)
        self.availability_bar = QProgressBar()
        self.availability_bar.setRange(0, 1000)
        self.availability_bar.setTextVisible(True)
        self.availability_bar.hide()
        detail_layout.addWidget(self.availability_bar)
        self.spatial_button = QPushButton("Gruppen-VCG in 3D und drei 2D-Projektionen öffnen")
        self.spatial_button.setToolTip(
            "Öffnet die vorhandene Gruppen-VCG-Forschungsansicht. "
            "Diese geometrische Darstellung ist kein individueller anatomischer Fokusnachweis."
        )
        self.spatial_button.clicked.connect(
            lambda _checked=False: self.spatial_requested.emit(self._selected_endpoint_id)
        )
        detail_layout.addWidget(self.spatial_button)
        design_row = QGridLayout()
        designs_button = QPushButton("Prognose, Therapie und Verlaufsanalysen")
        designs_button.setToolTip(
            "Öffnet die vorhandenen zielgerichteten Forschungsdesigns. "
            "Behandlungs-Response ist nicht automatisch kausaler Nutzen."
        )
        designs_button.clicked.connect(
            lambda _checked=False: self.designs_requested.emit()
        )
        design_row.addWidget(designs_button, 0, 0)
        scores_button = QPushButton("Scores und Fokusforschung")
        scores_button.setToolTip(
            "Öffnet die vorhandenen Score- und Lokalisations-Forschungsansichten. "
            "Nur tatsächlich berechnete und validierte Ziele werden interpretiert."
        )
        scores_button.clicked.connect(
            lambda _checked=False: self.scores_requested.emit()
        )
        design_row.addWidget(scores_button, 0, 1)
        leads_button = QPushButton("Ableitungsrang je Endpunkt")
        leads_button.setToolTip(
            "Öffnet gepaarte Development-Ableitungsvergleiche mit ΔAUC, "
            "Konfidenzintervallen und Fallzahlen; Rang ist deskriptiv."
        )
        leads_button.clicked.connect(
            lambda _checked=False: self.leads_requested.emit()
        )
        design_row.addWidget(leads_button, 1, 0)
        review_button = QPushButton("Beispiel- und Diskordanzfälle prüfen")
        review_button.setToolTip(
            "Öffnet ausschließlich die lokale Reviewliste dieses Ergebnisstands. "
            "Ein Fall wird nicht allein aus einer aggregierten Markerzeile zugeordnet."
        )
        review_button.clicked.connect(
            lambda _checked=False: self.review_requested.emit(
                self._last_request[0] if self._last_request else ""
            )
        )
        design_row.addWidget(review_button, 1, 1)
        detail_layout.addLayout(design_row)
        export_row = QHBoxLayout()
        plot_export = QPushButton("Diagramm als PNG/SVG")
        plot_export.setToolTip(
            "Exportiert das sichtbare Endpunktdiagramm lokal mit separater Herkunftsdatei."
        )
        plot_export.clicked.connect(self._export_endpoint_plot)
        export_row.addWidget(plot_export)
        table_export = QPushButton("Markertabelle als CSV")
        table_export.setToolTip(
            "Exportiert die begrenzte angezeigte Markerauswahl lokal als CSV."
        )
        table_export.clicked.connect(self._export_endpoint_table)
        export_row.addWidget(table_export)
        export_row.addStretch(1)
        detail_layout.addLayout(export_row)
        self.endpoint_plot = pg.PlotWidget()
        self.endpoint_plot.setMinimumHeight(240)
        self.endpoint_plot.setLabel("bottom", "ROC-AUC auf Validation")
        self.endpoint_plot.setXRange(0, 1, padding=0)
        self.endpoint_plot.showGrid(x=True, y=False, alpha=.15)
        self.detail_table = QTableWidget(0, 5)
        self.detail_table.setHorizontalHeaderLabels((
            "Marker", "AUC (95-%-CI)", "Fälle / Kontrollen", "Referenz", "Klassischer Vergleich",
        ))
        self.detail_table.setAlternatingRowColors(True)
        self.detail_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.detail_table.setMinimumHeight(240)
        self.detail_table.cellClicked.connect(self._select_marker_row)
        self.detail_table.cellDoubleClicked.connect(self._open_marker_row)
        self.endpoint_display = QTabWidget()
        self.endpoint_display.addTab(self.endpoint_plot, "AUC mit Intervall")
        self.endpoint_display.addTab(self.detail_table, "Markerzahlen")
        self.endpoint_display.setTabEnabled(0, False)
        self.endpoint_display.setCurrentIndex(1)
        detail_layout.addWidget(self.endpoint_display, 1)
        self.detail_navigation = QWidget()
        navigation = QHBoxLayout(self.detail_navigation)
        navigation.setContentsMargins(0, 0, 0, 0)
        back_button = QPushButton("← Einfacher Überblick")
        back_button.setToolTip("Zur verständlichen Startansicht desselben Ergebnisstands zurückkehren")
        back_button.clicked.connect(lambda _checked=False: self.simple_requested.emit())
        navigation.addWidget(back_button)
        self.history_back_button = QPushButton("← Zurück")
        self.history_back_button.setToolTip("Zur vorherigen Forschungsansicht desselben Ergebnisstands")
        self.history_back_button.clicked.connect(lambda: self._navigate_history(-1))
        self.history_back_button.setEnabled(False)
        navigation.addWidget(self.history_back_button)
        self.history_forward_button = QPushButton("Weiter →")
        self.history_forward_button.setToolTip("Zur zuletzt verlassenen Forschungsansicht")
        self.history_forward_button.clicked.connect(lambda: self._navigate_history(1))
        self.history_forward_button.setEnabled(False)
        navigation.addWidget(self.history_forward_button)
        self.breadcrumb = QLabel("Ergebnisatlas › klinische Frage")
        self.breadcrumb.setObjectName("mimicTextMuted")
        self.breadcrumb.setWordWrap(True)
        navigation.addWidget(self.breadcrumb, 1)
        layout.addWidget(self.detail_navigation)
        self.detail_navigation.hide()
        self.detail_tabs = QTabWidget()
        self.detail_tabs.addTab(self.detail_panel, "Klinische Frage")
        self.map_view = AtlasMapWidget()
        self.map_view.endpoint_requested.connect(self.endpoint_requested.emit)
        self.detail_tabs.addTab(self.map_view, "Markerlandkarte")
        self.baseline_view = AtlasRuleAuditWidget()
        self.baseline_view.endpoint_requested.connect(self.endpoint_requested.emit)
        self.detail_tabs.addTab(self.baseline_view, "Klassische Regeln")
        self.quality_panel = QWidget()
        quality_layout = QVBoxLayout(self.quality_panel)
        quality_title = QLabel("Evidenz und offene Prüfungen")
        quality_title.setStyleSheet("font-weight: 650; font-size: 16px;")
        quality_layout.addWidget(quality_title)
        self.quality_note = QLabel(
            "Gate-Zahlen sind überlappende Marker–Endpunkt-Paare, kein Patiententrichter. "
            "Ein Ergebnis kann an mehreren Prüfungen zugleich scheitern."
        )
        self.quality_note.setWordWrap(True)
        quality_layout.addWidget(self.quality_note)
        self.marker_gate_note = QLabel("Marker auswählen, um seine offenen Nachweise zu sehen.")
        self.marker_gate_note.setWordWrap(True)
        self.marker_gate_note.setObjectName("mimicInfoBoxNeutral")
        quality_layout.addWidget(self.marker_gate_note)
        self.marker_gate_table = QTableWidget(0, 2)
        self.marker_gate_table.setHorizontalHeaderLabels(("Prüfung", "Status"))
        self.marker_gate_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.marker_gate_table.setMinimumHeight(160)
        quality_layout.addWidget(self.marker_gate_table)
        self.quality_table = QTableWidget(0, 2)
        self.quality_table.setHorizontalHeaderLabels(("Offene Prüfung", "Betroffene Paare"))
        self.quality_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        quality_layout.addWidget(self.quality_table)
        self.pattern_note = QLabel(
            "Gemeinsam fehlende Nachweise: vollständige Validation-Marker–Endpunkt-Paare, "
            "keine Patienten und keine sequenzielle Evidenzleiter."
        )
        self.pattern_note.setWordWrap(True)
        quality_layout.addWidget(self.pattern_note)
        self.pattern_plot = pg.PlotWidget()
        self.pattern_plot.setMinimumHeight(210)
        self.pattern_plot.setLabel("bottom", "Marker–Endpunkt-Paare")
        quality_layout.addWidget(self.pattern_plot)
        self.pattern_table = QTableWidget(0, 3)
        self.pattern_table.setHorizontalHeaderLabels((
            "Gemeinsam offene Nachweise", "Paare", "Anteil am geprüften Umfang",
        ))
        self.pattern_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        quality_layout.addWidget(self.pattern_table)
        pattern_actions = QHBoxLayout()
        pattern_plot_export = QPushButton("Gate-Muster als PNG/SVG")
        pattern_plot_export.setToolTip(
            "Exportiert die zwölf häufigsten gemeinsamen Gate-Fehlmuster mit Herkunftsdatei."
        )
        pattern_plot_export.clicked.connect(self._export_gate_pattern_plot)
        pattern_actions.addWidget(pattern_plot_export)
        pattern_table_export = QPushButton("Gate-Muster als CSV")
        pattern_table_export.setToolTip(
            "Exportiert Musterhäufigkeiten; die Zeilen sind Marker–Endpunkt-Paare, keine Patienten."
        )
        pattern_table_export.clicked.connect(self._export_gate_patterns)
        pattern_actions.addWidget(pattern_table_export)
        pattern_actions.addStretch(1)
        quality_layout.addLayout(pattern_actions)
        self.detail_tabs.addTab(self.quality_panel, "Evidenz & Robustheit")
        self.availability_view = AtlasAvailabilityWidget()
        self.detail_tabs.addTab(self.availability_view, "Verfügbarkeit & Gruppen")
        self.curve_view = AtlasCurveWidget()
        self.detail_tabs.addTab(self.curve_view, "ROC, PR & Verteilung")
        self.paired_view = AtlasPairedWidget()
        self.detail_tabs.addTab(self.paired_view, "VCG gegen klassisches EKG")
        self.model_quality_view = AtlasModelQualityWidget()
        self.detail_tabs.addTab(self.model_quality_view, "Kalibration & Nutzen")
        self.discovery_view = AtlasDiscoveryWidget()
        self.detail_tabs.addTab(self.discovery_view, "Breiter Scan")
        self.broad_view = AtlasBroadAxisWidget()
        self.broad_tab_index = self.detail_tabs.addTab(self.broad_view, "Alle Achsentests")
        self.transport_view = AtlasTransportWidget()
        self.detail_tabs.addTab(self.transport_view, "Externe Kohorten")
        self.prognosis_view = AtlasPrognosisWidget()
        self.detail_tabs.addTab(self.prognosis_view, "Prognose")
        self.response_view = AtlasResponseWidget()
        self.detail_tabs.addTab(self.response_view, "Therapie-Response")
        self.score_view = AtlasScoreWidget()
        self.detail_tabs.addTab(self.score_view, "Forschungsscores")
        self.marker_view = AtlasMarkerWidget()
        self.marker_view.idle.connect(self._maybe_idle)
        self.marker_tab_index = self.detail_tabs.addTab(self.marker_view, "Marker-Dossier")
        self.detail_tabs.currentChanged.connect(self._on_detail_tab_changed)
        self.detail_tabs.tabBar().hide()
        self.detail_shell = QWidget()
        detail_shell_layout = QHBoxLayout(self.detail_shell)
        detail_shell_layout.setContentsMargins(0, 0, 0, 0)
        nav_column = QVBoxLayout()
        nav_title = QLabel("Forschungsansichten")
        nav_title.setStyleSheet("font-weight: 650; font-size: 14px;")
        nav_column.addWidget(nav_title)
        self.detail_nav = QTreeWidget()
        self.detail_nav.setHeaderHidden(True)
        self.detail_nav.setMinimumWidth(195)
        self.detail_nav.setMaximumWidth(245)
        self.detail_nav.setObjectName("resultsAtlasNavigation")
        self._detail_nav_items: dict[int, QTreeWidgetItem] = {}
        groups = (
            ("Klinische Frage", (0, 5, 4, self.marker_tab_index)),
            ("Vergleich & Nutzen", (6, 2, 7)),
            ("Entdeckung & Evidenz", (1, 3, 8, self.broad_tab_index)),
            ("Übertragbarkeit & Verlauf", (10, 11, 12, 13)),
        )
        for group_name, indices in groups:
            group_item = QTreeWidgetItem([group_name])
            group_item.setFlags(group_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.detail_nav.addTopLevelItem(group_item)
            for index in indices:
                child = QTreeWidgetItem([self.detail_tabs.tabText(index)])
                child.setData(0, Qt.ItemDataRole.UserRole, index)
                child.setToolTip(0, f"{group_name} › {self.detail_tabs.tabText(index)}")
                group_item.addChild(child)
                self._detail_nav_items[index] = child
        self.detail_nav.expandAll()
        self.detail_nav.itemClicked.connect(self._navigate_detail)
        self.detail_nav.itemActivated.connect(self._navigate_detail)
        self.detail_nav.setCurrentItem(self._detail_nav_items[0])
        nav_column.addWidget(self.detail_nav, 1)
        detail_shell_layout.addLayout(nav_column)
        self.detail_scroll = QScrollArea()
        self.detail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.detail_tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.detail_scroll.setWidget(self.detail_tabs)
        detail_shell_layout.addWidget(self.detail_scroll, 1)
        layout.addWidget(self.detail_shell)
        self.detail_shell.hide()
        self.footer = QLabel("AUC beschreibt Trennung in der ausgewerteten Gruppe. Sie ist weder Erkrankungswahrscheinlichkeit noch klinischer Nutzen.")
        self.footer.setWordWrap(True)
        self.footer.setObjectName("mimicTextMuted")
        layout.addWidget(self.footer)
        # The simple answer can contain three substantial cards. Scroll its
        # evidence/context below the fixed verdict instead of forcing the tab
        # (or even the main window) to grow beyond a laptop viewport.
        self.simple_panel = QWidget()
        simple_layout = QVBoxLayout(self.simple_panel)
        for widget in (
            self.cards_host, self.empty_state, self.evidence_summary, self.scope_summary,
            self.research_host, self.comparison_summary, self.limits_summary,
            self.nonfindings_summary, self.footer,
        ):
            layout.removeWidget(widget)
            simple_layout.addWidget(widget)
        simple_layout.addStretch(1)
        self.simple_scroll = QScrollArea()
        self.simple_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.simple_scroll.setWidgetResizable(True)
        self.simple_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.simple_scroll.setWidget(self.simple_panel)
        layout.insertWidget(layout.indexOf(self.detail_shell), self.simple_scroll, 1)

    def refresh(self, run_id: str, endpoint_id: str = "", *, revision: int = 0) -> None:
        if self._shutting_down:
            return
        request = (run_id, endpoint_id, revision)
        if request == self._last_request:
            return
        if self._last_request is not None and run_id != self._last_request[0]:
            self._detail_history = [self.detail_tabs.currentIndex()]
            self._detail_history_position = 0
            self._update_history_buttons()
        self._last_request = request
        self._generation += 1
        self._availability_generation += 1
        self._selected_marker_id = ""
        self._selected_endpoint_id = endpoint_id
        self._last_page = None
        self._detail_frame = pd.DataFrame()
        self.research_host.hide()
        self._clear_cards()
        self._has_answer_cards = False
        self.empty_state.setVisible(not self._detailed)
        self.empty_title.setText("Ergebnisse folgen nach abgeschlossener Validation")
        self.empty_message.setText(
            "Bis dahin wird aus fehlenden Ergebnisdateien weder ein negativer Befund "
            "noch eine neue klinische Regel abgeleitet."
        )
        self.nonfindings_summary.setText(
            "Nicht-Befunde und klassische Vergleiche werden geprüft …"
        )
        self.endpoint_plot.clear()
        self.endpoint_display.setTabEnabled(0, False)
        self.endpoint_display.setCurrentIndex(1)
        self.detail_table.setRowCount(0)
        self._update_breadcrumb()
        self.availability_bar.hide()
        self.availability_note.setText("Messverfügbarkeit noch nicht bestimmt.")
        self.quality_table.setRowCount(0)
        self.pattern_plot.clear()
        self.pattern_table.setRowCount(0)
        self._gate_patterns = pd.DataFrame()
        self.marker_gate_table.setRowCount(0)
        self.marker_gate_note.setText("Marker auswählen, um seine offenen Nachweise zu sehen.")
        self.model_quality_view.set_snapshot(ModelQualitySnapshot(
            "pending", "Für diesen Endpunkt werden vorhandene Modellkurven geprüft.",
            pd.DataFrame(), pd.DataFrame(), endpoint_id,
        ))
        self.availability_view.set_snapshot(AvailabilitySnapshot(
            endpoint_id, "", None, None, None, None, pd.DataFrame(),
            "Messverfügbarkeit und Untergruppen werden geprüft.",
        ))
        self.paired_view.set_snapshot(PairedSnapshot(
            "pending", "Gepaarte klassische Vergleiche werden geprüft.",
            pd.DataFrame(), endpoint_id, run_id,
        ))
        for view in (self.model_quality_view, self.availability_view, self.paired_view,
                     self.prognosis_view, self.response_view, self.score_view):
            self._set_nav_status(view, "pending")
        self.comparison_summary.setText("Klassischer ECG-Vergleich wird geprüft …")
        self.limits_summary.setText("Aussagegrenzen werden geprüft …")
        self.comparison_summary.show()
        self.limits_summary.show()
        if run_id != self._map_run_id or revision != self._map_revision:
            self._map_generation += 1
            self._map_run_id = ""
            self._map_revision = 0
            self.map_view.set_frame(pd.DataFrame())
            self.baseline_view.set_frame(pd.DataFrame())
            self.baseline_view.set_snapshot(RuleAuditSnapshot(
                pd.DataFrame(), pd.DataFrame(), "pending",
                "Klassische Regeln und Geräteberichte werden geprüft …",
            ))
            self.discovery_view.set_snapshot(DiscoverySnapshot(
                "pending", "Der breite Scan wird für diesen Ergebnisstand geprüft.",
                pd.DataFrame(), 0, pd.DataFrame(), run_id, "",
            ))
            self.transport_view.set_snapshot(TransportSnapshot(
                "pending", "Externe Kohorten werden für diesen Ergebnisstand geprüft.",
                pd.DataFrame(), pd.DataFrame(), run_id, "",
            ))
            self._set_nav_status(self.discovery_view, "pending")
            self._set_nav_status(self.transport_view, "pending")
        broad_revision = self._current_broad_revision(run_id)
        if run_id != self._broad_run_id or broad_revision != self._broad_revision:
            self._broad_generation += 1
            self._broad_run_id = ""
            self._broad_revision = 0
            self.broad_view.set_snapshot(BroadAxisSnapshot(
                "pending", "Der vollständige breite Achsenscan wird für diesen Run geprüft.",
                pd.DataFrame(), pd.DataFrame(), run_id,
            ))
            self._set_nav_status(self.broad_view, "pending")
        if not run_id:
            self.headline.setText("Die Analyse läuft oder es liegt noch kein kompatibler, abgeschlossener Ergebnisstand vor.")
            self.result_state.setText("Noch nicht berechnet: Keine abgeschlossenen Validation-Ergebnisse ausgewählt.")
            self._set_hero_tone("pending")
            self.context.setText("Keine Ergebnisartefakte ausgewählt · Locked Test bleibt versiegelt")
            self.evidence_summary.setText("Noch keine belastbare Ergebnisaussage verfügbar.")
            self.scope_summary.setText("Untersuchungsumfang wird erst aus abgeschlossenen Ergebnissen angezeigt.")
            self.comparison_summary.setText("Klassischer ECG-Vergleich: noch keine abgeschlossenen Ergebnisse.")
            self.limits_summary.setText("Kalibration, Entscheidungsnutzen und externe Bestätigung: noch nicht beurteilbar.")
            self.nonfindings_summary.setText(
                "Nicht-Befunde sind erst nach abgeschlossener gepaarter Validation interpretierbar."
            )
            return
        if self._workers or self._map_worker is not None:
            self.headline.setText("Die aktuelle Ergebnisabfrage wird abgeschlossen; danach wird die neue Auswahl geladen …")
            return
        self.headline.setText("Der Ergebnisstand wird gelesen …")
        self.result_state.setText("Ergebnisstatus wird geprüft …")
        self._set_hero_tone("pending")
        self.context.setText(f"Run {run_id} · Validation · Forschungsergebnisse")
        self.evidence_summary.setText("Evidenzzähler werden geladen …")
        self.scope_summary.setText("Untersuchungsumfang wird geprüft …")
        self._start_load(self._generation, run_id, endpoint_id)

    def _start_load(self, generation: int, run_id: str, endpoint_id: str) -> None:
        if self._shutting_down:
            return
        worker = _AtlasLoad(generation, self.repository, run_id, endpoint_id)
        worker.ready.connect(self._loaded)
        worker.failed.connect(self._failed)
        worker.finished.connect(lambda w=worker: self._retire(w))
        self._workers.append(worker)
        worker.start()

    def set_repository(self, repository: ResultRepository) -> None:
        if repository is not self.repository:
            self.repository = repository
            self._last_request = None
            self._map_generation += 1
            self._map_run_id = ""
            self._map_revision = 0
            self.map_view.set_frame(pd.DataFrame())
            self.baseline_view.set_frame(pd.DataFrame())
            self.baseline_view.set_snapshot(RuleAuditSnapshot(
                pd.DataFrame(), pd.DataFrame(), "pending",
                "Noch kein kompatibler Regelprüfstand ausgewählt.",
            ))
            self.discovery_view.set_snapshot(DiscoverySnapshot(
                "pending", "Noch kein kompatibler Discovery-Stand ausgewählt.",
                pd.DataFrame(), 0, pd.DataFrame(), "", "",
            ))
            self.transport_view.set_snapshot(TransportSnapshot(
                "pending", "Noch kein kompatibler externer Stand ausgewählt.",
                pd.DataFrame(), pd.DataFrame(), "", "",
            ))
            self._broad_generation += 1
            self._broad_run_id = ""
            self._broad_revision = 0
            self.broad_view.set_snapshot(BroadAxisSnapshot(
                "pending", "Noch kein breiter Achsenscan ausgewählt.",
                pd.DataFrame(), pd.DataFrame(), "",
            ))

    def set_detailed(self, detailed: bool) -> None:
        self._detailed = detailed
        root_layout = self.layout()
        root_layout.setStretch(root_layout.indexOf(self.simple_scroll), 0 if detailed else 1)
        root_layout.setStretch(root_layout.indexOf(self.detail_shell), 1 if detailed else 0)
        self.simple_scroll.setVisible(not detailed)
        self.cards_host.setVisible(not detailed)
        self.empty_state.setVisible(not detailed and not self._has_answer_cards)
        self.evidence_summary.setVisible(not detailed)
        self.scope_summary.setVisible(not detailed)
        self.comparison_summary.setVisible(not detailed and not self._has_answer_cards)
        self.limits_summary.setVisible(not detailed and not self._has_answer_cards)
        self.nonfindings_summary.setVisible(not detailed)
        self.research_host.setVisible(not detailed and bool(self.research_host.property("has_data")))
        self.detail_navigation.setVisible(detailed)
        self.detail_shell.setVisible(detailed)
        self.detail_tabs.setVisible(detailed)
        if detailed:
            self._resize_detail_scroll()
            self._request_map()
            self._request_broad_axis()

    def _navigate_detail(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(index, int):
            self.detail_tabs.setCurrentIndex(index)

    def _set_nav_status(self, view: QWidget, status: str) -> None:
        """Show availability without treating pending/empty results as negative findings."""
        index = self.detail_tabs.indexOf(view)
        item = self._detail_nav_items.get(index)
        if item is None:
            return
        label = {
            "ok": "Ergebnisse", "empty": "ohne Werte",
            "pending": "ausstehend", "missing": "ausstehend",
            "incompatible": "prüfen", "invalid": "prüfen", "failed": "Fehler",
        }.get(status, "Status unklar")
        title = self.detail_tabs.tabText(index)
        item.setText(0, f"{title} · {label}")
        item.setToolTip(0, f"{title}: {label}. Details und Aussagegrenzen in der Ansicht.")

    def _set_hero_tone(self, tone: str) -> None:
        if self.result_state.property("tone") == tone:
            return
        self.result_state.setProperty("tone", tone)
        self.result_state.style().unpolish(self.result_state)
        self.result_state.style().polish(self.result_state)

    def _on_detail_tab_changed(self, index: int) -> None:
        self._resize_detail_scroll()
        if not self._history_navigation and index != self._detail_history[self._detail_history_position]:
            del self._detail_history[self._detail_history_position + 1:]
            self._detail_history.append(index)
            self._detail_history_position = len(self._detail_history) - 1
        self._update_history_buttons()
        self._update_breadcrumb()
        item = self._detail_nav_items.get(index)
        if item is not None and self.detail_nav.currentItem() is not item:
            self.detail_nav.setCurrentItem(item)
        if self._detailed:
            self._request_map()
            self._request_broad_axis()

    def _navigate_history(self, direction: int) -> None:
        target = self._detail_history_position + direction
        if not 0 <= target < len(self._detail_history):
            return
        self._detail_history_position = target
        self._history_navigation = True
        try:
            self.detail_tabs.setCurrentIndex(self._detail_history[target])
        finally:
            self._history_navigation = False
        self._update_history_buttons()

    def _update_history_buttons(self) -> None:
        self.history_back_button.setEnabled(self._detail_history_position > 0)
        self.history_forward_button.setEnabled(
            self._detail_history_position < len(self._detail_history) - 1
        )

    def _update_breadcrumb(self) -> None:
        question = self._selected_endpoint_id or "alle klinischen Fragen"
        self.breadcrumb.setText(
            f"Ergebnisatlas › {self.detail_tabs.tabText(self.detail_tabs.currentIndex())} › {question}"
        )

    def _resize_detail_scroll(self) -> None:
        current = self.detail_tabs.currentWidget()
        if current is None:
            return
        self.detail_tabs.setMinimumHeight(max(0, current.minimumSizeHint().height() + 8))
        self.detail_scroll.verticalScrollBar().setValue(0)

    def _retire(self, worker: _AtlasLoad) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()
        if self._shutting_down:
            self._maybe_idle()
            return
        if not self._workers and worker.generation != self._generation and self._last_request:
            run_id, endpoint_id, _revision = self._last_request
            if run_id:
                self._start_load(self._generation, run_id, endpoint_id)
                return
        self._request_map()
        self._request_broad_axis()

    def _request_map(self) -> None:
        if (self._shutting_down or not self._detailed
                or self.detail_tabs.currentIndex() not in {1, 2, 8, 10}
                or self._workers or self._map_worker is not None):
            return
        run_id = self._last_request[0] if self._last_request else ""
        revision = self._last_request[2] if self._last_request else 0
        if not run_id or (self._map_run_id == run_id and self._map_revision == revision):
            return
        self._map_generation += 1
        self._map_run_id = run_id
        self._map_revision = revision
        self.map_view.selection.setText("Die klinische Landkarte wird vorbereitet …")
        worker = MapLoad(self._map_generation, self.repository, run_id)
        self._map_worker = worker
        worker.ready.connect(self._map_loaded)
        worker.failed.connect(self._map_failed)
        worker.finished.connect(lambda w=worker: self._retire_map(w))
        worker.start()

    def _map_loaded(
        self, generation: int,
        frames: tuple[pd.DataFrame, pd.DataFrame, DiscoverySnapshot, TransportSnapshot, RuleAuditSnapshot],
    ) -> None:
        if self._shutting_down:
            return
        if generation == self._map_generation:
            page = self._last_page
            if page is not None:
                self.map_view.set_provenance(page.run_id, page.snapshot_hash)
                self.baseline_view.set_provenance(page.run_id, page.snapshot_hash)
            self.map_view.set_frame(frames[0])
            self.baseline_view.set_frame(frames[1])
            self.baseline_view.set_snapshot(frames[4])
            self.discovery_view.set_snapshot(frames[2])
            self.transport_view.set_snapshot(frames[3])
            self._set_nav_status(self.discovery_view, frames[2].status)
            self._set_nav_status(self.transport_view, frames[3].status)

    def _map_failed(self, generation: int, message: str) -> None:
        if self._shutting_down:
            return
        if generation == self._map_generation:
            self.map_view.selection.setText(message)
            self.baseline_view.note.setText(message)
            self.discovery_view.set_snapshot(DiscoverySnapshot(
                "failed", message, pd.DataFrame(), 0, pd.DataFrame(),
                self._map_run_id, "",
            ))
            self.transport_view.set_snapshot(TransportSnapshot(
                "failed", message, pd.DataFrame(), pd.DataFrame(),
                self._map_run_id, "",
            ))
            self._set_nav_status(self.discovery_view, "failed")
            self._set_nav_status(self.transport_view, "failed")

    def _retire_map(self, worker: MapLoad) -> None:
        if self._map_worker is worker:
            self._map_worker = None
        worker.deleteLater()
        if self._shutting_down:
            self._maybe_idle()
            return
        if self._loaded_generation != self._generation and self._last_request:
            run_id, endpoint_id, _revision = self._last_request
            if run_id and not self._workers:
                self._start_load(self._generation, run_id, endpoint_id)
                return
        self._request_map()
        self._request_broad_axis()

    def _request_broad_axis(self) -> None:
        if (self._shutting_down or not self._detailed
                or self.detail_tabs.currentIndex() != self.broad_tab_index
                or self._workers or self._map_worker is not None
                or self._broad_worker is not None):
            return
        run_id = self._last_request[0] if self._last_request else ""
        revision = self._current_broad_revision(run_id)
        if not run_id or (run_id == self._broad_run_id and revision == self._broad_revision):
            return
        self._broad_generation += 1
        self._broad_run_id = run_id
        self._broad_revision = revision
        worker = BroadAxisLoad(self._broad_generation, self.repository, run_id)
        self._broad_worker = worker
        worker.ready.connect(self._broad_loaded)
        worker.failed.connect(self._broad_failed)
        worker.finished.connect(lambda w=worker: self._retire_broad(w))
        worker.start(QThread.Priority.LowPriority)

    def _current_broad_revision(self, run_id: str) -> int:
        if not run_id or self.repository is None:
            return 0
        path = self.repository.config.runs_root / run_id / "broad_axes" / "broad_axes.manifest.json"
        try:
            stat = path.stat()
        except OSError:
            return 0
        return hash((stat.st_mtime_ns, stat.st_size))

    def _broad_loaded(self, generation: int, snapshot: BroadAxisSnapshot) -> None:
        if not self._shutting_down and generation == self._broad_generation:
            self.broad_view.set_snapshot(snapshot)
            self._set_nav_status(self.broad_view, snapshot.status)

    def _broad_failed(self, generation: int, message: str) -> None:
        if not self._shutting_down and generation == self._broad_generation:
            self.broad_view.set_snapshot(BroadAxisSnapshot(
                "failed", message, pd.DataFrame(), pd.DataFrame(), self._broad_run_id,
            ))
            self._set_nav_status(self.broad_view, "failed")

    def _retire_broad(self, worker: BroadAxisLoad) -> None:
        if self._broad_worker is worker:
            self._broad_worker = None
        worker.deleteLater()
        if self._shutting_down:
            self._maybe_idle()
        elif worker.generation != self._broad_generation:
            self._request_broad_axis()

    def request_shutdown(self) -> bool:
        """Stop queuing reads; let active Qt threads finish without terminate()."""
        self._shutting_down = True
        marker_idle = self.marker_view.request_shutdown()
        for worker in self._workers:
            worker.requestInterruption()
        if self._map_worker is not None:
            self._map_worker.requestInterruption()
        if self._availability_worker is not None:
            self._availability_worker.requestInterruption()
        if self._broad_worker is not None:
            self._broad_worker.requestInterruption()
        return marker_idle and not any(worker.isRunning() for worker in self._workers) and (
            self._map_worker is None or not self._map_worker.isRunning()
        ) and (self._availability_worker is None or not self._availability_worker.isRunning()) and (
            self._broad_worker is None or not self._broad_worker.isRunning()
        )

    def _maybe_idle(self) -> None:
        if self._shutting_down and self.request_shutdown():
            self.idle.emit()

    def _loaded(self, generation: int, payload: AtlasPayload) -> None:
        if self._shutting_down:
            return
        if generation != self._generation:
            return
        self._loaded_generation = generation
        page = payload.page
        self._last_page = page
        self.prognosis_view.set_snapshot(payload.prognosis)
        self.response_view.set_snapshot(payload.response)
        self.score_view.set_snapshot(payload.scores)
        self._set_nav_status(self.prognosis_view, payload.prognosis.status)
        self._set_nav_status(self.response_view, payload.response.status)
        self._set_nav_status(self.score_view, payload.scores.status)
        self._show_research_summary(payload)
        if page.status not in {"ok", "empty"} and self._last_request:
            self._map_run_id = self._last_request[0]
            self._map_revision = self._last_request[2]
        headline, cards = cards_from_page(page, payload.paired)
        self.headline.setText(headline)
        self.result_state.setText(_atlas_state_text(page, cards))
        if page.status not in {"ok", "empty"}:
            self._set_hero_tone("pending")
        elif any(card.evidence_label == "Klinisch validiert" for card in cards):
            self._set_hero_tone("validated")
        elif cards:
            self._set_hero_tone("research")
        else:
            self._set_hero_tone("neutral")
        if page.status in {"ok", "empty"}:
            self.context.setText(
                f"Run {page.run_id or 'nicht ausgewählt'} · Validation · "
                f"{'vollständiger Snapshot' if page.analysis_completeness == 'complete_snapshot' else 'Vorschau/unklarer Datenstand'} · "
                f"{page.total_count:,} auswahlfähige Markerzeilen; "
                "höchstens 200 davon für diese Übersicht geladen."
            )
        else:
            self.context.setText(
                f"Run {page.run_id or 'nicht ausgewählt'} · keine freigegebene Ergebnisansicht · "
                "Locked Test bleibt versiegelt."
            )
        self._clear_cards()
        self._has_answer_cards = bool(cards)
        self.empty_state.setVisible(not self._detailed and not cards)
        if page.status in {"ok", "empty"}:
            self.empty_title.setText("Kein neuer auswahlfähiger Markerbefund in dieser Auswahl")
            self.empty_message.setText(
                "Die Analyse kann abgeschlossen sein, ohne dass daraus eine neue Befundungsregel folgt. "
                "Klassische Vergleiche und Forschungsdetails bleiben separat prüfbar."
            )
        else:
            self.empty_title.setText("Ergebnisse folgen nach abgeschlossener Validation")
            self.empty_message.setText(
                "Bis dahin wird aus fehlenden Ergebnisdateien weder ein negativer Befund "
                "noch eine neue klinische Regel abgeleitet."
            )
        for card_index, card in enumerate(cards):
            panel = QFrame()
            panel.setObjectName("resultsAtlasCard")
            panel.setProperty(
                "tier", "validated" if card.evidence_label == "Klinisch validiert"
                else "candidate" if card.evidence_label == "Kriterienkandidat"
                else "research",
            )
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(16, 15, 16, 15)
            panel_layout.setSpacing(9)
            card_header = QHBoxLayout()
            ordinal = QLabel(f"FUND {card_index + 1:02d}")
            ordinal.setObjectName("resultsAtlasOrdinal")
            card_header.addWidget(ordinal)
            card_header.addStretch(1)
            evidence = QLabel(card.evidence_label)
            evidence.setObjectName("resultsAtlasEvidenceTag")
            evidence.setProperty("tier", panel.property("tier"))
            card_header.addWidget(evidence)
            panel_layout.addLayout(card_header)
            title = QLabel(card.title)
            title.setObjectName("resultsAtlasCardTitle")
            panel_layout.addWidget(title)
            takeaway = QLabel(card.takeaway)
            takeaway.setWordWrap(True)
            panel_layout.addWidget(takeaway)
            body = QLabel(card.body)
            body.setWordWrap(True)
            panel_layout.addWidget(body)
            detail = QLabel(card.detail)
            detail.setWordWrap(True)
            detail.setObjectName("mimicTextMuted")
            panel_layout.addWidget(detail)
            comparison = QLabel(card.comparison)
            comparison.setWordWrap(True)
            panel_layout.addWidget(comparison)
            limits = QLabel(card.open_questions)
            limits.setWordWrap(True)
            limits.setObjectName("mimicTextMuted")
            panel_layout.addWidget(limits)
            panel_layout.addStretch(1)
            button = QPushButton("Belege und Detailansichten öffnen")
            button.setObjectName("resultsAtlasCardAction")
            button.clicked.connect(lambda _checked=False, endpoint=card.endpoint_id: self.endpoint_requested.emit(endpoint))
            panel_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignLeft)
            self.cards_layout.addWidget(panel, 0, card_index)
        self._show_endpoint_evidence(page)
        self._update_breadcrumb()
        self._show_quality(payload)
        self.model_quality_view.set_snapshot(payload.model_quality)
        self.availability_view.set_snapshot(payload.availability)
        self._set_nav_status(self.model_quality_view, payload.model_quality.status)
        self._set_nav_status(
            self.availability_view,
            "ok" if payload.availability.reference_eligible is not None else "empty",
        )
        self.curve_view.set_snapshot(CurveSnapshot(
            "pending", "Kurven für den ausgewählten Marker werden geladen.",
            pd.DataFrame(), pd.DataFrame(), self._selected_endpoint_id,
            self._selected_marker_id,
        ))
        self._start_availability_load()
        self.paired_view.set_snapshot(payload.paired)
        self._set_nav_status(self.paired_view, payload.paired.status)
        self.nonfindings_summary.setText(paired_outcome_summary(payload.paired))
        self.comparison_summary.setText(_paired_comparison_text(page.rows))
        self.limits_summary.setText(_limits_text(page.rows))
        self.comparison_summary.setVisible(not self._detailed and not cards)
        self.limits_summary.setVisible(not self._detailed and not cards)
        scope = payload.scope_counts
        self.scope_summary.setText(
            f"Ausgewertet: {scope['evaluated_endpoints']:,} Endpunkte mit Ergebniszeilen; "
            f"für {scope['eligible_endpoints']:,} davon liegen "
            f"{scope['eligible_pairs']:,} auswahlfähige Marker–Endpunkt-Zeilen vor. "
            "Zeilen sind keine unabhängigen Patienten; Fallzahlen und Messbarkeit stehen im Endpunkt-Dossier."
            if scope else
            "Der Umfang der geprüften Endpunkte ist für diesen Ergebnisstand nicht sicher bestimmbar."
        )
        stages = payload.stage_counts
        if stages:
            self.evidence_summary.setText(
                "Gesamter Lauf: "
                f"{stages.get('clinically_validated', 0):,} klinisch validierte, "
                f"{stages.get('guideline_candidate', 0):,} Kriterienkandidaten und "
                f"{stages.get('replicated_association', 0):,} replizierte Marker–Endpunkt-Paare. "
                "Das sind keine unabhängigen klinischen Entdeckungen; Formeln und Endpunkte können verwandt sein."
            )
        else:
            self.evidence_summary.setText(
                "Evidenzzähler fehlen. Aus einer AUC allein entsteht keine klinische Freigabe."
            )
        if self._detailed:
            self._resize_detail_scroll()

    def _show_quality(self, payload: AtlasPayload) -> None:
        stages = payload.stage_counts
        counts = (
            f"Gesamter Lauf: {stages.get('exploratory', 0):,} explorativ · "
            f"{stages.get('replicated_association', 0):,} replizierte Assoziationen · "
            f"{stages.get('guideline_candidate', 0):,} Kriterienkandidaten · "
            f"{stages.get('clinically_validated', 0):,} klinisch validiert."
        ) if stages else "Für diesen Ergebnisstand liegen keine vollständigen Evidenzzähler vor."
        self.quality_note.setText(
            counts + " Gate-Zahlen zählen überlappende Marker–Endpunkt-Paare, "
            "keine Patienten und keinen sequenziellen Trichter."
        )
        self.quality_table.setRowCount(len(payload.gate_counts))
        for row, (gate, count) in enumerate(payload.gate_counts):
            self.quality_table.setItem(row, 0, QTableWidgetItem(
                _GATE_NAMES_DE.get(gate, gate.replace("_", " "))
            ))
            self.quality_table.setItem(row, 1, QTableWidgetItem(f"{count:,}"))
        patterns = payload.gate_patterns
        self._gate_patterns = patterns
        self.pattern_plot.clear()
        self.pattern_table.setRowCount(len(patterns))
        if patterns.empty:
            self.pattern_note.setText(
                "Keine vollständige, vergleichbare Gate-Kombinationszählung vorhanden."
            )
            return
        total = int(patterns.iloc[0]["total_pairs"])
        distinct = int(patterns.iloc[0]["distinct_patterns"])
        shown = int(patterns["n"].sum())
        self.pattern_note.setText(
            f"Die {len(patterns)} häufigsten von {distinct} Gate-Kombinationen "
            f"umfassen {shown:,} von {total:,} Marker–Endpunkt-Paaren. "
            "Dies sind exakte gemeinsame Fehlmuster, keine unabhängigen Patienten; "
            "‚kein Gate gemeldet‘ beweist für sich noch keine klinische Freigabe."
        )
        ticks = []
        for index, (_, row) in enumerate(patterns.iterrows()):
            y = float(len(patterns) - index)
            pattern = str(row["pattern"])
            names = " + ".join(
                _GATE_NAMES_DE.get(item, item.replace("_", " "))
                for item in pattern.split("|")
            ) if pattern != "(kein Gate gemeldet)" else pattern
            n = int(row["n"])
            for column, value in enumerate((names, f"{n:,}", f"{n / total:.1%}")):
                self.pattern_table.setItem(index, column, QTableWidgetItem(value))
            self.pattern_plot.addItem(pg.BarGraphItem(
                x0=[0], y0=[y - .36], width=[n], height=.72,
                brush=pg.mkBrush("#2b8cbe" if pattern != "(kein Gate gemeldet)" else "#8a8a8a"),
            ))
            ticks.append((y, f"Muster {index + 1}"))
        self.pattern_plot.getAxis("left").setTicks([ticks])
        self.pattern_plot.setYRange(.5, len(patterns) + .5, padding=0)

    def _show_research_summary(self, payload: AtlasPayload) -> None:
        scores = payload.scores.models
        score_count = int(pd.to_numeric(scores["auc"], errors="coerce").notna().sum()) if (
            payload.scores.status == "ok" and "auc" in scores
        ) else 0
        score_endpoints = int(scores.loc[
            pd.to_numeric(scores["auc"], errors="coerce").notna(), "endpoint_id"
        ].nunique()) if score_count else 0
        fits = payload.prognosis.fits
        prognosis_count = int(fits["status"].eq("estimable").sum()) if (
            payload.prognosis.status == "ok" and "status" in fits
        ) else 0
        responses = payload.response.rows
        response_count = int((responses["scope"].eq("medication_class")
                              & responses["status"].isin(("estimable", "estimable_point_only"))).sum()) if (
            payload.response.status == "ok" and {"scope", "status"}.issubset(responses.columns)
        ) else 0
        present = score_count + prognosis_count + response_count > 0
        self.research_host.setProperty("has_data", present)
        self.research_host.setVisible(present and not self._detailed)
        if not present:
            return
        parts = []
        if score_count:
            parts.append(
                f"{score_count} OOF-{'Modellgruppe' if score_count == 1 else 'Modellgruppen'} "
                f"für {score_endpoints} {'Endpunkt' if score_endpoints == 1 else 'Endpunkte'}"
            )
        if prognosis_count:
            parts.append(
                f"{prognosis_count} schätzbare "
                f"Prognose-{'Assoziation' if prognosis_count == 1 else 'Assoziationen'}"
            )
        if response_count:
            parts.append(
                f"{response_count} "
                f"{'Medikamentenklasse' if response_count == 1 else 'Medikamentenklassen'} "
                "mit beobachteter Response"
            )
        self.research_summary.setText(
            "Weitere abgeschlossene Forschungsanalysen: " + "; ".join(parts) + ". "
            "Das sind weder neue klinische Befundungsregeln noch Therapieempfehlungen. "
            "Die jeweilige Aussagegrenze steht in der Detailansicht."
        )
        self.score_summary_button.setVisible(score_count > 0)
        self.prognosis_summary_button.setVisible(prognosis_count > 0)
        self.response_summary_button.setVisible(response_count > 0)

    def _gate_pattern_metadata(self) -> dict[str, object]:
        page = self._last_page
        return {
            "run_id": page.run_id if page else None,
            "snapshot_hash": page.snapshot_hash if page else None,
            "split": "validation",
            "unit": "marker_endpoint_pair",
            "note": "Gemeinsam fehlende Evidenz-Gates; keine unabhängigen Patienten.",
        }

    def _export_gate_pattern_plot(self, *_args) -> None:
        if getattr(self, "_gate_patterns", pd.DataFrame()).empty:
            return
        choose_plot_export(self, self.pattern_plot, self._gate_pattern_metadata(),
                           proposed_name="gate_fehlmuster.png")

    def _export_gate_patterns(self, *_args) -> None:
        patterns = getattr(self, "_gate_patterns", pd.DataFrame())
        if not patterns.empty:
            choose_table_export(self, patterns, self._gate_pattern_metadata(),
                                proposed_name="gate_fehlmuster.csv")

    def _show_endpoint_evidence(self, page: ResultPage) -> None:
        self.endpoint_plot.clear()
        self.detail_table.setRowCount(0)
        frame = endpoint_evidence_frame(page, self._last_request[1] if self._last_request else "")
        self._detail_frame = frame
        if frame.empty:
            self.endpoint_display.setTabEnabled(0, False)
            self.endpoint_display.setCurrentIndex(1)
            self.marker_gate_table.setRowCount(0)
            self.marker_gate_note.setText("Kein auswahlfähiger Marker für eine Einzelprüfung vorhanden.")
            self.detail_heading.setText("Keine auswahlfähigen Marker für diese Frage")
            self.detail_note.setText("Technische Selbstvergleiche, klassische Regeln und fehlende Messungen werden nicht als neue Marker gezeichnet.")
            return
        first = frame.iloc[0]
        self.detail_heading.setText(_string(first.get("endpoint_name_de"), _string(first.get("endpoint_id"))))
        eligible = _number(first.get("n_reference_eligible"))
        observed = _number(first.get("n_marker_observed"))
        if eligible is not None and observed is not None and eligible > 0 and 0 <= observed <= eligible:
            self.availability_note.setText(
                f"Messverfügbarkeit des zuerst gezeigten Markers: {int(observed)} "
                f"von {int(eligible)} Datensätzen mit geeigneter Referenz. "
                "Das ist keine Schätzung der Bevölkerungshäufigkeit."
            )
            self.availability_bar.setValue(round(1000 * observed / eligible))
            self.availability_bar.setFormat(f"{observed / eligible:.1%} messbar")
            self.availability_bar.show()
        else:
            self.availability_note.setText(
                "Für diesen Marker fehlt ein konsistenter Verfügbarkeitsnenner; "
                "es wird kein künstlicher Trichter gezeichnet."
            )
        self.detail_note.setText(
            "Gezeigt werden höchstens 25 Marker aus den ersten 200 auswahlfähigen Ergebniszeilen. "
            "Das Diagramm zeigt ausschließlich AUCs mit berechnetem Intervall; "
            "weitere Punktschätzer bleiben in der Zahlentabelle. AUC ist keine Erkrankungswahrscheinlichkeit."
            + (" " + _timing_label(first) if _timing_label(first) else "")
        )
        names: list[tuple[float, str]] = []
        self.detail_table.setRowCount(len(frame))
        for index, (_, row) in enumerate(frame.iterrows()):
            name = _string(row.get("feature_name_de"), _string(row.get("feature_id")))
            auc = _number(row.get("roc_auc"))
            low = _number(row.get("roc_auc_ci_low"))
            high = _number(row.get("roc_auc_ci_high"))
            if (auc is not None and low is not None and high is not None
                    and 0 <= low <= auc <= high <= 1):
                y = float(len(names) + 1)
                names.append((y, name[:36]))
                self.endpoint_plot.plot([low, high], [y, y], pen=pg.mkPen("#29a3a3", width=2))
                self.endpoint_plot.plot([auc], [y], pen=None, symbol="o", symbolSize=8,
                                        symbolBrush="#29a3a3")
            delta = _number(row.get("delta_auc"))
            delta_low = _number(row.get("delta_auc_ci_low"))
            delta_high = _number(row.get("delta_auc_ci_high"))
            if delta is not None and delta_low is not None and delta_high is not None:
                comparison = (
                    f"Explorer-ΔAUC {delta:+.2f} ({delta_low:+.2f}–{delta_high:+.2f}); "
                    "formale Paarinferenz separat prüfen"
                )
            else:
                comparison = "Gepaarter Zusatznutzen nicht belegt"
            values = (name, _auc_label(row), _case_label(row),
                      _reference_label(row.get("reference_kind")), comparison)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(
                        Qt.ItemDataRole.UserRole, _string(row.get("feature_id"), "")
                    )
                self.detail_table.setItem(index, column, item)
        self.endpoint_plot.getAxis("left").setTicks([names])
        self.endpoint_display.setTabEnabled(0, bool(names))
        if names:
            self.endpoint_plot.setYRange(.5, len(names) + .5, padding=0)
        else:
            self.endpoint_display.setCurrentIndex(1)
        self.endpoint_plot.setXRange(0, 1, padding=0)
        self._show_marker_gates(0)
        self._selected_marker_id = _string(frame.iloc[0].get("feature_id"), "")
        self._selected_endpoint_id = _string(frame.iloc[0].get("endpoint_id"), "")
        self._update_breadcrumb()

    def _select_marker_row(self, index: int, _column: int) -> None:
        self._show_marker_gates(index)
        if index < 0 or index >= len(self._detail_frame):
            return
        row = self._detail_frame.iloc[index]
        marker_id = _string(row.get("feature_id"), "")
        endpoint_id = _string(row.get("endpoint_id"), "")
        if not marker_id or (marker_id == self._selected_marker_id
                             and endpoint_id == self._selected_endpoint_id):
            return
        self._selected_marker_id = marker_id
        self._selected_endpoint_id = endpoint_id
        self._update_breadcrumb()
        self._availability_generation += 1
        self.availability_view.set_snapshot(AvailabilitySnapshot(
            endpoint_id, marker_id, None, None, None, None, pd.DataFrame(),
            f"Messverfügbarkeit und Untergruppen für {marker_id} werden geladen …",
        ))
        self.curve_view.set_snapshot(CurveSnapshot(
            "pending", "Kurven für den ausgewählten Marker werden geladen.",
            pd.DataFrame(), pd.DataFrame(), endpoint_id, marker_id,
        ))
        self._start_availability_load()

    def _start_availability_load(self) -> None:
        if (self._shutting_down or self._availability_worker is not None
                or self._last_page is None or not self._selected_marker_id):
            return
        worker = _AvailabilityLoad(
            self._availability_generation, self._last_page,
            self._selected_endpoint_id, self._selected_marker_id,
        )
        self._availability_worker = worker
        worker.ready.connect(self._availability_loaded)
        worker.failed.connect(self._availability_failed)
        worker.finished.connect(lambda w=worker: self._retire_availability(w))
        worker.start()

    def _availability_loaded(
        self, generation: int, marker_id: str,
        payload: tuple[AvailabilitySnapshot, CurveSnapshot],
    ) -> None:
        if (not self._shutting_down and generation == self._availability_generation
                and marker_id == self._selected_marker_id):
            self.availability_view.set_snapshot(payload[0])
            self.curve_view.set_snapshot(payload[1])

    def _availability_failed(self, generation: int, marker_id: str, message: str) -> None:
        if (not self._shutting_down and generation == self._availability_generation
                and marker_id == self._selected_marker_id):
            self.availability_view.set_snapshot(AvailabilitySnapshot(
                self._selected_endpoint_id, marker_id, None, None, None, None,
                pd.DataFrame(), message,
            ))
            self.curve_view.set_snapshot(CurveSnapshot(
                "failed", message, pd.DataFrame(), pd.DataFrame(),
                self._selected_endpoint_id, marker_id,
            ))

    def _retire_availability(self, worker: _AvailabilityLoad) -> None:
        if self._availability_worker is worker:
            self._availability_worker = None
        worker.deleteLater()
        if self._shutting_down:
            self._maybe_idle()
        elif worker.generation != self._availability_generation:
            self._start_availability_load()

    def _show_marker_gates(self, index: int) -> None:
        if index < 0 or index >= len(self._detail_frame):
            return
        row = self._detail_frame.iloc[index]
        name = _string(row.get("feature_name_de"), _string(row.get("feature_id")))
        stage = _stage_label(_stage(row))
        reference = _reference_label(row.get("reference_kind"))
        ceiling = _ceiling_label(row.get("claim_ceiling"))
        self.marker_gate_note.setText(
            f"{name}: {stage}. Referenz: {reference}. Aussagegrenze: {ceiling}. "
            "Offene Gates sind keine Nullbefunde; nicht gemeldete Gates werden hier nicht als bestanden gewertet."
        )
        gates = marker_gate_rows(row)
        self.marker_gate_table.setRowCount(len(gates))
        for i, (label, status) in enumerate(gates):
            self.marker_gate_table.setItem(i, 0, QTableWidgetItem(label))
            self.marker_gate_table.setItem(i, 1, QTableWidgetItem(status))

    def _open_marker_row(self, row: int, _column: int) -> None:
        item = self.detail_table.item(row, 0)
        feature_id = str(item.data(Qt.ItemDataRole.UserRole) or "") if item else ""
        run_id = self._last_request[0] if self._last_request else ""
        if feature_id and run_id:
            self.marker_view.select_marker(self.repository, run_id, feature_id)
            self.detail_tabs.setCurrentIndex(self.marker_tab_index)

    def _export_metadata(self) -> dict[str, object]:
        page = self._last_page
        return {
            "run_id": page.run_id if page else None,
            "snapshot_hash": page.snapshot_hash if page else None,
            "analysis_completeness": page.analysis_completeness if page else None,
            "split": "validation",
            "endpoint_id": self._last_request[1] if self._last_request else None,
            "displayed_rows": len(self._detail_frame),
            "note": "Forschungsansicht; AUC ist weder Erkrankungswahrscheinlichkeit noch klinischer Nutzen.",
        }

    def _export_endpoint_plot(self, *_args) -> None:
        if self._detail_frame.empty or not self.endpoint_display.isTabEnabled(0):
            return
        choose_plot_export(
            self, self.endpoint_plot, self._export_metadata(),
            proposed_name="endpunkt_auc_validation.png",
        )

    def _export_endpoint_table(self, *_args) -> None:
        if self._detail_frame.empty:
            return
        choose_table_export(
            self, self._detail_frame, self._export_metadata(),
            proposed_name="endpunkt_marker_validation.csv",
        )

    def _failed(self, generation: int, message: str) -> None:
        if self._shutting_down:
            return
        if generation == self._generation:
            self._loaded_generation = generation
            self._clear_cards()
            self.headline.setText(message)
            self.evidence_summary.setText("Evidenzzähler derzeit nicht verfügbar.")
            self.scope_summary.setText("Untersuchungsumfang wegen fehlgeschlagener Abfrage nicht verfügbar.")
            self.comparison_summary.setText("Klassischer ECG-Vergleich: nicht verfügbar.")
            self.limits_summary.setText("Aussagegrenzen: Ergebnisabfrage fehlgeschlagen.")

    def _clear_cards(self) -> None:
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
