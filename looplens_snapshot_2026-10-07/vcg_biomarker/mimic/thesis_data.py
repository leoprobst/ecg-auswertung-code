"""Data assembly for the fixed thesis versions.

Every query reads the development splits only (``train``, ``validation``);
the locked test is never touched. Measurements near an ECG are the nearest
plausible value inside the window of the version document, ties broken by
the earlier time and then the smaller value, so the choice is deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

DEVELOPMENT_SPLITS = ("train", "validation")
ADMINISTERED_EVENTS = ("Administered", "Delayed Administered", "Administered in Other Location")


def _sql_path(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


class ThesisData:
    """Lazy, cached access to the tables the thesis versions read."""

    def __init__(self, config, feature_glob: str, connection) -> None:
        self.config = config
        self.feature_glob = _sql_path(feature_glob)
        self.connection = connection
        self._cache: dict = {}
        self.master = _sql_path(config.master_root / "ecg_master_index.parquet")

    # ------------------------------------------------------------ helpers

    def _columns(self, relation: str) -> set[str]:
        key = ("columns", relation)
        if key not in self._cache:
            try:
                rows = self.connection.execute(f"DESCRIBE SELECT * FROM {relation} LIMIT 0").fetchall()
                self._cache[key] = {str(row[0]) for row in rows}
            except Exception:  # noqa: BLE001 - a missing source is reported, not raised
                self._cache[key] = set()
        return self._cache[key]

    def _feature_relation(self) -> str:
        from .storage import feature_parquet_clause

        if "feature_relation" not in self._cache:
            self._cache["feature_relation"] = feature_parquet_clause(
                self.connection, self.feature_glob
            )
        return self._cache["feature_relation"]

    def event_relation(self, table: str) -> Optional[str]:
        from .event_ingest import clinical_event_relation

        key = ("event", table)
        if key not in self._cache:
            try:
                self._cache[key] = clinical_event_relation(self.config, table)
            except Exception:  # noqa: BLE001
                self._cache[key] = None
        return self._cache[key]

    # ------------------------------------------------------------ base frame

    def base(self, marker_columns: Sequence[str]) -> pd.DataFrame:
        """Development ECGs with the requested markers, master fields and report states."""

        relation = self._feature_relation()
        available = self._columns(relation)
        markers = [name for name in dict.fromkeys(marker_columns) if name in available]
        key = ("base", tuple(markers))
        if key in self._cache:
            return self._cache[key]
        marker_sql = ", ".join(f"f.{_identifier(name)}" for name in markers)
        reports = _sql_path(self.config.evidence_root / "machine_report_labels.parquet")
        report_columns = self._columns(f"read_parquet('{reports}')")
        report_fields = [
            name for name in ("report_af_state", "report_lbbb_state", "report_pacemaker_state")
            if name in report_columns
        ]
        report_select = ", ".join(f"r.{name}" for name in report_fields)
        master_columns = self._columns(f"read_parquet('{self.master}')")
        master_fields = [
            name for name in ("hadm_id", "icustay_id", "age", "sex", "machine_report", "ecg_time")
            if name in master_columns
        ]
        master_select = ", ".join(f"m.{name}" for name in master_fields)
        heart_rate = "f.context_heart_rate_bpm" if "context_heart_rate_bpm" in available else "NULL::DOUBLE"
        query = f"""
            SELECT f.subject_id, f.study_id, f.split,
                   {heart_rate} AS heart_rate_bpm
                   {', ' + master_select if master_select else ''}
                   {', ' + report_select if report_select else ''}
                   {', ' + marker_sql if marker_sql else ''}
            FROM {relation} f
            JOIN read_parquet('{self.master}') m USING (subject_id, study_id)
            {f"LEFT JOIN read_parquet('{reports}') r USING (subject_id, study_id)" if report_select else ''}
            WHERE f.extraction_status = 'ok'
              AND f.split IN {DEVELOPMENT_SPLITS}
        """
        frame = self.connection.execute(query).df()
        frame.attrs["missing_markers"] = sorted(set(marker_columns) - set(markers))
        self._cache[key] = frame
        return frame

    # ------------------------------------------------------------ context

    def context(self, columns: Sequence[str]) -> pd.DataFrame:
        path = self.config.evidence_root / "clinical_context_features.parquet"
        relation = f"read_parquet('{_sql_path(path)}')"
        available = self._columns(relation)
        wanted = [name for name in dict.fromkeys(columns) if name in available]
        select = ", ".join(_identifier(name) for name in wanted)
        frame = self.connection.execute(
            f"SELECT subject_id, study_id{', ' + select if select else ''} FROM {relation} "
            f"WHERE split IN {DEVELOPMENT_SPLITS}"
        ).df()
        frame.attrs["missing_columns"] = sorted(set(columns) - set(wanted))
        return frame

    def labels(self, label_ids: Sequence[str]) -> pd.DataFrame:
        path = self.config.labels_root / "clinical_labels_wide.parquet"
        relation = f"read_parquet('{_sql_path(path)}')"
        available = self._columns(relation)
        wanted = [f"{label}_state" for label in label_ids if f"{label}_state" in available]
        select = ", ".join(_identifier(name) for name in wanted)
        return self.connection.execute(
            f"SELECT subject_id, study_id{', ' + select if select else ''} FROM {relation} "
            f"WHERE split IN {DEVELOPMENT_SPLITS}"
        ).df()

    # ------------------------------------------------------------ labs

    def nearest_lab(
        self,
        lab_group: str,
        window_hours: float,
        *,
        itemids: Optional[Iterable[int]] = None,
        statuses: Sequence[str] = ("usable",),
    ) -> pd.DataFrame:
        """Nearest lab value per development ECG inside +/- ``window_hours``."""

        path = _sql_path(self.config.evidence_root / "labs.parquet")
        relation = f"read_parquet('{path}')"
        columns = self._columns(relation)
        if not columns:
            return pd.DataFrame()
        hemolysis = "l.hemolysis_state" if "hemolysis_state" in columns else "NULL::VARCHAR"
        specimen = "l.specimen_id" if "specimen_id" in columns else "NULL::BIGINT"
        item_filter = (
            f"AND l.itemid IN ({', '.join(str(int(i)) for i in itemids)})" if itemids else ""
        )
        status_list = ", ".join(f"'{status}'" for status in statuses)
        minutes = int(round(float(window_hours) * 60))
        query = f"""
            WITH candidates AS (
                SELECT m.subject_id, m.study_id, l.labevent_id, l.itemid,
                       l.value_numeric, l.value_status, {hemolysis} AS hemolysis_state,
                       {specimen} AS specimen_id, l.charttime,
                       DATE_DIFF('second', m.ecg_time, l.charttime) / 3600.0 AS delta_hours,
                       ROW_NUMBER() OVER (
                           PARTITION BY m.subject_id, m.study_id
                           ORDER BY ABS(DATE_DIFF('second', m.ecg_time, l.charttime)),
                                    l.charttime, l.value_numeric, l.labevent_id
                       ) AS rank
                FROM read_parquet('{self.master}') m
                JOIN {relation} l
                  ON l.subject_id = m.subject_id
                 AND l.lab_group = '{lab_group}'
                 AND l.charttime BETWEEN m.ecg_time - INTERVAL {minutes} MINUTE
                                     AND m.ecg_time + INTERVAL {minutes} MINUTE
                 AND l.value_status IN ({status_list})
                 AND l.value_numeric IS NOT NULL
                 {item_filter}
                WHERE m.split IN {DEVELOPMENT_SPLITS}
            )
            SELECT * EXCLUDE (rank) FROM candidates WHERE rank = 1
        """
        return self.connection.execute(query).df()

    # ------------------------------------------------------------ chart values

    def nearest_chart(
        self,
        itemids: Sequence[int],
        window_hours: float,
        low: float,
        high: float,
        *,
        fahrenheit_itemids: Sequence[int] = (),
    ) -> pd.DataFrame:
        """Nearest plausible chartevents value per development ECG."""

        relation = self.event_relation("chartevents")
        if relation is None:
            return pd.DataFrame()
        all_ids = sorted({int(i) for i in itemids} | {int(i) for i in fahrenheit_itemids})
        fahrenheit = ", ".join(str(int(i)) for i in fahrenheit_itemids) or "-1"
        minutes = int(round(float(window_hours) * 60))
        query = f"""
            WITH charted AS (
                SELECT subject_id, charttime,
                       CASE WHEN itemid IN ({fahrenheit}) THEN (valuenum - 32.0) * 5.0 / 9.0
                            ELSE valuenum END AS value
                FROM {relation}
                WHERE itemid IN ({', '.join(str(i) for i in all_ids)})
                  AND valuenum IS NOT NULL
            ),
            candidates AS (
                SELECT m.subject_id, m.study_id, c.value,
                       DATE_DIFF('second', m.ecg_time, c.charttime) / 3600.0 AS delta_hours,
                       ROW_NUMBER() OVER (
                           PARTITION BY m.subject_id, m.study_id
                           ORDER BY ABS(DATE_DIFF('second', m.ecg_time, c.charttime)),
                                    c.charttime, c.value
                       ) AS rank
                FROM read_parquet('{self.master}') m
                JOIN charted c
                  ON c.subject_id = m.subject_id
                 AND c.charttime BETWEEN m.ecg_time - INTERVAL {minutes} MINUTE
                                     AND m.ecg_time + INTERVAL {minutes} MINUTE
                 AND c.value BETWEEN {float(low)} AND {float(high)}
                WHERE m.split IN {DEVELOPMENT_SPLITS}
            )
            SELECT subject_id, study_id, value, delta_hours FROM candidates WHERE rank = 1
        """
        return self.connection.execute(query).df()

    def infusion_running(self, groups: dict[str, Sequence[int]]) -> pd.DataFrame:
        """Per development ECG with an ICU stay: which infusions ran at ECG time."""

        relation = self.event_relation("inputevents")
        if relation is None:
            return pd.DataFrame()
        flags = ",\n".join(
            f"BOOL_OR(i.itemid IN ({', '.join(str(int(x)) for x in ids)})) AS {name}"
            for name, ids in groups.items()
        )
        all_ids = sorted({int(x) for ids in groups.values() for x in ids})
        query = f"""
            SELECT m.subject_id, m.study_id, {flags}
            FROM read_parquet('{self.master}') m
            LEFT JOIN {relation} i
              ON i.subject_id = m.subject_id
             AND i.itemid IN ({', '.join(str(x) for x in all_ids)})
             AND i.starttime <= m.ecg_time
             AND COALESCE(i.endtime, i.starttime) >= m.ecg_time
             AND COALESCE(i.rate, i.amount, 0) > 0
            WHERE m.split IN {DEVELOPMENT_SPLITS} AND m.icustay_id IS NOT NULL
            GROUP BY m.subject_id, m.study_id
        """
        frame = self.connection.execute(query).df()
        for name in groups:
            frame[name] = frame[name].fillna(False).astype(bool)
        return frame

    # ------------------------------------------------------------ medications

    def administrations(self, drug_names: Sequence[str]) -> pd.DataFrame:
        """Administered eMAR events of the named drugs, all subjects in the
        development splits, with the dose where eMAR detail records one."""

        emar = self.event_relation("emar")
        if emar is None:
            return pd.DataFrame()
        detail = self.event_relation("emar_detail")
        case = " ".join(
            f"WHEN LOWER(e.medication) LIKE '%{name.lower()}%' THEN '{name.lower()}'"
            for name in drug_names
        )
        dose = (
            f"""LEFT JOIN (
                    SELECT subject_id, emar_id, MAX(TRY_CAST(dose_given AS DOUBLE)) AS dose_given,
                           MAX(dose_given_unit) AS dose_given_unit
                    FROM {detail} GROUP BY subject_id, emar_id
                ) d USING (subject_id, emar_id)"""
            if detail is not None
            else ""
        )
        dose_select = "d.dose_given, d.dose_given_unit" if detail is not None else "NULL::DOUBLE AS dose_given, NULL::VARCHAR AS dose_given_unit"
        events = ", ".join(f"'{event}'" for event in ADMINISTERED_EVENTS)
        query = f"""
            SELECT e.subject_id, e.hadm_id, e.charttime, CASE {case} END AS drug, {dose_select}
            FROM {emar} e
            {dose}
            WHERE e.event_txt IN ({events})
              AND e.subject_id IN (
                  SELECT subject_id FROM read_parquet('{self.master}')
                  WHERE split IN {DEVELOPMENT_SPLITS}
              )
              AND (CASE {case} END) IS NOT NULL
        """
        return self.connection.execute(query).df()

    # ------------------------------------------------------------ codes

    def hadm_with_codes(
        self, icd9: Sequence[str], icd10: Sequence[str], *, table: str = "diagnoses_icd"
    ) -> set:
        path = _sql_path(self.config.standardized_root / f"{table}.parquet")
        conditions = []
        if icd9:
            conditions.append(
                "(icd_version = 9 AND (" + " OR ".join(
                    f"REPLACE(icd_code, '.', '') LIKE '{code}%'" for code in icd9) + "))"
            )
        if icd10:
            conditions.append(
                "(icd_version = 10 AND (" + " OR ".join(
                    f"REPLACE(icd_code, '.', '') LIKE '{code}%'" for code in icd10) + "))"
            )
        if not conditions:
            return set()
        rows = self.connection.execute(
            f"SELECT DISTINCT hadm_id FROM read_parquet('{path}') WHERE {' OR '.join(conditions)}"
        ).fetchall()
        return {row[0] for row in rows if row[0] is not None}

    def procedure_events(self, icd9: Sequence[str], icd10: Sequence[str]) -> pd.DataFrame:
        path = _sql_path(self.config.standardized_root / "procedures_icd.parquet")
        conditions = []
        if icd9:
            conditions.append("(icd_version = 9 AND (" + " OR ".join(
                f"REPLACE(icd_code, '.', '') LIKE '{code}%'" for code in icd9) + "))")
        if icd10:
            conditions.append("(icd_version = 10 AND (" + " OR ".join(
                f"REPLACE(icd_code, '.', '') LIKE '{code}%'" for code in icd10) + "))")
        return self.connection.execute(
            f"SELECT DISTINCT subject_id, hadm_id, chartdate FROM read_parquet('{path}') "
            f"WHERE {' OR '.join(conditions)}"
        ).df()

    def admissions(self) -> pd.DataFrame:
        path = _sql_path(self.config.standardized_root / "admissions.parquet")
        return self.connection.execute(
            f"SELECT subject_id, hadm_id, admittime, dischtime FROM read_parquet('{path}')"
        ).df()

    def reference_ranges(self, run_root: Path) -> pd.DataFrame:
        """Stratified reference percentiles written by this run's reference-range stage."""

        path = Path(run_root) / "reference_ranges" / "marker_reference_ranges.parquet"
        return pd.read_parquet(path) if path.is_file() else pd.DataFrame()
