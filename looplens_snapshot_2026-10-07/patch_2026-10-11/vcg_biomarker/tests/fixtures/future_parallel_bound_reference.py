def _future_label_summary(
    index: pd.DataFrame,
    events: pd.DataFrame,
    definitions: pd.DataFrame,
    *,
    horizons_days: tuple[int, ...],
    min_inference: int,
    max_phenotypes: Optional[int],
) -> tuple[pd.DataFrame, tuple[tuple[str, int], ...]]:
    definition_lookup = definitions.set_index("phenotype_id")
    records = []
    for phenotype_id, event_frame in events.groupby("phenotype_id", sort=True):
        phenotype_id = str(phenotype_id)
        if phenotype_id not in definition_lookup.index:
            continue
        definition = definition_lookup.loc[phenotype_id]
        event_series = event_frame.set_index("subject_id")["first_event_time"]
        for horizon in horizons_days:
            masks = _future_state_masks(
                index,
                event_series,
                definition,
                horizon_days=horizon,
            )
            record = {
                "phenotype_id": phenotype_id,
                "phenotype_name": str(definition["display_name"]),
                "body_system": str(definition["body_system"]),
                "horizon_days": int(horizon),
                "target_kind": "post_discharge_prognostic_association",
                "index_policy": (
                    "first feature-planned ECG within seven days before a "
                    "completed index-admission discharge landmark, selected "
                    "without outcome or extraction-success information"
                ),
                "positive_rule": (
                    "first phenotype-coded later admission is documented at "
                    f"discharge after the discharge landmark and within {horizon} days"
                ),
                "negative_rule": (
                    "not a binary control definition; all event-free person-"
                    "time contributes until censoring, competing death, or "
                    f"the {horizon}-day horizon"
                ),
                "censoring_rule": (
                    "right censor at last observation or horizon; death before "
                    "phenotype is an explicit competing event and cause-"
                    "specific censoring time"
                ),
                "same_encounter_diagnosis_excluded": True,
                "time_origin": "index_admission_discharge_landmark",
                "ecg_to_landmark_max_days": (
                    MAX_ECG_TO_DISCHARGE_LANDMARK_DAYS
                ),
                "same_day_death_event_primary_policy": "exclude_ambiguous_tie",
                "locked_test_scanned": False,
            }
            for split in ("train", "validation"):
                split_mask = index["split"].eq(split).to_numpy()
                record[f"{split}_positive_count"] = int(
                    np.sum(masks["positive"] & split_mask)
                )
                record[f"{split}_negative_count"] = int(
                    np.sum(masks["negative"] & split_mask)
                )
                record[f"{split}_unavailable_count"] = int(
                    np.sum(masks["unavailable"] & split_mask)
                )
                record[f"{split}_excluded_history_count"] = int(
                    np.sum(masks["excluded_history"] & split_mask)
                )
                record[f"{split}_excluded_sex_count"] = int(
                    np.sum(masks["excluded_sex"] & split_mask)
                )
                record[f"{split}_excluded_same_day_tie_count"] = int(
                    np.sum(masks["excluded_time_tie"] & split_mask)
                )
                record[f"{split}_at_risk_count"] = int(
                    np.sum(masks["at_risk"] & split_mask)
                )
                record[f"{split}_event_count"] = int(
                    np.sum(masks["event_observed"] & split_mask)
                )
                record[f"{split}_censored_count"] = int(
                    np.sum(masks["censored"] & split_mask)
                )
                record[f"{split}_competing_death_count"] = int(
                    np.sum(masks["competing_death"] & split_mask)
                )
            record["eligible_for_screen"] = (
                record["train_event_count"] >= min_inference
                and (
                    record["train_at_risk_count"]
                    - record["train_event_count"]
                ) >= min_inference
            )
            record["selection_split"] = "train"
            record["validation_outcomes_used_for_selection"] = False
            record["validation_assessable_after_freeze"] = (
                record["validation_event_count"] >= min_inference
                and (
                    record["validation_at_risk_count"]
                    - record["validation_event_count"]
                ) >= min_inference
            )
            records.append(record)
    summary = pd.DataFrame(records)
    if summary.empty:
        return summary, ()
    eligible = summary.loc[summary["eligible_for_screen"]].copy()
    if max_phenotypes is not None and not eligible.empty:
        ranked_ids = (
            eligible.assign(
                _train_cases=eligible["train_event_count"]
            )
            .sort_values(
                ["_train_cases", "phenotype_id"],
                ascending=[False, True],
            )["phenotype_id"]
            .drop_duplicates()
            .head(int(max_phenotypes))
        )
        eligible = eligible.loc[
            eligible["phenotype_id"].isin(set(ranked_ids))
        ]
    selected = tuple(
        (str(row.phenotype_id), int(row.horizon_days))
        for row in eligible.sort_values(
            ["phenotype_id", "horizon_days"]
        ).itertuples(index=False)
    )
    return summary, selected

def _future_state_masks(
    frame: pd.DataFrame,
    event_series: pd.Series,
    definition: pd.Series,
    *,
    horizon_days: int,
    same_day_tie_policy: str = "exclude",
) -> dict[str, np.ndarray]:
    if same_day_tie_policy not in {"exclude", "event_first", "death_first"}:
        raise ValueError(f"Invalid same-day tie policy: {same_day_tie_policy}")
    subjects = frame["subject_id"].astype(int)
    index_end = pd.to_datetime(frame["index_end_time"], errors="coerce")
    last_observed = pd.to_datetime(
        frame["last_observed_time"],
        errors="coerce",
    )
    event_time = pd.to_datetime(
        subjects.map(event_series),
        errors="coerce",
    )
    death_source = (
        frame["date_of_death"]
        if "date_of_death" in frame
        else pd.Series(pd.NaT, index=frame.index)
    )
    death_time = pd.to_datetime(death_source, errors="coerce")
    event_delta = (
        event_time.reset_index(drop=True)
        - index_end.reset_index(drop=True)
    ).dt.total_seconds().to_numpy(dtype=float) / 86_400.0
    followup_delta = (
        last_observed.reset_index(drop=True)
        - index_end.reset_index(drop=True)
    ).dt.total_seconds().to_numpy(dtype=float) / 86_400.0
    death_delta = (
        death_time.reset_index(drop=True)
        - index_end.reset_index(drop=True)
    ).dt.total_seconds().to_numpy(dtype=float) / 86_400.0
    sex = frame.get("sex", pd.Series("", index=frame.index)).astype(str).str.upper()
    male_only = _source_bool(definition.get("male_only", False))
    female_only = _source_bool(definition.get("female_only", False))
    excluded_sex = np.zeros(len(frame), dtype=bool)
    if male_only:
        excluded_sex = ~sex.isin(("M", "MALE")).to_numpy()
    elif female_only:
        excluded_sex = ~sex.isin(("F", "FEMALE")).to_numpy()
    has_event = np.isfinite(event_delta)
    excluded_history = has_event & (event_delta <= 0)
    event_in_horizon = (
        ~excluded_sex
        & ~excluded_history
        & has_event
        & (event_delta > 0)
        & (event_delta <= float(horizon_days))
    )
    death_in_horizon = (
        ~excluded_sex
        & ~excluded_history
        & np.isfinite(death_delta)
        & (death_delta > 0)
        & (death_delta <= float(horizon_days))
    )
    event_calendar_day = event_time.reset_index(drop=True).dt.normalize()
    death_calendar_day = death_time.reset_index(drop=True).dt.normalize()
    ambiguous_same_day_death_event = (
        event_in_horizon
        & death_in_horizon
        & event_calendar_day.eq(death_calendar_day).to_numpy(dtype=bool)
    )
    excluded_time_tie = (
        ambiguous_same_day_death_event
        if same_day_tie_policy == "exclude"
        else np.zeros(len(frame), dtype=bool)
    )
    # For cause-specific hazards, a death before the phenotype event is a
    # competing event and censors observation at death. It is counted and
    # reported explicitly; it is not silently treated as an event-free
    # survivor.
    competing_death = death_in_horizon & ~ambiguous_same_day_death_event & (
        ~event_in_horizon | (death_delta < event_delta)
    )
    if same_day_tie_policy == "death_first":
        competing_death |= ambiguous_same_day_death_event
    positive = (
        event_in_horizon
        & ~competing_death
        & ~excluded_time_tie
    )
    survived_horizon = (
        ~np.isfinite(death_delta)
        | (death_delta >= float(horizon_days))
    )
    complete_followup = (
        np.isfinite(followup_delta)
        & (followup_delta >= float(horizon_days))
        & survived_horizon
    )
    no_event_in_horizon = ~has_event | (event_delta > float(horizon_days))
    negative = (
        ~excluded_sex
        & ~excluded_history
        & ~excluded_time_tie
        & ~positive
        & no_event_in_horizon
        & complete_followup
    )
    unavailable = (
        ~excluded_sex
        & ~excluded_history
        & ~excluded_time_tie
        & ~positive
        & ~negative
    )
    administrative_followup = np.minimum(
        np.maximum(followup_delta, 0.0),
        float(horizon_days),
    )
    # Censored rows run to the administrative horizon or to the last observed
    # contact, whichever comes first. A recorded event or death instead sets
    # the exact exit time: both are bounded by the horizon upstream, and the
    # event itself proves observation until it happened even when a
    # source-specific observation-end field is incomplete.
    duration_days = np.where(
        positive,
        event_delta,
        np.where(competing_death, death_delta, administrative_followup),
    )
    at_risk = (
        ~excluded_sex
        & ~excluded_history
        & ~excluded_time_tie
        & np.isfinite(duration_days)
        & (duration_days > 0)
    )
    censored = at_risk & ~positive & ~competing_death
    return {
        "positive": positive,
        "negative": negative,
        "unavailable": unavailable,
        "excluded_history": excluded_history & ~excluded_sex,
        "excluded_sex": excluded_sex,
        "excluded_time_tie": excluded_time_tie,
        "ambiguous_same_day_death_event": ambiguous_same_day_death_event,
        "same_day_tie_policy": same_day_tie_policy,
        "at_risk": at_risk,
        "duration_days": duration_days,
        "event_observed": positive,
        "competing_death": competing_death,
        "censored": censored,
    }

def _screen_future_targets(
    features: pd.DataFrame,
    *,
    selected_targets: tuple[tuple[str, int], ...],
    event_lookup: Mapping[str, pd.Series],
    definition_lookup: pd.DataFrame,
    feature_ids: tuple[str, ...],
    panel_map: Mapping[str, str],
    min_inference: int,
    min_formal: int,
    progress: Optional[Callable[[str], None]],
) -> pd.DataFrame:
    records = []
    numeric = {
        feature_id: pd.to_numeric(
            features[feature_id],
            errors="coerce",
        ).to_numpy(dtype=float)
        for feature_id in feature_ids
    }
    for target_index, (phenotype_id, horizon) in enumerate(
        selected_targets,
        start=1,
    ):
        definition = definition_lookup.loc[phenotype_id]
        masks = _future_state_masks(
            features,
            event_lookup.get(phenotype_id, pd.Series(dtype="datetime64[ns]")),
            definition,
            horizon_days=horizon,
        )
        case_rows = np.flatnonzero(masks["positive"])
        control_rows = np.flatnonzero(masks["negative"])
        at_risk_rows = np.flatnonzero(masks["at_risk"])
        for feature_id in feature_ids:
            values = numeric[feature_id]
            marker_assessment = _survival_marker_assessability(values, masks)
            case_values = values[case_rows]
            case_values = case_values[np.isfinite(case_values)]
            control_values = values[control_rows]
            control_values = control_values[np.isfinite(control_values)]
            raw_auc, _ = _auc_and_p(
                case_values,
                control_values,
            )
            valid_at_risk = at_risk_rows[
                np.isfinite(values[at_risk_rows])
            ]
            adjustment = _future_adjustment_covariates(
                features,
                valid_at_risk,
            )
            survival = _fit_cause_specific_cox(
                values[valid_at_risk],
                np.asarray(masks["duration_days"])[valid_at_risk],
                np.asarray(masks["event_observed"])[valid_at_risk],
                adjustment,
            )
            direction = int(survival["direction"])
            all_adjustment = _future_adjustment_covariates(
                features,
                at_risk_rows,
            )
            missingness_sensitivity = _adverse_missing_marker_sensitivity(
                values[at_risk_rows],
                np.asarray(masks["duration_days"])[at_risk_rows],
                np.asarray(masks["event_observed"])[at_risk_rows],
                all_adjustment,
                direction=direction,
            )
            tie_sensitivity = _same_day_tie_sensitivity(
                features,
                values,
                event_lookup.get(
                    phenotype_id, pd.Series(dtype="datetime64[ns]")
                ),
                definition,
                horizon_days=horizon,
                primary_direction=direction,
            )
            n_event = int(survival["n_event"])
            n_at_risk = int(survival["n_at_risk"])
            # Absolute risk beside the hazard ratio: a clinician needs the
            # risk difference, and the decision-curve stage of the evidence
            # ladder needs a level, not a ratio.
            absolute = _absolute_risk_block(
                duration_days=np.asarray(masks["duration_days"])[valid_at_risk],
                event_observed=np.asarray(masks["event_observed"])[
                    valid_at_risk
                ],
                competing_death=np.asarray(masks["competing_death"])[
                    valid_at_risk
                ],
                marker_values=values[valid_at_risk],
                direction=direction,
                horizon_days=float(horizon),
            )
            records.append(
                {
                    "phenotype_id": phenotype_id,
                    "phenotype_name": str(definition["display_name"]),
                    "body_system": str(definition["body_system"]),
                    "horizon_days": int(horizon),
                    "panel_id": panel_map[feature_id],
                    "feature_id": feature_id,
                    "evaluation_split": "train",
                    "analysis_type": "future_cause_specific_cox_phewas",
                    "target_kind": "post_discharge_prognostic_association",
                    "n_case": len(case_values),
                    "n_control": len(control_values),
                    "n_at_risk": n_at_risk,
                    "n_event": n_event,
                    "n_censored": int(
                        np.sum(np.asarray(masks["censored"])[valid_at_risk])
                    ),
                    "n_competing_death": int(
                        np.sum(
                            np.asarray(masks["competing_death"])[
                                valid_at_risk
                            ]
                        )
                    ),
                    "marker_missingness_at_risk": (
                        marker_assessment["overall_missingness"]
                    ),
                    "case_missingness": (
                        1.0 - len(case_values) / len(case_rows)
                        if len(case_rows)
                        else 1.0
                    ),
                    "control_missingness": (
                        1.0 - len(control_values) / len(control_rows)
                        if len(control_rows)
                        else 1.0
                    ),
                    "raw_auc": raw_auc,
                    "landmark_complete_followup_auc": raw_auc,
                    "direction": direction,
                    "aligned_auc": (
                        max(raw_auc, 1.0 - raw_auc)
                        if np.isfinite(raw_auc)
                        else np.nan
                    ),
                    "rank_biserial": (
                        2.0 * raw_auc - 1.0
                        if np.isfinite(raw_auc)
                        else np.nan
                    ),
                    **absolute,
                    "cox_log_hazard_ratio_per_sd": survival[
                        "log_hazard_ratio_per_sd"
                    ],
                    "cox_hazard_ratio_per_sd": survival[
                        "hazard_ratio_per_sd"
                    ],
                    "cox_hr_ci_low": survival["hr_ci_low"],
                    "cox_hr_ci_high": survival["hr_ci_high"],
                    "p_value": survival["p_value"],
                    "p_value_method": "adjusted_cause_specific_cox_wald",
                    "survival_model_status": survival["status"],
                    "cox_adjustment_covariates": (
                        "age|sex|heart_rate|ed_setting|acquisition_hour_cyclic|"
                        "anchor_era_start_year|"
                        "signal_quality|ecg_to_landmark_days|"
                        "index_length_of_stay_days"
                    ),
                    "cox_complete_case_test_yield": (
                        n_at_risk / len(at_risk_rows)
                        if len(at_risk_rows)
                        else 0.0
                    ),
                    "ph_diagnostic_p_value": survival[
                        "ph_diagnostic_p_value"
                    ],
                    "ph_assumption_gate_passed": survival[
                        "ph_assumption_gate_passed"
                    ],
                    "adjusted_model_gate_passed": bool(
                        survival["status"] == "evaluable"
                        and int(survival["adjustment_covariate_count"]) >= 2
                    ),
                    **marker_assessment,
                    **missingness_sensitivity,
                    **tie_sensitivity,
                    "confirmatory_assessability_gate_passed": bool(
                        marker_assessment[
                            "survival_marker_assessability_gate_passed"
                        ]
                        and missingness_sensitivity[
                            "missingness_sensitivity_gate_passed"
                        ]
                        and (
                            n_at_risk / len(at_risk_rows)
                            if len(at_risk_rows)
                            else 0.0
                        ) >= MINIMUM_SURVIVAL_MARKER_TEST_YIELD
                    ),
                    "average_precision": np.nan,
                    "inference_status": inference_status(
                        n_event,
                        max(0, n_at_risk - n_event),
                        minimum_inference=min_inference,
                        minimum_primary=min_formal,
                    ),
                    "label_state_policy": (
                        "post-discharge landmark; prior/same-discharge history "
                        "excluded; right censoring modeled; death censored "
                        "cause-specifically; same-day death/event ties excluded"
                    ),
                    "claim_scope": (
                        "adjusted_complete_case_cause_specific_association_only"
                    ),
                    "absolute_risk_claim_allowed": False,
                    "locked_test_scanned": False,
                }
            )
        if progress and target_index % 100 == 0:
            progress(
                f"Future-PheWAS Train {target_index:,}/"
                f"{len(selected_targets):,}"
            )
    return pd.DataFrame(records) if records else _empty_future_results()

def _future_adjustment_covariates(
    frame: pd.DataFrame,
    rows: np.ndarray,
) -> np.ndarray:
    scoped = frame.iloc[np.asarray(rows, dtype=int)]
    columns: list[np.ndarray] = []
    if "age" in scoped:
        columns.append(
            pd.to_numeric(scoped["age"], errors="coerce").to_numpy(dtype=float)
        )
    if "sex" in scoped:
        sex = scoped["sex"].astype("string").str.upper()
        columns.append(
            np.where(
                sex.isin(("M", "MALE")),
                1.0,
                np.where(sex.isin(("F", "FEMALE")), 0.0, np.nan),
            )
        )
    # Heart rate: strong non-specific outcome predictor and mechanically
    # correlated with most rate-dependent markers.
    rate = pd.to_numeric(
        scoped.get("heart_rate_bpm", pd.Series(np.nan, index=scoped.index)),
        errors="coerce",
    ).combine_first(
        pd.to_numeric(
            scoped.get(
                "machine_heart_rate_bpm",
                pd.Series(np.nan, index=scoped.index),
            ),
            errors="coerce",
        )
    )
    if rate.notna().any():
        columns.append(rate.to_numpy(dtype=float))
    if "ed_stay_id" in scoped:
        columns.append(scoped["ed_stay_id"].notna().to_numpy(dtype=float))
    if "ecg_time" in scoped:
        # Acquisition hour as a circular pair: a night recording exists for a
        # different reason than a morning one, and the hour is preserved by
        # MIMIC's date shifting.
        stamps = pd.to_datetime(scoped["ecg_time"], errors="coerce")
        hour = (
            stamps.dt.hour.astype(float)
            + stamps.dt.minute.astype(float) / 60.0
        ).to_numpy(dtype=float)
        columns.append(np.sin(2 * np.pi * hour / 24.0))
        columns.append(np.cos(2 * np.pi * hour / 24.0))
    if "anchor_year_group" in scoped:
        # Real era. ecg_time's year is de-identification shifted per patient
        # and carries no cross-patient calendar information.
        era = anchor_year_group_start(scoped["anchor_year_group"])
        if era.notna().any():
            columns.append(era.to_numpy(dtype=float))
    for name in (
        "signal_quality_score",
        "ecg_to_landmark_days",
        "index_length_of_stay_days",
    ):
        if name in scoped:
            columns.append(
                pd.to_numeric(scoped[name], errors="coerce").to_numpy(
                    dtype=float
                )
            )
    return (
        np.column_stack(columns)
        if columns
        else np.empty((len(scoped), 0), dtype=float)
    )

def _same_day_tie_sensitivity(
    frame: pd.DataFrame,
    values: np.ndarray,
    event_series: pd.Series,
    definition: pd.Series,
    *,
    horizon_days: int,
    primary_direction: int,
) -> dict[str, object]:
    primary = _future_state_masks(
        frame,
        event_series,
        definition,
        horizon_days=horizon_days,
        same_day_tie_policy="exclude",
    )
    ambiguous_count = int(
        np.sum(primary["ambiguous_same_day_death_event"])
    )
    if ambiguous_count == 0:
        return {
            "same_day_tie_count": 0,
            "same_day_tie_sensitivity_status": "no_ambiguous_ties",
            "same_day_tie_sensitivity_gate_passed": True,
            "event_first_log_hr": np.nan,
            "death_first_log_hr": np.nan,
        }
    effects: dict[str, float] = {}
    directions: dict[str, int] = {}
    for policy in ("event_first", "death_first"):
        masks = _future_state_masks(
            frame,
            event_series,
            definition,
            horizon_days=horizon_days,
            same_day_tie_policy=policy,
        )
        rows = np.flatnonzero(masks["at_risk"])
        fit = _fit_cause_specific_cox(
            np.asarray(values, dtype=float)[rows],
            np.asarray(masks["duration_days"])[rows],
            np.asarray(masks["event_observed"])[rows],
            _future_adjustment_covariates(frame, rows),
        )
        effects[policy] = float(fit["log_hazard_ratio_per_sd"])
        directions[policy] = int(fit["direction"])
    gate = bool(
        primary_direction in (-1, 1)
        and all(
            directions.get(policy) == primary_direction
            and np.isfinite(effects.get(policy, np.nan))
            for policy in ("event_first", "death_first")
        )
    )
    return {
        "same_day_tie_count": ambiguous_count,
        "same_day_tie_sensitivity_status": (
            "direction_robust" if gate else "direction_not_robust"
        ),
        "same_day_tie_sensitivity_gate_passed": gate,
        "event_first_log_hr": effects.get("event_first", np.nan),
        "death_first_log_hr": effects.get("death_first", np.nan),
    }
