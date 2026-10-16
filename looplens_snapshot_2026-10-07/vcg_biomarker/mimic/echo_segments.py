"""Source-named resting LV segments; stress fields never certify resting motion."""

RWMA_CORE_SEGMENTS = (
    "ant_base", "ant_lat_base", "ant_sept_base", "inf_base", "inf_lat_base", "inf_sept_base",
    "ant_mid", "ant_lat_mid", "ant_sept_mid", "inf_mid", "inf_lat_mid", "inf_sept_mid",
    "ant_apex", "inf_apex", "lat_apex", "sept_apex",
)


def resting_wall_motion_coverage_sql(source: str) -> str:
    names = ", ".join(f"'{name}'" for name in (*RWMA_CORE_SEGMENTS, "apical_cap"))
    return f"""
        WITH named AS (
            SELECT measurement_id,
                regexp_replace(measurement, '^wm_(rest_)?', '') AS segment,
                LOWER(TRIM(COALESCE(result_raw, ''))) AS result
            FROM read_parquet('{source}')
            WHERE starts_with(measurement, 'wm_') AND NOT starts_with(measurement, 'wm_stress_')
        ), segments AS (
            SELECT measurement_id, segment,
                BOOL_OR(result IN ('normal', 'hyperdynamic')) AS normal,
                BOOL_OR(regexp_matches(result, '(not|poorly|unable).*(seen|visual|assess)')) AS not_visualized,
                BOOL_OR(regexp_matches(result, '^(hypokinetic|akinetic|dyskinetic)($|[^a-z])')) AS abnormal,
                BOOL_OR(result <> '' AND result NOT IN ('normal', 'hyperdynamic')
                    AND NOT regexp_matches(result, '^(hypokinetic|akinetic|dyskinetic)($|[^a-z])')) AS uncertain
            FROM named WHERE segment IN ({names}) GROUP BY measurement_id, segment
        )
        SELECT measurement_id,
            COUNT(*) AS rwma_reported_segments,
            COUNT(*) FILTER (WHERE normal) AS rwma_normal_segments,
            COUNT(*) FILTER (WHERE abnormal) AS rwma_abnormal_segments,
            COUNT(*) FILTER (WHERE not_visualized) AS rwma_not_visualized_segments,
            COUNT(*) FILTER (WHERE uncertain) AS rwma_uncertain_segments,
            BOOL_OR(normal AND abnormal) AS rwma_reference_conflict,
            (COUNT(*) FILTER (WHERE segment <> 'apical_cap' AND normal AND NOT abnormal AND NOT uncertain) = 16
                AND NOT BOOL_OR(abnormal OR uncertain)
                AND BOOL_AND(normal)) AS rwma_complete_normal_coverage
        FROM segments GROUP BY measurement_id
    """
