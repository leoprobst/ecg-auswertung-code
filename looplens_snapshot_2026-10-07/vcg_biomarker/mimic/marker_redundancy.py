"""Markers that cannot be told apart, grouped before anything is screened.

Measured on 5,924 development ECGs across the 1,736 markers with usable
variance:

======  ======
|rho| >= Paare
======  ======
0.9999     104
0.999      120
0.99       221
0.95       765
======  ======

At the 0.999 level that is 94 groups covering 201 markers, **107 of them
redundant** -- effectively 1,629 independent quantities where the catalog
lists 1,736. Some are naming variants of one computation:

* ``qrs_loop_nonplanarity`` = ``qrs_loop_planarity`` =
  ``qrs_loop_planarity_deviation``
* ``classical_qtc_fridericia`` = ``qtc_fridericia_ms`` =
  ``classical_qtc_gt_480`` = ``classical_qtc_gt_500`` -- two thresholds carry
  nothing over the continuous value they are cut from

Others are substantive findings in their own right:
``loop_near_self_approach_count`` is rank-identical to ``qrs_duration_ms``, so
that piece of loop geometry adds nothing to a measurement made since 1902.

**Why this must happen before the screen and not after.** Two consequences,
and only the first is cosmetic:

1. A "top 20 markers" list is dominated by whichever quantity has the most
   spellings, not by the strongest signal. The 2026-08-28 results showed one
   quantity appearing under five names in the same ranking.
2. The FDR family counts perfectly correlated tests as separate ones. That
   makes Benjamini-Hochberg conservative rather than wrong, but it also makes
   "21,099 replicated hits" a number about the catalog, not about the data.

**What this deliberately does not do.** Redundant markers are *labelled*, not
deleted. A marker excluded from the screening family still gets computed,
still appears in its cluster, and can still be reported -- the cluster
membership is what tells a reader that four rows are one finding. Dropping
them would destroy the information that they agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import MimicProjectConfig
from .feature_roles import is_candidate_biomarker
from .panels import STUDY_PANEL_IDS, default_feature_panels
from .storage import (
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    development_feature_glob,
    scientific_config_hash,
)


MARKER_REDUNDANCY_VERSION = "1.1.0"  # audit 5.3.4 effective number of tests

# Prespecified before any screen has run. 0.99 rather than 0.999: two markers
# sharing 98 % of their rank variance cannot support separate claims, and the
# measurement error of a single ECG marker (see fiducial_dependency) is of
# that order on its own. The looser threshold groups 221 pairs instead of 120.
REDUNDANCY_RHO_THRESHOLD = 0.99

# Rows drawn for the correlation. The full development cohort is 640,104
# ECGs; ranking all of them across 1,736 markers is 8.9 GB before pandas
# overhead, and buys nothing -- the standard error of a correlation at
# n = 50,000 is about 0.004, far below any threshold worth arguing about.
# The sample is seeded, so the grouping is reproducible.
REDUNDANCY_SAMPLE_ROWS = 50_000
REDUNDANCY_SAMPLE_SEED = 20260829

# A marker needs this many usable values in the sample before it can be
# compared at all. Below it a correlation says more about who is missing than
# about what is measured.
REDUNDANCY_MINIMUM_OBSERVATIONS = 500

# Panel order for electing a cluster representative. The clinically
# established marker wins over the exploratory one carrying the same signal.
_PANEL_RANK = {"primary_core": 0, "sentinel": 1, "discovery_reserve": 2}


_REDUNDANCY_COLUMNS = (
    "feature_id",
    "cluster_id",
    "cluster_size",
    "representative_feature_id",
    "is_representative",
    "abs_rho_to_representative",
    "coverage",
    "panel_rank",
    "distinct_values",
    "is_candidate",
    "classical_twin",
    "abs_rho_to_classical_twin",
)


@dataclass(frozen=True)
class RedundancyPolicy:
    """Everything that decides the grouping, in one auditable object."""

    rho_threshold: float = REDUNDANCY_RHO_THRESHOLD
    sample_rows: int = REDUNDANCY_SAMPLE_ROWS
    sample_seed: int = REDUNDANCY_SAMPLE_SEED
    minimum_observations: int = REDUNDANCY_MINIMUM_OBSERVATIONS

    def to_dict(self) -> dict:
        return {
            "rho_threshold": float(self.rho_threshold),
            "sample_rows": int(self.sample_rows),
            "sample_seed": int(self.sample_seed),
            "minimum_observations": int(self.minimum_observations),
            "version": MARKER_REDUNDANCY_VERSION,
        }


def panel_rank_map() -> dict[str, int]:
    """Marker -> panel rank, lowest wins when electing a representative."""

    ranks: dict[str, int] = {}
    for panel in default_feature_panels():
        rank = _PANEL_RANK.get(str(panel.panel_id), len(_PANEL_RANK))
        for feature_id in panel.feature_ids:
            name = str(feature_id)
            if name not in ranks or rank < ranks[name]:
                ranks[name] = rank
    return ranks


def rank_correlation_matrix(
    frame: pd.DataFrame,
    marker_ids: Sequence[str],
    *,
    minimum_observations: int = REDUNDANCY_MINIMUM_OBSERVATIONS,
) -> tuple[np.ndarray, list[str]]:
    """Spearman correlations between every usable pair of markers.

    Pearson on ranks *is* Spearman. Computed as one matrix rather than pair by
    pair: 1,736 columns is 1.5 million pairs, and a loop over them takes hours
    where the matrix takes seconds.
    """

    numeric = frame.reindex(columns=list(marker_ids)).apply(
        pd.to_numeric, errors="coerce"
    )
    usable = [
        name
        for name in marker_ids
        if int(numeric[name].notna().sum()) >= int(minimum_observations)
        and int(numeric[name].nunique(dropna=True)) > 1
    ]
    if not usable:
        return np.zeros((0, 0)), []
    ranks = numeric[usable].rank()
    matrix = ranks.corr(
        method="pearson", min_periods=int(minimum_observations)
    ).to_numpy()
    return matrix, usable


def cluster_markers(
    matrix: np.ndarray,
    names: Sequence[str],
    *,
    threshold: float = REDUNDANCY_RHO_THRESHOLD,
) -> list[list[str]]:
    """Connected components of "indistinguishable at ``threshold``".

    Transitive on purpose. If A and B agree and B and C agree, all three
    describe one quantity even when A and C fall just under the line -- and
    reporting them as two groups would put the same signal in a ranking
    twice, which is the defect this exists to remove.

    The sign is dropped: a marker and its negation carry the same information
    and would produce the same finding with the direction flipped.
    """

    count = len(names)
    if count == 0:
        return []
    neighbours: dict[int, set[int]] = {index: set() for index in range(count)}
    absolute = np.abs(matrix)
    for i in range(count):
        for j in range(i + 1, count):
            value = absolute[i, j]
            if np.isfinite(value) and value >= float(threshold):
                neighbours[i].add(j)
                neighbours[j].add(i)

    seen: set[int] = set()
    clusters: list[list[str]] = []
    for start in range(count):
        if start in seen:
            continue
        stack, component = [start], []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(neighbours[current] - seen)
        clusters.append(sorted(names[index] for index in component))
    return sorted(clusters, key=lambda group: (-len(group), group[0]))


def elect_representative(
    cluster: Sequence[str],
    coverage: Mapping[str, float],
    panel_ranks: Mapping[str, int],
    distinct_values: Optional[Mapping[str, int]] = None,
) -> str:
    """The marker that speaks for a cluster.

    Ordered so that information is never traded away for familiarity:

    1. **A measurement beats a threshold cut from it.** ``qrs_duration_ms``
       and ``classical_qrs_widening_ge_120`` are rank-equivalent here, but the
       millisecond value carries strictly more than the yes/no. Detected from
       the data -- more than ten distinct values -- not from the name, so a
       marker that happens to be binary in this cohort is judged as it is.
    2. **Then the panel.** Among equals, the marker in ``primary_core`` is the
       one a reader recognises.
    3. Coverage, then the name, so the choice is reproducible and does not
       depend on dictionary order.

    Classical baselines never reach this function: they are comparators, not
    candidates, and electing one would delete a candidate from the screen in
    favour of the yardstick it is measured against.
    """

    counts = distinct_values or {}

    def key(name: str) -> tuple:
        binary_like = int(counts.get(name, 1_000_000)) <= 10
        return (
            binary_like,
            int(panel_ranks.get(name, len(_PANEL_RANK))),
            -float(coverage.get(name, 0.0)),
            len(name),
            name,
        )

    return min(cluster, key=key)


def _closest_classical(
    name: str,
    classical: Sequence[str],
    matrix: np.ndarray,
    position: Mapping[str, int],
) -> tuple[Optional[str], float]:
    """The classical baseline a candidate most resembles, if any."""

    best_name, best_value = None, float("nan")
    if name not in position:
        return best_name, best_value
    row = position[name]
    for other in classical:
        value = matrix[row, position[other]]
        if not np.isfinite(value):
            continue
        value = float(abs(value))
        if not np.isfinite(best_value) or value > best_value:
            best_name, best_value = other, value
    return best_name, best_value


def build_redundancy_table(
    frame: pd.DataFrame,
    marker_ids: Sequence[str],
    *,
    policy: RedundancyPolicy = RedundancyPolicy(),
) -> pd.DataFrame:
    """One row per marker: its cluster, its representative, its agreement."""

    matrix, names = rank_correlation_matrix(
        frame, marker_ids, minimum_observations=policy.minimum_observations
    )
    numeric = frame.reindex(columns=list(marker_ids)).apply(
        pd.to_numeric, errors="coerce"
    )
    coverage = {
        name: float(numeric[name].notna().mean()) for name in marker_ids
    }
    distinct_values = {
        name: int(numeric[name].nunique(dropna=True)) for name in marker_ids
    }
    panel_ranks = panel_rank_map()
    position = {name: index for index, name in enumerate(names)}

    # Classical baselines are yardsticks, not candidates. Clustering them
    # together with candidates would let a comparator be elected -- and a
    # screen that then tested ``classical_qrs_widening_ge_120`` in place of
    # ``qrs_duration_ms`` would be measuring the ruler.
    candidates = [name for name in names if is_candidate_biomarker(name)]
    classical = [name for name in names if not is_candidate_biomarker(name)]
    candidate_matrix = matrix[
        np.ix_(
            [position[n] for n in candidates], [position[n] for n in candidates]
        )
    ] if candidates else np.zeros((0, 0))
    clusters = cluster_markers(
        candidate_matrix, candidates, threshold=policy.rho_threshold
    )

    records: list[dict] = []
    for number, cluster in enumerate(clusters):
        representative = elect_representative(
            cluster, coverage, panel_ranks, distinct_values
        )
        twin, twin_rho = _closest_classical(
            representative, classical, matrix, position
        )
        for name in cluster:
            if name == representative:
                agreement = 1.0
            else:
                value = matrix[position[name], position[representative]]
                agreement = float(abs(value)) if np.isfinite(value) else float("nan")
            records.append(
                {
                    "feature_id": name,
                    "cluster_id": f"C{number:04d}",
                    "cluster_size": len(cluster),
                    "representative_feature_id": representative,
                    "is_representative": name == representative,
                    "abs_rho_to_representative": agreement,
                    "coverage": coverage.get(name, float("nan")),
                    "panel_rank": int(
                        panel_ranks.get(name, len(_PANEL_RANK))
                    ),
                    "distinct_values": int(distinct_values.get(name, 0)),
                    "is_candidate": True,
                    # Not a grouping, a finding: a candidate that agrees this
                    # closely with a classical criterion is that criterion
                    # under another name, and a paper should say so.
                    "classical_twin": twin,
                    "abs_rho_to_classical_twin": twin_rho,
                }
            )

    # Markers that could not be compared at all are their own cluster, marked
    # as such: silently dropping them would make the screening family shrink
    # for a reason nobody could see.
    compared = {record["feature_id"] for record in records}
    for number, name in enumerate(
        sorted(set(marker_ids) - compared), start=len(clusters)
    ):
        records.append(
            {
                "feature_id": name,
                "cluster_id": f"C{number:04d}",
                "cluster_size": 1,
                "representative_feature_id": name,
                "is_representative": True,
                "abs_rho_to_representative": float("nan"),
                "coverage": coverage.get(name, float("nan")),
                "panel_rank": int(panel_ranks.get(name, len(_PANEL_RANK))),
                "distinct_values": int(distinct_values.get(name, 0)),
                "is_candidate": bool(is_candidate_biomarker(name)),
                "classical_twin": None,
                "abs_rho_to_classical_twin": float("nan"),
            }
        )
    if not records:
        return pd.DataFrame(columns=_REDUNDANCY_COLUMNS)
    return pd.DataFrame(records).sort_values(
        ["cluster_id", "feature_id"], ignore_index=True
    )


def screening_family(table: pd.DataFrame) -> tuple[str, ...]:
    """The markers a screen should treat as separate tests.

    Representatives only, and candidates only. A classical baseline forms its
    own singleton group -- it is still worth showing next to the candidate it
    matches -- but it is a yardstick, and a screen that tested the yardstick
    would be measuring itself.
    """

    if table.empty:
        return ()
    keep = table["is_representative"].astype(bool)
    if "is_candidate" in table:
        keep &= table["is_candidate"].astype(bool)
    return tuple(sorted(str(value) for value in table.loc[keep, "feature_id"]))


def effective_number_of_tests(matrix: np.ndarray) -> dict:
    """Effective number of independent tests from a correlation matrix.

    Audit 5.3.4. Two published estimators from the eigenvalues:

    * Li & Ji, Heredity 2005 (PMID 16077740):
      M_eff = sum f(|lambda_i|), f(x) = I(x >= 1) + (x - floor(x)).
    * Nyholt, Am J Hum Genet 2004 (PMID 14997420):
      M_eff = 1 + (M - 1) * (1 - Var(lambda) / M), Var with M - 1 in the
      denominator.

    A pair without enough joint observations has no correlation estimate;
    it is set to 0 (treated as independent), which can only raise M_eff.
    The pairwise-complete matrix need not be positive semidefinite, hence
    the absolute eigenvalues in Li & Ji.
    """

    size = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    if size == 0:
        return {"n_tests": 0, "m_eff_li_ji": 0.0, "m_eff_nyholt": 0.0}
    filled = np.where(np.isfinite(matrix), matrix, 0.0)
    filled = (filled + filled.T) / 2.0
    np.fill_diagonal(filled, 1.0)
    eigenvalues = np.linalg.eigvalsh(filled)
    absolute = np.abs(eigenvalues)
    li_ji = float(np.sum((absolute >= 1.0).astype(float) + (absolute - np.floor(absolute))))
    variance = float(np.var(eigenvalues, ddof=1)) if size > 1 else 0.0
    nyholt = float(1.0 + (size - 1) * (1.0 - variance / size)) if size > 1 else 1.0
    return {
        "n_tests": size,
        "m_eff_li_ji": li_ji,
        "m_eff_nyholt": nyholt,
        "bonferroni_alpha_0_05_li_ji": 0.05 / li_ji if li_ji > 0 else float("nan"),
        "n_missing_pairwise_correlations": int(np.sum(~np.isfinite(matrix)) // 2),
    }


def redundancy_summary(table: pd.DataFrame) -> dict:
    if table.empty:
        return {
            "n_markers": 0,
            "n_clusters": 0,
            "n_redundant": 0,
            "largest_cluster": 0,
        }
    sizes = table.groupby("cluster_id")["cluster_size"].first()
    return {
        "n_markers": int(len(table)),
        "n_clusters": int(len(sizes)),
        "n_redundant": int(len(table) - len(sizes)),
        "largest_cluster": int(sizes.max()),
        "n_clusters_above_one": int((sizes > 1).sum()),
    }


def sample_feature_rows(
    feature_glob: str,
    marker_ids: Sequence[str],
    *,
    policy: RedundancyPolicy = RedundancyPolicy(),
    catalog_path: Optional[Path] = None,
    memory_limit_gb: int = 8,
    threads: int = 4,
) -> pd.DataFrame:
    """A systematic sample of **training** ECGs, markers only.

    **Train only, on purpose.** Which markers are treated as one quantity is a
    preprocessing decision. Fitting it on validation rows would let the
    held-out split shape the family that is later tested on it -- a small leak,
    but a leak, and one that costs nothing to avoid.

    **Whole partitions rather than sampled rows.** DuckDB's reservoir sampling
    reads every row before it discards any: measured here, ``USING SAMPLE``
    over 1,527 partitions of 1,753 columns was still running after 13 minutes
    at 7.7 GB. Taking every k-th partition file reads about two hundred of
    them instead, at the same width, and spans the whole set rather than its
    beginning -- so nothing that varies along extraction order can bias it.
    """

    import glob as _glob

    import pyarrow.parquet as pq

    pattern = str(feature_glob).replace("\\", "/")
    train_pattern = pattern.replace("split=*", "split=train")
    files = sorted(_glob.glob(train_pattern))
    if not files:
        files = sorted(_glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=list(marker_ids))

    per_file = max(1, int(pq.ParquetFile(files[0]).metadata.num_rows))
    needed = max(1, -(-int(policy.sample_rows) // per_file))
    step = max(1, len(files) // needed)
    chosen = files[::step][:needed]

    wanted = list(dict.fromkeys(list(marker_ids) + ["extraction_status"]))
    pieces: list[pd.DataFrame] = []
    for path in chosen:
        # ``ParquetFile`` reads this one file.  ``pq.read_table`` would build a
        # dataset around it instead, and a dataset infers hive partitioning
        # from the parent directory: the ``split=train`` folder yields a
        # dictionary-typed ``split`` key that cannot merge with the plain
        # string ``split`` column the extraction writes into every file.  On
        # the real store that raised ArrowTypeError on the first partition, so
        # the stage could not read a single row of it.
        handle = pq.ParquetFile(path)
        available = set(handle.schema_arrow.names)
        columns = [name for name in wanted if name in available]
        frame = handle.read(columns=columns).to_pandas()
        if "extraction_status" in frame:
            frame = frame[frame["extraction_status"].eq("ok")]
        frame = frame.drop(columns=["extraction_status"], errors="ignore")
        # An empty piece contributes nothing but changes the concatenated
        # dtypes, which pandas warns about and which would silently turn a
        # numeric column into object.
        if not frame.empty:
            pieces.append(frame)
    if not pieces:
        return pd.DataFrame(columns=list(marker_ids))
    return pd.concat(pieces, ignore_index=True)


def run_marker_redundancy(
    config: MimicProjectConfig,
    *,
    run_id: str,
    panel_ids: Sequence[str] = STUDY_PANEL_IDS,
    feature_glob: Optional[str] = None,
    policy: RedundancyPolicy = RedundancyPolicy(),
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    """Group the marker catalog before anything is screened against it."""

    output_root = config.runs_root / run_id / "redundancy"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "marker_redundancy.manifest.json"

    panels = {panel.panel_id: panel for panel in default_feature_panels()}
    unknown = sorted(set(panel_ids) - set(panels))
    if unknown:
        raise ValueError(f"Unknown feature panels: {unknown}")
    marker_ids = tuple(
        dict.fromkeys(
            str(feature_id)
            for panel_id in panel_ids
            for feature_id in panels[panel_id].feature_ids
        )
    )
    if not marker_ids:
        raise ValueError("No markers selected for the redundancy stage")

    resolved_glob = feature_glob or development_feature_glob(
        config.features_root, panel_ids
    )
    input_hash = content_hash(
        {
            "run_id": run_id,
            "feature_glob": resolved_glob,
            "markers": list(marker_ids),
            "policy": policy.to_dict(),
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, MARKER_REDUNDANCY_VERSION):
            return previous

    if progress:
        progress(
            f"Redundanz: {len(marker_ids):,} Marker auf "
            f"{policy.sample_rows:,} EKGs vergleichen"
        )
    frame = sample_feature_rows(
        resolved_glob,
        marker_ids,
        policy=policy,
        memory_limit_gb=int(config.duckdb_memory_limit_gb),
        threads=int(config.workers),
    )
    table = build_redundancy_table(frame, marker_ids, policy=policy)
    summary = redundancy_summary(table)
    # Audit 5.3.4: the real multiplicity load beside the nominal count.
    matrix, names = rank_correlation_matrix(
        frame, marker_ids, minimum_observations=policy.minimum_observations
    )
    family = set(screening_family(table))
    family_index = [index for index, name in enumerate(names) if name in family]
    summary["effective_number_of_tests"] = {
        "all_markers": effective_number_of_tests(matrix),
        "screening_family": effective_number_of_tests(
            matrix[np.ix_(family_index, family_index)]
            if family_index else np.zeros((0, 0))
        ),
        "correlation": "spearman_on_the_redundancy_sample",
    }
    summary.update(
        {
            "run_id": run_id,
            "panel_ids": list(panel_ids),
            "policy": policy.to_dict(),
            "n_rows_sampled": int(len(frame)),
            "screening_family_size": len(screening_family(table)),
        }
    )
    if progress:
        progress(
            f"Redundanz: {summary['n_clusters']:,} Gruppen, "
            f"{summary['n_redundant']:,} Marker redundant"
        )

    table_path = output_root / "marker_redundancy.parquet"
    summary_path = output_root / "marker_redundancy_summary.json"
    atomic_write_parquet(table, table_path)
    atomic_write_json(summary_path, summary)

    manifest = StageManifest.complete(
        stage="marker-redundancy",
        stage_version=MARKER_REDUNDANCY_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(str(table_path.resolve()), str(summary_path.resolve())),
        row_counts={"markers": int(len(table)), "clusters": summary["n_clusters"]},
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "MARKER_REDUNDANCY_VERSION",
    "REDUNDANCY_RHO_THRESHOLD",
    "RedundancyPolicy",
    "build_redundancy_table",
    "cluster_markers",
    "elect_representative",
    "panel_rank_map",
    "rank_correlation_matrix",
    "redundancy_summary",
    "run_marker_redundancy",
    "sample_feature_rows",
    "screening_family",
]
