from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AppConfig:
    ptbxl_path: Optional[Path] = None
    sampling_rate_hr: int = 500
    sampling_rate_lr: int = 100
    use_hr: bool = True

    vcg_method: str = "kors"

    delineation_weights: dict = field(default_factory=lambda: {
        "neurokit": 0.4,
        "multilead": 0.35,
        "vcg": 0.25,
    })
    delineation_conflict_threshold: float = 20.0

    sweep_percentile_steps: int = 100
    bootstrap_iterations: int = 1000
    significance_level: float = 0.05
    multiple_testing_correction: str = "fdr_bh"
    max_combination_tasks: int = 200_000
    greedy_combination_seeds: int = 3
    greedy_combination_max_depth: int = 2
    analysis_biomarker_chunk_size: int = 4
    checkpoint_interval_seconds: float = 120.0
    analysis_checkpoint_min_new_results: int = 128
    analysis_live_update_interval_seconds: float = 30.0
    analysis_live_update_min_new_results: int = 64
    analysis_progress_update_interval_seconds: float = 1.0

    analysis_compute_subgroups: bool = False
    subgroup_min_n: int = 30
    subgroup_auc_threshold: float = 0.65

    results_output_dir: Optional[Path] = None
    keep_signal_data: bool = False

    @property
    def sampling_rate(self) -> int:
        return self.sampling_rate_hr if self.use_hr else self.sampling_rate_lr
