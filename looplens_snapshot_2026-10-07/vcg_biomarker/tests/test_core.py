import warnings

import numpy as np
import pytest
from vcg_biomarker.core.vcg_transform import (
    KorsTransform, InverseDowerTransform, VCGTransformRegistry,
    KORS_MATRIX, INVERSE_DOWER_MATRIX,
)
from vcg_biomarker.core.biomarkers.base import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.ecg_biomarkers import register_ecg_biomarkers
from vcg_biomarker.core.biomarkers.vcg_biomarkers import register_vcg_biomarkers
from vcg_biomarker.core.biomarkers.loop_biomarkers import register_loop_biomarkers
from vcg_biomarker.core.pathologies.registry import PathologyRegistry
from vcg_biomarker.core.pathologies.ptbxl_pathologies import register_ptbxl_pathologies
from vcg_biomarker.core.delineation.base import BeatDelineation, MarkerPoint, RecordDelineation
from vcg_biomarker.core.analysis.threshold_sweep import (
    compute_roc_curve, compute_auc, compute_auprc, threshold_sweep,
)
from vcg_biomarker.core.analysis.results import AnalysisRun
from vcg_biomarker.core.data_loader import ECGRecord
from vcg_biomarker.core.pipeline import Pipeline, PipelineState, ProcessedRecord
from vcg_biomarker.core.parallel_pipeline import ParallelPipeline
from vcg_biomarker.core.cache import ParquetEngineUnavailableError, ResultCache
from vcg_biomarker.config import AppConfig


class TestVCGTransform:
    def test_kors_matrix_shape(self):
        assert KORS_MATRIX.shape == (3, 8)

    def test_inverse_dower_matrix_shape(self):
        assert INVERSE_DOWER_MATRIX.shape == (3, 8)

    def test_kors_transform(self):
        ecg = np.random.randn(1000, 12)
        transform = KorsTransform()
        result = transform.transform(ecg)
        assert result.vx.shape == (1000,)
        assert result.vy.shape == (1000,)
        assert result.vz.shape == (1000,)
        assert result.method_name == "kors"

    def test_idt_transform(self):
        ecg = np.random.randn(1000, 12)
        transform = InverseDowerTransform()
        result = transform.transform(ecg)
        assert result.vx.shape == (1000,)
        assert result.vy.shape == (1000,)
        assert result.vz.shape == (1000,)
        assert result.method_name == "inverse_dower"

    def test_registry_available(self):
        methods = VCGTransformRegistry.available()
        assert "kors" in methods
        assert "inverse_dower" in methods

    def test_spatial_magnitude(self):
        ecg = np.random.randn(1000, 12)
        result = VCGTransformRegistry.transform(ecg, "kors")
        sm = result.spatial_magnitude
        assert sm.shape == (1000,)
        assert np.all(sm >= 0)

    def test_spatial_velocity(self):
        ecg = np.random.randn(1000, 12)
        result = VCGTransformRegistry.transform(ecg, "kors")
        sv = result.spatial_velocity
        assert sv.shape == (999,)


class TestBiomarkerRegistry:
    @classmethod
    def setup_class(cls):
        register_ecg_biomarkers()
        register_vcg_biomarkers()
        register_loop_biomarkers()

    def test_ecg_biomarkers_registered(self):
        ecg_biomarkers = BiomarkerRegistry.by_category("ecg")
        assert len(ecg_biomarkers) >= 12

    def test_vcg_biomarkers_registered(self):
        vcg_biomarkers = BiomarkerRegistry.by_category("vcg")
        assert len(vcg_biomarkers) >= 11

    def test_loop_biomarkers_registered(self):
        loop_biomarkers = BiomarkerRegistry.by_category("loop")
        assert len(loop_biomarkers) >= 8

    def test_candidate_catalog_all_pdf_markers_registered(self):
        pdf_definitions = [
            definition for definition in BiomarkerRegistry.definitions()
            if definition.source == "biomarker.pdf"
        ]
        assert len(pdf_definitions) == 1155

        final_marker = BiomarkerRegistry.definition("vcg_vector_field_divergence_proxy")
        assert final_marker.source_index == 1155
        assert final_marker.unit == "mixed"
        assert final_marker.implementation_status == "implemented"
        assert "threshold sweep" in final_marker.pathological_note
        assert all(definition.implementation_status == "implemented" for definition in pdf_definitions)

    def test_candidate_alias_lookup(self):
        definition = BiomarkerRegistry.definition("vcg_x_max")
        assert definition.name == "vcg_x_max/min"
        assert "vcg_x_min" in definition.aliases
        assert "vcg_x_max" not in BiomarkerRegistry.available()

    def test_lead_to_vcg_consistency_is_not_a_tautological_forward_fit(self):
        rng = np.random.default_rng(42)
        ecg = rng.normal(size=(1000, 12))
        transformed = VCGTransformRegistry.transform(ecg, "kors")
        vcg = np.column_stack((transformed.vx, transformed.vy, transformed.vz))
        delineation = RecordDelineation(
            beats=[],
            fs=500,
            n_samples=len(ecg),
            ecg_id=1,
        )

        result = BiomarkerRegistry.get(
            "lead_to_vcg_consistency_score"
        ).compute(delineation, ecg, vcg, 500)

        assert result.value is not None
        assert 0.0 <= result.value < 0.9
        assert result.per_lead

    def test_svg_magnitude_uses_uv_ms_units(self):
        fs = 1000
        beat = BeatDelineation(
            r_peak=MarkerPoint(sample=120, confidence=1.0, source="test"),
            p_onset=MarkerPoint(sample=20, confidence=1.0, source="test"),
            p_offset=MarkerPoint(sample=40, confidence=1.0, source="test"),
            r_onset=MarkerPoint(sample=80, confidence=1.0, source="test"),
            r_offset=MarkerPoint(sample=180, confidence=1.0, source="test"),
            t_onset=MarkerPoint(sample=210, confidence=1.0, source="test"),
            t_offset=MarkerPoint(sample=410, confidence=1.0, source="test"),
        )
        delineation = RecordDelineation(beats=[beat], fs=fs, n_samples=411, ecg_id=1)
        origin = np.asarray([0.1, -0.2, 0.3])
        vcg_xyz = np.tile(origin, (411, 1))
        vcg_xyz[80:181, 0] += 1.0
        # This includes the ST/JT interval; an implementation starting at the
        # visual T onset would omit a material part of the SVG.
        vcg_xyz[180:, 1] += 2.0

        result = BiomarkerRegistry.get("svg_magnitude").compute(delineation, vcg_xyz=vcg_xyz, fs=fs)

        assert result.unit == "uV*ms"
        expected_vector = np.trapezoid(
            vcg_xyz[80:411] - origin,
            dx=1.0,
            axis=0,
        ) * 1000.0
        assert result.value == pytest.approx(np.linalg.norm(expected_vector))
        assert "T-area=JT" in result.computation_notes

    def test_loop_axis_ratio_and_width_use_main_plane_axes(self):
        fs = 1000
        n = 240
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        vcg_xyz = np.column_stack([
            2.0 * np.cos(theta),
            1.0 * np.sin(theta),
            np.zeros_like(theta),
        ])
        beat = BeatDelineation(
            r_peak=MarkerPoint(sample=n // 4, confidence=1.0, source="test"),
            r_onset=MarkerPoint(sample=0, confidence=1.0, source="test"),
            r_offset=MarkerPoint(sample=n - 1, confidence=1.0, source="test"),
        )
        delineation = RecordDelineation(beats=[beat], fs=fs, n_samples=n, ecg_id=1)

        ratio = BiomarkerRegistry.get("qrs_loop_max_min_ratio").compute(delineation, vcg_xyz=vcg_xyz, fs=fs)
        width = BiomarkerRegistry.get("qrs_loop_width").compute(delineation, vcg_xyz=vcg_xyz, fs=fs)

        assert ratio.value == pytest.approx(2.0, rel=0.03)
        assert width.value == pytest.approx(2000.0, rel=0.03)

    def test_zero_vectors_do_not_create_zero_degree_angles(self):
        fs = 1000
        beat = BeatDelineation(
            r_peak=MarkerPoint(sample=120, confidence=1.0, source="test"),
            p_onset=MarkerPoint(sample=20, confidence=1.0, source="test"),
            p_offset=MarkerPoint(sample=40, confidence=1.0, source="test"),
            r_onset=MarkerPoint(sample=80, confidence=1.0, source="test"),
            r_offset=MarkerPoint(sample=180, confidence=1.0, source="test"),
            t_onset=MarkerPoint(sample=210, confidence=1.0, source="test"),
            t_offset=MarkerPoint(sample=410, confidence=1.0, source="test"),
        )
        delineation = RecordDelineation(beats=[beat], fs=fs, n_samples=411, ecg_id=1)
        vcg_xyz = np.zeros((411, 3))

        result = BiomarkerRegistry.get("spatial_qrs_t_angle_mean").compute(
            delineation, vcg_xyz=vcg_xyz, fs=fs
        )

        assert result.value is None
        assert "too small" in result.computation_notes

    def test_compute_heart_rate(self):
        fs = 500
        r_peaks = np.array([100, 600, 1100, 1600])
        beat = BeatDelineation(r_peak=MarkerPoint(sample=600, confidence=0.9, source="test"))

        delineation = RecordDelineation(
            beats=[
                BeatDelineation(r_peak=MarkerPoint(sample=s, confidence=0.9, source="test"))
                for s in r_peaks
            ],
            fs=fs,
            n_samples=2000,
            ecg_id=1,
        )

        hr_biomarker = BiomarkerRegistry.get("heart_rate")
        result = hr_biomarker.compute(delineation)
        assert result.value is not None
        assert 50 < result.value < 120

    def test_catalog_correlations_are_warning_free(self):
        fs = 500
        n = 1200
        t = np.linspace(0.0, 1.0, n)
        vcg_xyz = np.column_stack([
            800.0 * t ** 2,
            200.0 * np.sin(5.0 * np.pi * t),
            80.0 * t,
        ])
        ecg = np.tile(np.linalg.norm(vcg_xyz, axis=1)[:, None], (1, 12))

        beats = []
        for center in [200, 430, 700, 960]:
            beats.append(BeatDelineation(
                r_peak=MarkerPoint(sample=center, confidence=0.9, source="test"),
                q_peak=MarkerPoint(sample=center - 20, confidence=0.9, source="test"),
                r_onset=MarkerPoint(sample=center - 35, confidence=0.9, source="test"),
                r_offset=MarkerPoint(sample=center + 35, confidence=0.9, source="test"),
                s_peak=MarkerPoint(sample=center + 20, confidence=0.9, source="test"),
                j_point=MarkerPoint(sample=center + 40, confidence=0.9, source="test"),
                t_onset=MarkerPoint(sample=center + 80, confidence=0.9, source="test"),
                t_peak=MarkerPoint(sample=center + 120, confidence=0.9, source="test"),
                t_offset=MarkerPoint(sample=center + 170, confidence=0.9, source="test"),
            ))
        delineation = RecordDelineation(beats=beats, fs=fs, n_samples=n, ecg_id=1)

        names = [
            "loop_curvature_speed_correlation",
            "loop_speed_shape_correlation",
            "qrs_t_vector_correlation_distance",
            "qt_rr_hysteresis_surrogate",
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            results = [
                BiomarkerRegistry.get(name).compute(
                    delineation, ecg_12lead=ecg, vcg_xyz=vcg_xyz, fs=fs
                )
                for name in names
            ]

        messages = [str(w.message) for w in caught]
        assert not any("invalid value encountered in divide" in msg for msg in messages)
        assert all(result.computation_notes.startswith("Error:") is False for result in results)


class TestPathologyRegistry:
    @classmethod
    def setup_class(cls):
        register_ptbxl_pathologies()

    def test_superclasses(self):
        sc = PathologyRegistry.superclasses()
        assert "NORM" in sc
        assert "MI" in sc
        assert "STTC" in sc
        assert "CD" in sc
        assert "HYP" in sc

    def test_check_all(self):
        scp_codes = {"NORM": 100}
        results = PathologyRegistry.check_all(scp_codes)
        assert results.get("NORM") == True

    def test_check_mi(self):
        scp_codes = {"IMI": 80}
        results = PathologyRegistry.check_all(scp_codes)
        assert results.get("IMI") == True

    def test_lvh(self):
        scp_codes = {"LVH": 90}
        results = PathologyRegistry.check_all(scp_codes)
        assert results.get("LVH") == True


class TestThresholdSweep:
    def test_roc_curve(self):
        np.random.seed(42)
        n = 200
        values = np.concatenate([
            np.random.normal(5, 1, n//2),
            np.random.normal(7, 1, n//2),
        ])
        labels = np.concatenate([np.zeros(n//2), np.ones(n//2)])

        fpr, tpr, thresholds = compute_roc_curve(values, labels)
        assert len(fpr) > 0
        assert len(tpr) > 0

    def test_auc(self):
        np.random.seed(42)
        n = 200
        values = np.concatenate([
            np.random.normal(5, 1, n//2),
            np.random.normal(7, 1, n//2),
        ])
        labels = np.concatenate([np.zeros(n//2), np.ones(n//2)])

        fpr, tpr, thresholds = compute_roc_curve(values, labels)
        auc = compute_auc(fpr, tpr)
        assert 0.5 < auc <= 1.0

    def test_threshold_sweep(self):
        np.random.seed(42)
        n = 200
        values = np.concatenate([
            np.random.normal(5, 1, n//2),
            np.random.normal(7, 1, n//2),
        ])
        labels = np.concatenate([np.zeros(n//2), np.ones(n//2)])

        result = threshold_sweep(values, labels, "test_biomarker", "test_pathology", bootstrap_iterations=10)
        assert result.auc > 0.5
        assert result.optimal_threshold > 0
        assert 0 <= result.optimal_sensitivity <= 1
        assert 0 <= result.optimal_specificity <= 1

    def test_evidence_metrics(self):
        np.random.seed(7)
        n = 200
        values = np.concatenate([
            np.random.normal(1, 0.3, n//2),
            np.random.normal(2, 0.3, n//2),
        ])
        labels = np.concatenate([np.zeros(n//2), np.ones(n//2)])

        result = threshold_sweep(values, labels, "candidate", "target", bootstrap_iterations=10)

        assert result.threshold_direction == "above"
        assert result.auprc > result.prevalence
        assert result.lr_positive >= 1
        assert result.lr_negative <= 1
        assert 0 <= result.balanced_accuracy <= 1
        assert 0 <= result.stability_score <= 1

    def test_inverse_biomarker_direction(self):
        np.random.seed(8)
        n = 200
        values = np.concatenate([
            np.random.normal(2, 0.3, n//2),
            np.random.normal(1, 0.3, n//2),
        ])
        labels = np.concatenate([np.zeros(n//2), np.ones(n//2)])

        result = threshold_sweep(values, labels, "low_is_bad", "target", bootstrap_iterations=10)

        assert result.auc > 0.5
        assert result.direction == -1.0
        assert result.threshold_direction == "below"

    def test_auprc_bounds(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        auprc = compute_auprc(scores, labels)
        assert 0.0 <= auprc <= 1.0
        assert auprc > 0.5

    def test_tiny_pvalue_survives_json_roundtrip(self):
        """Regression: p-values like 1e-50 must not become 1.0 after JSON roundtrip.

        Root cause: summary_dict rounds p_value to 6 decimal places → 0.0 for tiny
        values; from_json_dict used ``float(0.0) or 1.0`` which evaluates to 1.0
        because 0.0 is falsy in Python.
        """
        from vcg_biomarker.core.analysis.threshold_sweep import SweepResult
        labels = np.array([0] * 50 + [1] * 50)
        values = np.array([0.0] * 50 + [1.0] * 50)
        result = threshold_sweep(values, labels, "marker", "target", bootstrap_iterations=5)

        assert result.p_value < 1e-10, f"sanity: perfectly-separated data → tiny p, got {result.p_value}"
        assert result.is_significant

        d = result.to_json_dict(include_curves=False)
        r2 = SweepResult.from_json_dict(d)

        assert r2.p_value < 1e-10, f"p_value mangled to {r2.p_value} after roundtrip"
        assert r2.is_significant, "significant result must remain significant after roundtrip"

    def test_analysis_run_json_roundtrip(self, tmp_path):
        values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
        result = threshold_sweep(values, labels, "candidate", "target", bootstrap_iterations=10)
        run = AnalysisRun(
            run_id="json_test",
            biomarker_names=["candidate"],
            pathology_names=["target"],
            vcg_method="kors",
            results=[result],
        )
        run.apply_corrections()

        path = tmp_path / "evidence.json"
        run.export_json(path)
        loaded = AnalysisRun.import_json(path)

        assert loaded.run_id == "json_test"
        assert len(loaded.results) == 1
        assert loaded.results[0].threshold_direction == "above"

    def test_legacy_csv_import(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        csv_path = root / "result.csv"
        if not csv_path.exists():
            pytest.skip("legacy result.csv not available")

        run = AnalysisRun.import_csv(csv_path, run_id="legacy")
        assert run.results
        assert run.results[0].threshold_status in ("exploratory", "exploratory_score")
        assert "auc" in run.to_dataframe().columns


class TestDelineation:
    def test_beat_delineation_intervals(self):
        fs = 500
        beat = BeatDelineation(
            r_peak=MarkerPoint(sample=300, confidence=0.9, source="test"),
            p_onset=MarkerPoint(sample=200, confidence=0.8, source="test"),
            r_onset=MarkerPoint(sample=280, confidence=0.8, source="test"),
            r_offset=MarkerPoint(sample=340, confidence=0.8, source="test"),
            t_offset=MarkerPoint(sample=500, confidence=0.7, source="test"),
        )

        pr_samples = beat.pr_interval_samples
        assert pr_samples is not None
        assert pr_samples == 80

        pr_ms = beat.pr_interval_ms(fs)
        assert pr_ms is not None
        assert 100 < pr_ms < 300

        qrs_ms = beat.qrs_duration_ms(fs)
        assert qrs_ms is not None
        assert 50 < qrs_ms < 200

        qt_ms = beat.qt_interval_ms(fs)
        assert qt_ms is not None

    def test_missing_p_wave(self):
        beat = BeatDelineation(
            r_peak=MarkerPoint(sample=300, confidence=0.9, source="test"),
            p_wave_absent=True,
        )
        assert beat.p_wave_absent == True
        assert beat.p_onset is None


class TestPipeline:
    def test_pipeline_requires_data_loader(self):
        config = AppConfig()
        pipeline = Pipeline(config)
        with pytest.raises(RuntimeError, match="Data loader not initialized"):
            pipeline.process_records([1, 2, 3])

    def test_pipeline_data_loader_setter(self):
        config = AppConfig()
        pipeline = Pipeline(config)
        assert pipeline.data_loader is None
        pipeline.data_loader = "mock"
        assert pipeline.data_loader == "mock"

    def test_pipeline_state_idle(self):
        pipeline = Pipeline()
        assert pipeline.state == PipelineState.IDLE

    def test_run_analysis_preserves_missing_rate(self):
        pipeline = Pipeline(AppConfig())
        values = [np.nan, np.nan, 0, 0.1, 0.2, 0.3, 0.4, 1, 1.1, 1.2, 1.3, 1.4]
        records = []
        for idx, value in enumerate(values):
            record = ECGRecord(
                ecg_id=idx,
                patient_id=idx,
                age=None,
                sex=None,
                height=None,
                weight=None,
                recording_date=None,
                scp_codes={},
                report="",
                heart_axis=None,
                diagnostic_superclass=[],
                diagnostic_subclass=[],
                validated_by_human=False,
                strat_fold=0,
            )
            records.append(ProcessedRecord(
                ecg_record=record,
                preprocessed={},
                biomarker_dict={"marker": value},
                pathology_labels={"target": idx >= 7},
            ))
        pipeline._processed_records = records

        run = pipeline.run_analysis(bootstrap_iterations=5)

        assert len(run.results) == 1
        assert run.results[0].missing_rate == pytest.approx(2 / 12)

    def test_analysis_checkpoint_reuses_partial_results_when_settings_change(self):
        pipeline = ParallelPipeline(AppConfig())
        labels = np.array([0] * 10 + [1] * 10)
        values = np.array([0.0] * 10 + [1.0] * 10)
        single = threshold_sweep(
            values,
            labels,
            biomarker_name="marker_a",
            pathology_name="target",
            bootstrap_iterations=5,
        )
        combined = threshold_sweep(
            values,
            labels,
            biomarker_name="marker_a + marker_b",
            pathology_name="target",
            bootstrap_iterations=5,
        )
        checkpoint = pipeline._analysis_checkpoint_payload(
            run_id="old",
            biomarker_names=["marker_a", "marker_b"],
            pathology_names=["target"],
            correction_method="bonferroni",
            bootstrap_iterations=5,
            max_biomarker_combination=2,
            results=[single, combined],
        )
        checkpoint["schema"] = "vcg_biomarker.analysis_checkpoint.v1"

        compatible, reason = pipeline._analysis_checkpoint_compatible(
            checkpoint,
            ["marker_a", "marker_b", "marker_c"],
            ["target"],
        )
        restored = {}
        restored_count = pipeline._merge_analysis_checkpoint_results(
            checkpoint,
            restored,
            ["marker_a", "marker_b", "marker_c"],
            ["target"],
            max_biomarker_combination=1,
        )

        assert compatible, reason
        assert restored_count == 1
        assert ("marker_a", "target") in restored
        assert ("marker_a + marker_b", "target") not in restored

    def test_empty_analysis_run_does_not_clear_checkpoint(self, tmp_path):
        cache = ResultCache(tmp_path)
        assert cache.save_analysis_checkpoint({
            "run_id": "partial",
            "vcg_method": "kors",
            "biomarker_names": ["marker"],
            "pathology_names": ["target"],
            "results": [{"biomarker": "marker", "pathology": "target"}],
        })

        empty_run = AnalysisRun(
            run_id="empty",
            biomarker_names=[],
            pathology_names=[],
            vcg_method="kors",
            results=[],
            correction_method="fdr_bh",
            cohort_filter={},
            split_strategy="development_only",
            comparison_baseline={},
            analysis_mode="exploratory_threshold_sweep",
        )

        assert cache.save_analysis_run(empty_run)
        assert cache.has_analysis_checkpoint()


class TestCombinedAnalysis:
    def test_combined_score_computation(self):
        arr1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        arr2 = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

        config = AppConfig()
        pipeline = Pipeline(config)
        score = pipeline._compute_combined_score([arr1, arr2], method="zscore_sum")

        assert len(score) == 10
        assert not np.any(np.isnan(score))

    def test_combined_score_aligns_opposite_pathologic_directions(self):
        labels = np.array([0] * 20 + [1] * 20)
        high_is_bad = np.array([0.0] * 20 + [1.0] * 20)
        low_is_bad = np.array([1.0] * 20 + [0.0] * 20)

        pipeline = Pipeline(AppConfig())
        results = pipeline._run_combined_analysis(
            {
                "high_is_bad": high_is_bad,
                "low_is_bad": low_is_bad,
            },
            {"target": labels},
            combination_size=2,
            bootstrap_iterations=5,
        )

        assert len(results) == 1
        result = results[0]
        assert result.auc > 0.9
        assert result.threshold_status == "exploratory_score"
        assert result.validation_split == "development_in_sample"
        assert result.quality_summary["component_directions"] == {
            "high_is_bad": 1.0,
            "low_is_bad": -1.0,
        }

    def test_combined_analysis_skips_checkpointed_sweeps(self):
        labels = np.array([0] * 10 + [1] * 10)
        pipeline = Pipeline(AppConfig())
        emitted = []

        results = pipeline._run_combined_analysis(
            {
                "marker_a": np.array([0.0] * 10 + [1.0] * 10),
                "marker_b": np.array([1.0] * 10 + [0.0] * 10),
            },
            {"target": labels},
            combination_size=2,
            bootstrap_iterations=5,
            on_result=emitted.append,
            existing_keys={("marker_a + marker_b", "target")},
        )

        assert results == []
        assert emitted == []
        assert pipeline.progress.current_step == 1
        assert pipeline.progress.total_steps == 1

    def test_combined_expected_pairs_include_effective_combinations(self):
        labels = [0] * 10 + [1] * 10
        pipeline = Pipeline(AppConfig())
        records = []
        for idx, label in enumerate(labels):
            record = ECGRecord(
                ecg_id=idx,
                patient_id=idx,
                age=None,
                sex=None,
                height=None,
                weight=None,
                recording_date=None,
                scp_codes={},
                report="",
                heart_axis=None,
                diagnostic_superclass=[],
                diagnostic_subclass=[],
                validated_by_human=False,
                strat_fold=0,
            )
            records.append(ProcessedRecord(
                ecg_record=record,
                preprocessed={},
                biomarker_dict={
                    "high_is_bad": float(label),
                    "low_is_bad": float(1 - label),
                },
                pathology_labels={"target": bool(label)},
            ))
        pipeline._processed_records = records

        run = pipeline.run_analysis(bootstrap_iterations=5, max_biomarker_combination=2)

        assert len(run.results) == 3
        assert run.cohort_filter["n_expected_pairs"] == 3
        assert run.cohort_filter["n_current_pairs"] == 3

    def test_combined_analysis_respects_task_limit(self):
        config = AppConfig(max_combination_tasks=2)
        pipeline = Pipeline(config)
        logs = []
        pipeline.set_log_callback(logs.append)
        biomarker_data = {
            "a": np.linspace(0, 1, 20),
            "b": np.linspace(1, 2, 20),
            "c": np.linspace(2, 3, 20),
        }
        pathology_data = {"target": np.array([0] * 10 + [1] * 10)}

        results = pipeline._run_combined_analysis(
            biomarker_data,
            pathology_data,
            combination_size=2,
            bootstrap_iterations=5,
        )

        assert results == []
        assert any("max_combination_tasks" in message for message in logs)

    def test_combined_sweep(self):
        np.random.seed(42)
        n = 200
        bm1 = np.concatenate([np.random.normal(5, 1, n//2), np.random.normal(7, 1, n//2)])
        bm2 = np.concatenate([np.random.normal(3, 1, n//2), np.random.normal(6, 1, n//2)])
        labels = np.concatenate([np.zeros(n//2), np.ones(n//2)])

        zscore1 = (bm1 - np.mean(bm1)) / np.std(bm1)
        zscore2 = (bm2 - np.mean(bm2)) / np.std(bm2)
        combined = zscore1 + zscore2

        result = threshold_sweep(combined, labels, "bm1 + bm2", "test_path", bootstrap_iterations=10)
        assert result.auc > 0.5


class TestCache:
    def test_cache_store_and_load(self, tmp_path):
        from vcg_biomarker.core.cache import ResultCache
        cache = ResultCache(cache_dir=tmp_path)
        cache.store(
            ecg_id=1,
            biomarker_dict={"heart_rate": 75.0, "qrs_duration": 90.0},
            pathology_labels={"_MI": True, "NORM": False},
            quality_score=0.9,
            age=65,
            sex="M",
        )
        assert cache.is_cached(1)
        assert not cache.is_cached(2)

        records = cache.load_cached_records([1])
        assert len(records) == 1
        assert records[0].biomarker_dict["heart_rate"] == 75.0
        assert records[0].pathology_labels["_MI"] is True
        assert records[0].ecg_record.ecg_id == 1

    def test_cache_get_uncached_ids(self, tmp_path):
        from vcg_biomarker.core.cache import ResultCache
        cache = ResultCache(cache_dir=tmp_path)
        cache.store(ecg_id=1, biomarker_dict={"hr": 75.0}, pathology_labels={}, quality_score=0.9)
        cache.store(ecg_id=3, biomarker_dict={"hr": 80.0}, pathology_labels={}, quality_score=0.8)

        uncached = cache.get_uncached_ids([1, 2, 3, 4])
        assert uncached == [2, 4]

    def test_cache_config_invalidation(self, tmp_path):
        from vcg_biomarker.core.cache import ResultCache
        cache = ResultCache(cache_dir=tmp_path)
        cache.set_config_hash({"vcg_method": "kors"})
        cache.save_config_hash()
        assert cache.check_config_valid()

        cache2 = ResultCache(cache_dir=tmp_path)
        cache2.set_config_hash({"vcg_method": "inverse_dower"})
        assert not cache2.check_config_valid()

        cache3 = ResultCache(cache_dir=tmp_path)
        cache3.set_config_hash({"vcg_method": "kors"})
        assert cache3.check_config_valid()

    def test_parquet_only_cache_without_engine_reports_clear_error(self, tmp_path, monkeypatch):
        import vcg_biomarker.core.cache as cache_module

        cache = ResultCache(cache_dir=tmp_path)
        (tmp_path / "batch_results.parquet").write_bytes(b"not a real parquet file")
        monkeypatch.setattr(cache_module, "_HAS_PARQUET", False)

        with pytest.raises(ParquetEngineUnavailableError, match="no parquet engine is installed"):
            cache.get_analysis_arrays()

    def test_parquet_cache_is_read_when_engine_is_available(self, tmp_path):
        pytest.importorskip("pyarrow")
        cache = ResultCache(cache_dir=tmp_path)
        cache.store(
            ecg_id=1,
            biomarker_dict={"marker": 3.0},
            pathology_labels={"target": True},
            quality_score=1.0,
        )

        loaded = ResultCache(cache_dir=tmp_path)
        biomarker_data, pathology_data = loaded.get_analysis_arrays()

        assert biomarker_data["marker"].tolist() == [3.0]
        assert pathology_data["target"].tolist() == [1]

    def test_format_eta(self):
        from vcg_biomarker.core.pipeline import _format_eta
        assert _format_eta(30) == "30s"
        assert _format_eta(125) == "2m 5s"
        assert _format_eta(3665) == "1h 1m"
        assert _format_eta(float('nan')) == "—"


class TestNormativeRanges:
    def test_normative_ranges_compute_from_norm_cohort(self):
        from vcg_biomarker.core.analysis.normative_ranges import NormativeRangeStore

        # 60 NORM records: 30 M (ages 35–44), 30 F (ages 55–64)
        # 30 non-NORM records (should be excluded)
        n_norm = 60
        ages = np.array(
            [40.0] * 30 + [60.0] * 30 + [50.0] * 30,  # NORM M, NORM F, non-NORM
            dtype=np.float32,
        )
        sexes = np.array(
            ["M"] * 30 + ["F"] * 30 + ["M"] * 30, dtype=object
        )
        norm_mask = np.array([True] * n_norm + [False] * 30)
        rng = np.random.default_rng(0)
        bm_values = np.concatenate([
            rng.normal(2.5, 0.3, n_norm).astype(np.float32),
            rng.normal(5.0, 0.3, 30).astype(np.float32),  # non-NORM (excluded)
        ])

        store = NormativeRangeStore()
        store.compute({"qrs_area": bm_values}, ages, sexes, norm_mask)

        assert len(store) == 1
        assert "qrs_area" in store.biomarkers()

        overall = store.get_overall("qrs_area")
        assert overall is not None
        assert overall.label == "all"
        assert overall.n == 60
        assert overall.p5 < overall.p50 < overall.p95

        # Sex strata
        s_m = store.get("qrs_area", sex="M")
        assert s_m is not None
        assert s_m.label == "sex_M"
        assert s_m.n == 30

        s_f = store.get("qrs_area", sex="F")
        assert s_f is not None
        assert s_f.label == "sex_F"
        assert s_f.n == 30

        # Age lookup: age 40 → bin 30-49
        s_age = store.get("qrs_area", age=40)
        assert s_age is not None

        # Unknown biomarker returns None
        assert store.get("nonexistent") is None

    def test_normative_ranges_save_and_load_roundtrip(self, tmp_path):
        from vcg_biomarker.core.analysis.normative_ranges import NormativeRangeStore

        n = 40
        bm = np.random.default_rng(1).normal(1.0, 0.2, n).astype(np.float32)
        ages = np.full(n, 45.0, dtype=np.float32)
        sexes = np.array(["M"] * 20 + ["F"] * 20, dtype=object)
        norm_mask = np.ones(n, dtype=bool)

        store = NormativeRangeStore()
        store.compute({"marker": bm}, ages, sexes, norm_mask)
        path = tmp_path / "normative_ranges.pkl"
        store.save(path)

        loaded = NormativeRangeStore.load(path)
        assert len(loaded) == 1
        overall_orig = store.get_overall("marker")
        overall_loaded = loaded.get_overall("marker")
        assert overall_orig is not None and overall_loaded is not None
        assert overall_orig.p50 == pytest.approx(overall_loaded.p50, abs=1e-6)


class TestSubgroupAnalysis:
    def _make_pipeline_with_metadata(self, n_per_sex: int = 50, min_n: int = 5):
        config = AppConfig(subgroup_auc_threshold=0.65, subgroup_min_n=min_n)
        pipeline = Pipeline(config)
        # half M, half F; all age 40 (bin 30-49)
        ages = np.full(n_per_sex * 2, 40.0, dtype=np.float32)
        sexes = np.array(["M"] * n_per_sex + ["F"] * n_per_sex, dtype=object)
        pipeline._metadata_arrays = {"age": ages, "sex": sexes}
        return pipeline

    def test_subgroup_sweep_stored_in_subgroup_metrics(self):
        # 25 M-neg, 25 M-pos, 25 F-neg, 25 F-pos → each sex stratum has pos + neg
        labels = np.array([0] * 25 + [1] * 25 + [0] * 25 + [1] * 25)
        values = np.array([0.0] * 25 + [1.0] * 25 + [0.0] * 25 + [1.0] * 25, dtype=np.float32)
        ages = np.full(100, 40.0, dtype=np.float32)
        sexes = np.array(["M"] * 50 + ["F"] * 50, dtype=object)

        config = AppConfig(subgroup_auc_threshold=0.65, subgroup_min_n=5)
        pipeline = Pipeline(config)
        pipeline._metadata_arrays = {"age": ages, "sex": sexes}

        result = threshold_sweep(
            values, labels,
            biomarker_name="test_marker",
            pathology_name="test_path",
            bootstrap_iterations=5,
        )
        assert result.auc > 0.9

        pipeline._compute_subgroup_results(
            [result],
            {"test_marker": values},
            {"test_path": labels},
            bootstrap_iterations=5,
        )

        assert "sex_M" in result.subgroup_metrics
        assert "sex_F" in result.subgroup_metrics
        m = result.subgroup_metrics["sex_M"]
        assert m["n"] == 50
        assert "auc" in m
        assert "threshold" in m
        assert "sensitivity" in m
        assert "specificity" in m

    def test_subgroup_sweep_skips_low_auc_results(self):
        n = 50
        rng = np.random.default_rng(7)
        labels = np.array([0] * n + [1] * n)
        values = rng.normal(0, 1, n * 2).astype(np.float32)  # random → low AUC

        pipeline = self._make_pipeline_with_metadata(n_per_sex=n)
        result = threshold_sweep(
            values, labels,
            biomarker_name="noise",
            pathology_name="test_path",
            bootstrap_iterations=5,
        )

        pipeline._compute_subgroup_results(
            [result],
            {"noise": values},
            {"test_path": labels},
            bootstrap_iterations=5,
        )
        # AUC is near 0.5 → below default threshold of 0.65 → no subgroup metrics
        assert result.subgroup_metrics == {}


class TestStratifiedExport:
    def test_stratified_export_columns_in_dataframe(self, tmp_path):
        import pandas as pd

        labels = np.array([0] * 20 + [1] * 20)
        values = np.array([0.0] * 20 + [1.0] * 20)
        result = threshold_sweep(
            values, labels,
            biomarker_name="marker",
            pathology_name="target",
            bootstrap_iterations=5,
        )
        result.normative_p5 = 0.05
        result.normative_p95 = 0.95
        result.subgroup_metrics = {
            "sex_M": {
                "n": 20, "n_pos": 10, "auc": 0.80,
                "threshold": 0.5, "threshold_direction": "above",
                "sensitivity": 0.80, "specificity": 0.80,
            },
            "sex_F": {
                "n": 20, "n_pos": 10, "auc": 0.75,
                "threshold": 0.5, "threshold_direction": "above",
                "sensitivity": 0.75, "specificity": 0.75,
            },
        }

        run = AnalysisRun(
            run_id="test",
            biomarker_names=["marker"],
            pathology_names=["target"],
            vcg_method="kors",
            results=[result],
        )

        df = run.to_dataframe()

        # Normative columns present
        assert "norm_p5" in df.columns
        assert "norm_p95" in df.columns
        assert df["norm_p5"].iloc[0] == pytest.approx(0.05, abs=1e-4)
        assert df["norm_p95"].iloc[0] == pytest.approx(0.95, abs=1e-4)

        # Sex subgroup columns present and correct
        assert "auc_sex_M" in df.columns
        assert "threshold_sex_M" in df.columns
        assert "sens_sex_M" in df.columns
        assert "spec_sex_M" in df.columns
        assert df["auc_sex_M"].iloc[0] == pytest.approx(0.80, abs=1e-4)
        assert df["auc_sex_F"].iloc[0] == pytest.approx(0.75, abs=1e-4)

        # Age subgroup columns present but empty (stratum not computed)
        assert "auc_age_18_29" in df.columns
        assert pd.isna(df["auc_age_18_29"].iloc[0])

        # CSV roundtrip preserves the columns
        csv_path = tmp_path / "export.csv"
        run.export_csv(csv_path)
        reloaded = pd.read_csv(csv_path)
        assert "norm_p5" in reloaded.columns
        assert "auc_sex_M" in reloaded.columns
        assert reloaded["auc_sex_M"].iloc[0] == pytest.approx(0.80, abs=1e-3)
