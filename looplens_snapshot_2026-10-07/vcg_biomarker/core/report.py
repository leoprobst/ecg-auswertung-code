from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class ReportConfig:
    include_ecg: bool = True
    include_vcg_3d: bool = True
    include_vcg_2d: bool = True
    include_biomarker_table: bool = True
    include_analysis: bool = True
    include_reference_comparison: bool = False
    title: str = "VCG Biomarker Analysis Report"
    institution: str = ""


class ClinicalReport:
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self._pages = []

    def generate(self, processed_record, vcg_result=None, analysis_results=None,
                 reference_loop=None, output_path: Optional[Path] = None) -> Optional[bytes]:
        if not HAS_MATPLOTLIB:
            return None

        pr = processed_record
        ecg_record = pr.ecg_record
        biomarkers = pr.biomarker_results

        if not biomarkers and pr.biomarker_dict:
            from .biomarkers.base import BiomarkerResult, BiomarkerRegistry
            biomarkers = []
            for bname, bval in pr.biomarker_dict.items():
                try:
                    bm = BiomarkerRegistry.get(bname)
                    biomarkers.append(BiomarkerResult(
                        name=bname, value=bval, unit=bm.unit,
                        category=bm.category, confidence=1.0,
                        description=bm.description,
                    ))
                except Exception:
                    biomarkers.append(BiomarkerResult(
                        name=bname, value=bval, unit="",
                        category="unknown", confidence=0.0,
                    ))

        delineation = pr.delineation

        buf = io.BytesIO()

        with PdfPages(buf) as pdf:
            self._generate_summary_page(pdf, ecg_record, biomarkers)

            if self.config.include_ecg and pr.preprocessed and pr.preprocessed.get("cleaned") is not None:
                self._generate_ecg_page(pdf, pr)

            if self.config.include_vcg_2d and vcg_result is not None:
                self._generate_vcg_2d_page(pdf, vcg_result)

            if self.config.include_vcg_3d and vcg_result is not None:
                self._generate_vcg_3d_page(pdf, vcg_result)

            if self.config.include_biomarker_table:
                self._generate_biomarker_page(pdf, biomarkers)

            if self.config.include_analysis and analysis_results:
                self._generate_analysis_page(pdf, analysis_results)

        pdf_bytes = buf.getvalue()

        if output_path:
            output_path.write_bytes(pdf_bytes)

        return pdf_bytes

    def _generate_summary_page(self, pdf, ecg_record, biomarkers):
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")

        y = 0.95
        ax.text(0.5, y, self.config.title, fontsize=16, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)
        y -= 0.05
        if self.config.institution:
            ax.text(0.5, y, self.config.institution, fontsize=10, ha="center",
                    va="top", transform=ax.transAxes)
            y -= 0.03

        ax.text(0.5, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                fontsize=8, ha="center", va="top", transform=ax.transAxes)
        y -= 0.05

        ax.text(0.05, y, "Patient Information", fontsize=12, fontweight="bold",
                transform=ax.transAxes)
        y -= 0.03

        info_lines = [
            f"ECG ID: {ecg_record.ecg_id}",
            f"Age: {ecg_record.age or 'N/A'}",
            f"Sex: {ecg_record.sex or 'N/A'}",
            f"Date: {ecg_record.recording_date or 'N/A'}",
            f"Validated: {'Yes' if ecg_record.validated_by_human else 'No'}",
        ]
        for line in info_lines:
            ax.text(0.07, y, line, fontsize=9, transform=ax.transAxes)
            y -= 0.025

        y -= 0.03
        ax.text(0.05, y, "Key Biomarkers", fontsize=12, fontweight="bold",
                transform=ax.transAxes)
        y -= 0.03

        key_biomarkers = [
            "heart_rate", "qrs_duration", "pr_interval", "qt_interval",
            "qtc_bazett", "spatial_qrs_t_angle_mean", "qrs_vector_magnitude",
            "t_vector_magnitude", "svg_magnitude",
        ]

        for bname in key_biomarkers:
            bm = next((b for b in biomarkers if b.name == bname), None)
            if bm is None:
                continue

            val_str = f"{bm.value:.2f}" if bm.value is not None else "N/A"

            try:
                from .biomarkers.base import BiomarkerRegistry
                bm_obj = BiomarkerRegistry.get(bname)
                nr = bm_obj.normal_range(ecg_record.age, ecg_record.sex)
                range_str = f" ({nr[0]:.0f}-{nr[1]:.0f})" if nr[0] is not None else ""
                if bm.value is not None and nr[0] is not None:
                    if bm.value < nr[0] or bm.value > nr[1]:
                        marker = " [!]"
                    else:
                        marker = " [OK]"
                else:
                    marker = ""
                    range_str = ""
            except Exception:
                marker = ""
                range_str = ""

            text = f"{bm.description}: {val_str} {bm.unit}{range_str}{marker}"
            color = "red" if "[!]" in text else "black"
            ax.text(0.07, y, text, fontsize=9, color=color, transform=ax.transAxes)
            y -= 0.025

        y -= 0.03
        ax.text(0.05, y, "Diagnostic Classification", fontsize=12, fontweight="bold",
                transform=ax.transAxes)
        y -= 0.03

        for sc in ecg_record.diagnostic_superclass:
            ax.text(0.07, y, f"Superclass: {sc}", fontsize=9, transform=ax.transAxes)
            y -= 0.025
        for ssc in ecg_record.diagnostic_subclass:
            ax.text(0.07, y, f"Subclass: {ssc}", fontsize=9, transform=ax.transAxes)
            y -= 0.025

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _generate_biomarker_page(self, pdf, biomarkers):
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")

        y = 0.95
        ax.text(0.5, y, "Biomarker Detail Table", fontsize=14, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)
        y -= 0.05

        categories = ["ecg", "vcg", "loop"]
        for cat in categories:
            cat_biomarkers = [b for b in biomarkers if b.category == cat]
            if not cat_biomarkers:
                continue

            cat_name = {"ecg": "ECG Biomarkers", "vcg": "VCG Biomarkers",
                        "loop": "Loop Morphology"}.get(cat, cat)
            ax.text(0.05, y, cat_name, fontsize=11, fontweight="bold",
                    transform=ax.transAxes)
            y -= 0.025

            for bm in cat_biomarkers:
                val_str = f"{bm.value:.4f}" if bm.value is not None else "N/A"
                conf_str = f" (conf: {bm.confidence:.0%})" if bm.confidence < 0.8 else ""
                text = f"  {bm.name}: {val_str} {bm.unit}{conf_str}"
                if y < 0.05:
                    break
                ax.text(0.05, y, text, fontsize=7.5, transform=ax.transAxes)
                y -= 0.018

            y -= 0.02
            if y < 0.05:
                break

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _generate_ecg_page(self, pdf, processed_record):
        signal = processed_record.preprocessed.get("cleaned")
        if signal is None:
            return
        fs = processed_record.preprocessed.get("fs", 500)
        n_leads = min(signal.shape[1], 12)
        t = np.arange(signal.shape[0]) / fs

        fig, axes = plt.subplots(n_leads, 1, figsize=(8.27, 11.69), sharex=True)
        if n_leads == 1:
            axes = [axes]

        from .data_loader import LEAD_NAMES_12
        for i, ax in enumerate(axes):
            ax.plot(t, signal[:, i], linewidth=0.5, color="blue")
            ax.set_ylabel(LEAD_NAMES_12[i], fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xlim(t[0], t[-1])

        axes[-1].set_xlabel("Time (s)", fontsize=8)
        fig.suptitle("12-Lead ECG", fontsize=12, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _generate_vcg_2d_page(self, pdf, vcg_result):
        fig, axes = plt.subplots(1, 3, figsize=(8.27, 4))

        projections = [
            (vcg_result.vx, vcg_result.vy, "Frontal (X-Y)", "X (mV)", "Y (mV)"),
            (vcg_result.vx, vcg_result.vz, "Horizontal (X-Z)", "X (mV)", "Z (mV)"),
            (vcg_result.vy, vcg_result.vz, "Sagittal (Y-Z)", "Y (mV)", "Z (mV)"),
        ]

        n = len(vcg_result.vx)
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, n))

        for ax, (x, y, title, xlabel, ylabel) in zip(axes, projections):
            for i in range(n - 1):
                ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=colors[i], linewidth=0.8)
            ax.plot(0, 0, "ko", markersize=3)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        fig.suptitle("VCG Loop Projections", fontsize=12, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _generate_vcg_3d_page(self, pdf, vcg_result):
        fig = plt.figure(figsize=(8.27, 6))
        ax = fig.add_subplot(111, projection="3d")

        n = len(vcg_result.vx)
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, n))

        for i in range(n - 1):
            ax.plot(
                [vcg_result.vx[i], vcg_result.vx[i+1]],
                [vcg_result.vy[i], vcg_result.vy[i+1]],
                [vcg_result.vz[i], vcg_result.vz[i+1]],
                color=colors[i], linewidth=0.8,
            )

        ax.set_xlabel("X (mV)", fontsize=8)
        ax.set_ylabel("Y (mV)", fontsize=8)
        ax.set_zlabel("Z (mV)", fontsize=8)
        ax.tick_params(labelsize=6)

        ax.plot([0], [0], [0], "ko", markersize=3)

        fig.suptitle("3D VCG Loop", fontsize=12, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _generate_analysis_page(self, pdf, analysis_results):
        from .analysis.results import AnalysisRun

        if not isinstance(analysis_results, AnalysisRun):
            return

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")

        y = 0.95
        ax.text(0.5, y, "Analysis Results", fontsize=14, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)
        y -= 0.05

        significant = analysis_results.get_significant(min_auc=0.6)
        if not significant:
            ax.text(0.5, 0.5, "No significant biomarker-pathology associations found",
                    fontsize=12, ha="center", va="center", transform=ax.transAxes)
        else:
            sorted_sig = sorted(significant, key=lambda r: r.auc, reverse=True)
            for r in sorted_sig[:30]:
                text = (f"{r.biomarker_name} vs {r.pathology_name}: "
                        f"AUC={r.auc:.3f}, Thr={r.optimal_threshold:.4f}, "
                        f"Sens={r.optimal_sensitivity:.3f}, Spec={r.optimal_specificity:.3f}")
                if y < 0.05:
                    break
                ax.text(0.05, y, text, fontsize=7.5, transform=ax.transAxes)
                y -= 0.02

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
