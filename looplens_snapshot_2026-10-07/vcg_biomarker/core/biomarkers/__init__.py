from .ecg_biomarkers import register_ecg_biomarkers
from .vcg_biomarkers import register_vcg_biomarkers
from .loop_biomarkers import register_loop_biomarkers
from .catalog import register_candidate_biomarkers
from .catalog_biomarkers import register_catalog_biomarkers
from .advanced_vcg_biomarkers import register_advanced_vcg_biomarkers
from .electrophysiology_markers import register_electrophysiology_markers
from .discovery_v2_biomarkers import register_discovery_v2_biomarkers
from .discovery_v3_biomarkers import register_discovery_v3_biomarkers
from .discovery_v4_biomarkers import register_discovery_v4_biomarkers
from .segment_free_biomarkers import register_segment_free_biomarkers
from .morphology_variability_biomarkers import (
    register_morphology_variability_biomarkers,
)
from .literature_2026_biomarkers import register_literature_2026_biomarkers
from .literature_2026b_biomarkers import register_literature_2026b_biomarkers
from .invariant_biomarkers import register_invariant_biomarkers
from .propagation_sequence_biomarkers import (
    register_propagation_sequence_biomarkers,
)
from .loop_parametrisation_biomarkers import (
    register_loop_parametrisation_biomarkers,
)
from .thesis_derived_biomarkers import register_thesis_derived_biomarkers
from .ischemia_thesis_biomarkers import register_ischemia_thesis_biomarkers
from .physiology_thesis_biomarkers import register_physiology_thesis_biomarkers
from .decision_thesis_biomarkers import register_decision_thesis_biomarkers
from .conduction_uniformity_biomarkers import register_conduction_uniformity_biomarkers
from .qrs_t_coupling_extension_biomarkers import (
    register_qrs_t_coupling_extension_biomarkers,
)
from .circular_projection_biomarkers import (
    register_circular_projection_biomarkers,
)
from .base import BiomarkerRegistry
from .ndat_biomarkers import register_ndat_biomarkers
from .pathology_mechanism_biomarkers import register_pathology_mechanism_biomarkers


def register_all_biomarkers():
    register_candidate_biomarkers(limit=1155)
    register_ecg_biomarkers()
    register_vcg_biomarkers()
    register_loop_biomarkers()
    register_catalog_biomarkers()
    register_advanced_vcg_biomarkers()
    register_electrophysiology_markers()
    register_discovery_v2_biomarkers()
    register_discovery_v3_biomarkers()
    register_discovery_v4_biomarkers()
    register_segment_free_biomarkers()
    register_morphology_variability_biomarkers()
    register_literature_2026_biomarkers()
    register_literature_2026b_biomarkers()
    register_ndat_biomarkers()
    register_invariant_biomarkers()
    register_propagation_sequence_biomarkers()
    register_loop_parametrisation_biomarkers()
    register_thesis_derived_biomarkers()
    register_ischemia_thesis_biomarkers()
    register_decision_thesis_biomarkers()
    register_conduction_uniformity_biomarkers()
    register_pathology_mechanism_biomarkers()
    register_qrs_t_coupling_extension_biomarkers()
    register_physiology_thesis_biomarkers()
    # Last: every projection reads its source marker back out of the registry,
    # so all of them must already be in it.
    register_circular_projection_biomarkers()


register_all_biomarkers()
