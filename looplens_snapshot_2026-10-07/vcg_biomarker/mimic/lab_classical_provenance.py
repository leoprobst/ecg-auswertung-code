"""Capture current rule provenance at computation time, never at display time."""
from copy import deepcopy


def capture_classical_provenance(criteria, registry):
    from .classical import CONTIGUOUS_GROUPS, T_INVERSION_GROUPS
    output = deepcopy(criteria)
    registry_hash = registry.registry_hash
    source_lock_hash = registry.source_lock_hash
    by_criterion = {}
    for item in registry.all():
        by_criterion.setdefault(item.criterion_id, []).append(item)
    for name, result in output.items():
        definitions = by_criterion.get(name, [])
        result['registry_hash'] = registry_hash
        result['source_lock_sha256'] = source_lock_hash
        result['source_records'] = [dict(baseline_id=item.baseline_id,
            source_title=item.source_title, source_version=item.source_version,
            source_url=item.source_url, required_inputs=list(item.required_inputs),
            exclusions=list(item.exclusions), notes=item.notes,
            endpoint_ids=list(item.endpoint_ids)) for item in definitions]
        if name in {'st_elevation_contiguous', 'st_depression_contiguous', 'st_depression_contiguous_mm'}:
            result['contiguity_groups'] = [list(group) for group in CONTIGUOUS_GROUPS]
            result['contiguity_scope'] = 'implemented_extended_lateral_pairs_not_literal_UDMI_group_list'
        elif name == 't_inversion_contiguous':
            result['contiguity_groups'] = [list(group) for group in T_INVERSION_GROUPS]
            result['contiguity_scope'] = 'implemented_pairs_exclude_V1_and_aVR_with_R_S_condition'
        elif name == 'pathological_q_waves':
            result['contiguity_groups'] = [['I','aVL'], ['II','III','aVF'], ['V4','V5','V6']]
            result['contiguity_scope'] = 'implemented_Q_arm_V2_or_V3_separate_single_lead_arm_III_not_independently_eligible'
        result['provenance_schema_version'] = '1.0.0'
    return output
