"""A marker that changes when the frame turns is measuring the frame.

Three catalog markers took the standard deviation across the three spatial
components of one vector and called it a dispersion.  One of the three,
``spatial_ventricular_gradient_dispersion``, also read a key that the
producing branch does not write, so it raised KeyError on every recording
where the GEH metrics were computable at all -- 328.358 of the 630.044
extracted records, which is practically the entire biomarker-failure count of
the 2026-08-20 run, all under this one name.

These tests pin both halves: that the repaired markers produce a value, and
that the invariants stay put under rotation while the coordinate-wise marker
that is declared coordinate-wise does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.invariant_biomarkers import (
    INVARIANT_SPECS,
    _icosahedral_centres,
    _signature_level3,
    rotation_invariant_marker_ids,
)
from vcg_biomarker.core.delineation.base import RecordDelineation
from vcg_biomarker.tests.test_vcg_geh_scientific_contract import (
    _beat,
    _official_90757_qrst_mv,
)


FS = 500
_BEAT_SAMPLES = 300
_BEATS = 8


def _record(noise: float = 0.002, seed: int = 11):
    qrst = _official_90757_qrst_mv()
    xyz = np.zeros((_BEAT_SAMPLES * _BEATS, 3), dtype=float)
    beats = []
    for index in range(_BEATS):
        base = index * _BEAT_SAMPLES
        xyz[base + 40 : base + 40 + 184] = qrst * (1.0 + 0.05 * np.sin(index))
        beats.append(
            _beat(
                base + 58,
                base + 5,
                base + 15,
                base + 40,
                base + 81,
                base + 100,
                base + 223,
            )
        )
    if noise:
        xyz = xyz + np.random.default_rng(seed).normal(
            scale=noise, size=xyz.shape
        )
    return RecordDelineation(beats, FS, len(xyz), 90757), xyz


def _rotation(angle_z: float = 0.9, angle_x: float = 0.4) -> np.ndarray:
    around_z = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0.0],
            [np.sin(angle_z), np.cos(angle_z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    around_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle_x), -np.sin(angle_x)],
            [0.0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )
    return around_x @ around_z


def _value(name: str, delineation, xyz):
    return BiomarkerRegistry.get(name).compute(
        delineation, vcg_xyz=xyz, fs=FS
    ).value


def _record_without_a_qrs_segment():
    """A record whose QRS boundaries are missing, so the segment is unreadable.

    Not a flat signal: a flat signal still has bounds, and ``vcg_segment``
    returns the zeros rather than nothing. The dispatchers guard on the segment
    being absent, which is what a beat without an R onset and offset produces.
    """

    from vcg_biomarker.core.delineation.base import BeatDelineation, MarkerPoint

    beat = BeatDelineation(
        r_peak=MarkerPoint(sample=58, confidence=0.9, source="test"),
    )
    xyz = np.zeros((600, 3), dtype=float)
    return xyz, RecordDelineation([beat], FS, len(xyz), 7)


def test_the_repaired_svg_dispersion_produces_a_value():
    # 0 of 10.000 records carried one before: the consumer asked for
    # per_lead["x"] while the producer writes "svg_x_uv_ms".
    delineation, xyz = _record()
    value = _value("spatial_ventricular_gradient_dispersion", delineation, xyz)
    assert value is not None
    assert value > 0.0


def test_the_repaired_svg_dispersion_is_a_beat_to_beat_scatter():
    """The catalog asks for the spread of beat-level SVG vectors."""

    # Identical beats have no beat-to-beat spread; varying ones do.
    qrst = _official_90757_qrst_mv()
    for scales, expectation in (
        ((1.0, 1.0, 1.0, 1.0), 0.0),
        ((0.6, 0.9, 1.2, 1.5), 1.0),
    ):
        xyz = np.zeros((_BEAT_SAMPLES * len(scales), 3), dtype=float)
        beats = []
        for index, scale in enumerate(scales):
            base = index * _BEAT_SAMPLES
            xyz[base + 40 : base + 40 + 184] = qrst * scale
            beats.append(
                _beat(
                    base + 58,
                    base + 5,
                    base + 15,
                    base + 40,
                    base + 81,
                    base + 100,
                    base + 223,
                )
            )
        delineation = RecordDelineation(beats, FS, len(xyz), 90757)
        value = _value(
            "spatial_ventricular_gradient_dispersion", delineation, xyz
        )
        assert value is not None
        if expectation == 0.0:
            assert value == pytest.approx(0.0, abs=1e-6)
        else:
            assert value > 1.0


def test_a_single_beat_abstains_rather_than_returning_zero():
    qrst = _official_90757_qrst_mv()
    xyz = np.zeros((_BEAT_SAMPLES, 3), dtype=float)
    xyz[40 : 40 + 184] = qrst
    delineation = RecordDelineation(
        [_beat(58, 5, 15, 40, 81, 100, 223)], FS, len(xyz), 90757
    )
    result = BiomarkerRegistry.get(
        "spatial_ventricular_gradient_dispersion"
    ).compute(delineation, vcg_xyz=xyz, fs=FS)
    assert result.value is None
    assert "fewer than two" in result.computation_notes


def test_the_tpeak_tend_dispersion_is_also_beat_to_beat_now():
    delineation, xyz = _record()
    value = _value("tpeak_tend_vector_dispersion", delineation, xyz)
    assert value is not None
    assert value >= 0.0


@pytest.mark.parametrize("marker_id", sorted(rotation_invariant_marker_ids()))
def test_every_declared_invariant_survives_a_rotation(marker_id: str):
    delineation, xyz = _record()
    plain = _value(marker_id, delineation, xyz)
    turned = _value(marker_id, delineation, xyz @ _rotation().T)
    assert plain is not None, marker_id
    assert turned is not None, marker_id
    assert turned == pytest.approx(plain, rel=1e-9, abs=1e-12), marker_id


def test_the_two_repaired_dispersions_are_invariant_too():
    delineation, xyz = _record()
    for marker_id in (
        "spatial_ventricular_gradient_dispersion",
        "tpeak_tend_vector_dispersion",
    ):
        plain = _value(marker_id, delineation, xyz)
        turned = _value(marker_id, delineation, xyz @ _rotation().T)
        assert turned == pytest.approx(plain, rel=1e-9), marker_id


def test_the_coordinate_wise_dispersion_is_left_declared_not_repaired():
    """``vcg_t_area_dispersion_xyz`` asks for components, and gets them.

    It is frame dependent, and that is what its catalog entry says. Silently
    redefining a declared marker would be worse than leaving it: the point of
    the invariant module is to sit beside it, not to overwrite it.
    """

    delineation, xyz = _record()
    plain = _value("vcg_t_area_dispersion_xyz", delineation, xyz)
    turned = _value("vcg_t_area_dispersion_xyz", delineation, xyz @ _rotation().T)
    assert plain is not None and turned is not None
    assert abs(turned - plain) / abs(plain) > 1e-3


def test_the_signature_norm_does_not_change_with_the_sampling_grid():
    """Reparametrisation invariance is the whole reason level 3 is here."""

    times = np.linspace(0.0, 1.0, 120)
    path = np.column_stack(
        [np.sin(2 * np.pi * times), 0.6 * np.cos(2 * np.pi * times), 0.3 * times]
    )
    # The same shape traversed on a strongly non-uniform grid.
    warped = np.sort(np.concatenate([times**1.7, [1.0]]))
    resampled = np.column_stack(
        [np.interp(warped, times, path[:, axis]) for axis in range(3)]
    )
    uniform = np.linalg.norm(_signature_level3(path))
    nonuniform = np.linalg.norm(_signature_level3(resampled))
    assert nonuniform == pytest.approx(uniform, rel=2e-3)


def test_the_signature_scale_is_itself_rotation_invariant():
    """The first version divided by the bounding box, which turns with the frame.

    That made the marker move by 2,1 % under a rotation -- the exact defect the
    module exists to remove, reintroduced by the normalisation rather than by
    the tensor.
    """

    delineation, xyz = _record()
    plain = _value("qrs_signature_level3_norm", delineation, xyz)
    turned = _value("qrs_signature_level3_norm", delineation, xyz @ _rotation().T)
    assert turned == pytest.approx(plain, rel=1e-9)


def test_the_icosahedral_sectors_are_twenty_congruent_faces():
    centres = _icosahedral_centres()
    assert centres.shape == (20, 3)
    assert np.allclose(np.linalg.norm(centres, axis=1), 1.0)
    # Congruent: every face centre has the same nearest-neighbour distance.
    distances = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    assert np.allclose(nearest, nearest[0], atol=1e-9)


def test_the_new_markers_do_not_displace_a_curated_panel():
    """The trap the TQ-window module fell into, pinned so it cannot recur.

    Curated panels have fixed sizes and are filled by a family-balanced round
    robin, so a marker admitted to the candidate pool pushes another one out
    rather than joining. Registering thirteen TQ markers without declaring
    them post-outcome removed eight sentinel markers, ``p_loop_linearity``
    among them.
    """

    from vcg_biomarker.mimic.panels import default_feature_panels

    panels = {panel.panel_id: set(panel.feature_ids) for panel in default_feature_panels()}
    new = {spec.name for spec in INVARIANT_SPECS}
    assert len(panels["primary_core"]) == 80
    assert len(panels["sentinel"]) == 240
    assert not new & panels["primary_core"]
    assert not new & panels["sentinel"]
    assert new <= panels["discovery_reserve"]


def test_every_marker_declares_a_unit_a_formula_and_a_reason():
    for spec in INVARIANT_SPECS:
        assert spec.unit
        assert spec.formula
        assert spec.evidence_note
        assert spec.family
        definition = BiomarkerRegistry.definition(spec.name)
        assert definition.unit == spec.unit
        assert definition.source == "rotation_invariant_v1"


def test_every_marker_abstains_instead_of_raising_on_an_empty_record():
    xyz = np.zeros((600, 3), dtype=float)
    delineation = RecordDelineation(
        [_beat(58, 5, 15, 40, 81, 100, 223)], FS, len(xyz), 1
    )
    for spec in INVARIANT_SPECS:
        result = BiomarkerRegistry.get(spec.name).compute(
            delineation, vcg_xyz=xyz, fs=FS
        )
        assert result.value is None or np.isfinite(result.value), spec.name


def test_the_tremor_ratio_separates_a_systolic_from_a_diastolic_band_signal():
    """The marker of the pair that is worth reading, and why.

    A body tremor is on the skin for the whole strip; fibrillatory activity
    exists only between the ventricular complexes. Reading band power alone
    cannot tell them apart -- the ratio can.
    """

    delineation, xyz = _record(noise=0.0)
    fs = FS
    samples = np.arange(len(xyz)) / fs
    tone = 0.05 * np.sin(2 * np.pi * 5.0 * samples)

    everywhere = xyz.copy()
    everywhere[:, 0] += tone

    diastole_only = xyz.copy()
    mask = np.ones(len(xyz), dtype=bool)
    for index in range(_BEATS):
        base = index * _BEAT_SAMPLES
        mask[base + 40 : base + 224] = False
    diastole_only[mask, 0] += tone[mask]

    spread = _value("tremor_band_systole_to_diastole_ratio", delineation, everywhere)
    confined = _value(
        "tremor_band_systole_to_diastole_ratio", delineation, diastole_only
    )
    assert spread is not None and confined is not None
    assert spread > confined


def test_a_record_without_a_qrs_does_not_empty_the_next_records_markers():
    """The dispatch cache bound a marker to whoever answered first.

    ``_experimental_feature`` and ``_morph_signal_feature`` answered before
    looking at the name: when the QRS segment was unusable they returned an
    abstention for *every* name that reached them, and the dispatch loop cached
    that as ownership for the life of the worker process. The fifteen panel
    markers owned by the five dispatchers after ``_experimental_feature`` then
    came back empty from that worker for every later recording -- among them
    ``spatial_peak_qrs_t_angle_deg`` and
    ``qrs_fragmentation_high_frequency_energy``.

    So the same recording could give different values depending on which
    worker took it and what that worker happened to see first. This is the
    order that produced it.
    """

    from vcg_biomarker.core.biomarkers import catalog_biomarkers as catalog

    # Three of the fifteen. ``qrs_fragmentation_high_frequency_energy`` is a
    # fourth, left out here only because it abstains on this fixture for its
    # own reason -- it needs a wideband acquisition -- so it cannot show the
    # difference between being emptied and abstaining.
    affected = (
        "spatial_ventricular_gradient_dispersion",
        "spatial_peak_qrs_t_angle_deg",
        "microvolt_t_wave_alternans_vector",
    )
    saved = dict(catalog._CATALOG_DISPATCH_CACHE)
    try:
        for marker_id in affected:
            catalog._CATALOG_DISPATCH_CACHE.pop(marker_id, None)

        empty, broken = _record_without_a_qrs_segment()
        for marker_id in affected:
            BiomarkerRegistry.get(marker_id).compute(
                broken, vcg_xyz=empty, fs=FS
            )

        # Now a good record, in the same process.
        delineation, xyz = _record()
        for marker_id in affected:
            result = BiomarkerRegistry.get(marker_id).compute(
                delineation, vcg_xyz=xyz, fs=FS
            )
            assert result.value is not None, (
                f"{marker_id} was emptied by an earlier unusable record: "
                f"{result.computation_notes}"
            )
    finally:
        catalog._CATALOG_DISPATCH_CACHE.clear()
        catalog._CATALOG_DISPATCH_CACHE.update(saved)


def test_an_unavailable_input_never_claims_a_name_it_does_not_own():
    from vcg_biomarker.core.biomarkers import catalog_biomarkers as catalog

    empty, broken = _record_without_a_qrs_segment()
    context = catalog._context(broken, None, empty, FS)
    assert context.vcg_segment("qrs") is None
    for dispatcher in (
        catalog._experimental_feature,
        catalog._morph_signal_feature,
    ):
        answer = dispatcher(context, "spatial_ventricular_gradient_dispersion")
        assert isinstance(answer, catalog._InputUnavailable), dispatcher.__name__
