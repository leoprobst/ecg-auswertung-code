"""One configuration for the study, plus a preview arm for development.

There used to be five presets behind a dropdown labelled "choose only how
thoroughly to compute".  That framing was wrong in the dangerous direction,
and the choices had decayed into each other:

* ``final_development`` was byte-identical to ``recommended_clinical`` -- same
  panels, not one differing policy field -- and neither appeared in the
  interface;
* ``bootstrap_iterations`` was measured to saturate around 500 resamples, so
  the 1000-vs-2000 step bought nothing but weeks of arithmetic;
* the combination-search budgets became a runtime decision once the deep
  search turned open-ended, stoppable and resumable with a visible gain.
  Choosing 25 or 250 triples in advance answers a question the run now answers
  better.

What actually differed was **scope**, not thoroughness: 320 markers against
1753.  That is a five-fold change in the number of hypotheses and therefore in
the multiplicity burden -- a scientific decision wearing a performance dial's
clothing.  A user picking the slowest option because they had time was
changing what the study claims.

So the dial is gone.  The study runs at full scope, always, and the remaining
question -- how long to keep refining -- is answered while it runs, by
stopping when the reported gain stops moving.  That control already exists and
is already one button.

``fast_preview`` stays because development needs a smoke test that reaches
every stage in hours; it is deliberately not offered in the interface and its
results are not freezable.
"""

from __future__ import annotations

from dataclasses import replace

from .models import AnalysisPreset, ParameterPolicy
from .panels import STUDY_PANEL_IDS


BASE_POLICY = ParameterPolicy()

# The single study configuration.  The identifier is unchanged so that runs,
# manifests and queued jobs recorded before the collapse keep resolving.
STUDY_PRESET_ID = "exhaustive_research"
PREVIEW_PRESET_ID = "fast_preview"
# The full production path -- every stage the study runs after the pilots,
# with adjusted models, calibration, mechanism fits and score discovery
# switched on -- on a bounded cohort. ``fast_preview`` reaches every stage
# but as a preview: it skips the validate/redundancy/therapy/repeatability
# block and keeps the probability models off. Before A117 the stages after
# ``validate`` had never run on real data with the current code; four runs in
# a row had failed inside the pilots on type and schema defects, and a defect
# in a final stage would have surfaced after two or three days of extraction.
# This arm exists so that never happens again. It is not freezable, not a
# scientific release and carries no development run identity.
PRODUCTION_PREVIEW_PRESET_ID = "production_path_preview"
PRODUCTION_PREVIEW_RECORDS = 10_000
# The one stage the preview bounds on purpose: the population reference
# cohorts extract every CODE-15 and SaMi-Trop recording (345,779 + 1,631) in
# production, measured at about 18,000 per hour -- a day of work whose only
# purpose here is to show that the stage runs.
PRODUCTION_PREVIEW_EXTERNAL_RECORDS = 20_000

# Retired identifiers, resolved rather than rejected.  An old queued job or a
# manifest from a previous run must not abort a month-long study over a name.
# Resolution is safe because the preset is part of the configuration hash: the
# scope difference is detected there and forces recomputation instead of
# silently reusing results from a narrower study.
RETIRED_PRESET_IDS: dict[str, str] = {
    "recommended_clinical": STUDY_PRESET_ID,
    "final_development": STUDY_PRESET_ID,
    "expert_sensitivity": STUDY_PRESET_ID,
}


PRESETS: dict[str, AnalysisPreset] = {
    STUDY_PRESET_ID: AnalysisPreset(
        preset_id=STUDY_PRESET_ID,
        display_name_de="Vollständige Untersuchung",
        panel_ids=STUDY_PANEL_IDS,
        policy=replace(
            BASE_POLICY,
            # Measured to saturate near 500 resamples; 1000 keeps a factor of
            # two in hand.  The former 2000 was not more careful, only slower,
            # and more can still be added afterwards through refinement.
            bootstrap_iterations=1000,
            grouped_cv_repeats=5,
            # Starting budget for the open-ended combination search, not a
            # ceiling: the search continues until it is stopped.
            score_representatives_per_family=8,
            score_validation_pairs=50,
            score_validation_triples=25,
            score_triple_seed_pairs=250,
            include_broad_phenome_scan=True,
        ),
        description_de=(
            "Alle 1.753 Marker aus Core, Sentinel und Discovery Reserve, "
            "fünffach wiederholte gruppierte Score-Suche, breiter "
            "Phenome-Scan und Sensitivitätsarme. Discovery-Ergebnisse werden "
            "getrennt als explorativ berichtet. Die Verfeinerung läuft weiter, "
            "bis sie beendet wird; Locked Test und PTB Fold 10 bleiben "
            "versiegelt."
        ),
        expert_only=False,
    ),
    PREVIEW_PRESET_ID: AnalysisPreset(
        preset_id=PREVIEW_PRESET_ID,
        display_name_de="Entwicklungs-Vorschau",
        panel_ids=("primary_core",),
        policy=replace(
            BASE_POLICY,
            bootstrap_iterations=200,
            score_representatives_per_family=2,
            score_validation_pairs=1,
            score_validation_triples=1,
            score_triple_seed_pairs=5,
        ),
        description_de=(
            "Werkzeug der Entwicklung: erreicht jede Stufe in Stunden, um "
            "Fehler vor einem langen Lauf zu finden. Nicht einfrierbar und "
            "nicht für Aussagen bestimmt."
        ),
        expert_only=True,
    ),
    PRODUCTION_PREVIEW_PRESET_ID: AnalysisPreset(
        preset_id=PRODUCTION_PREVIEW_PRESET_ID,
        display_name_de="Produktionspfad-Vorschau",
        panel_ids=STUDY_PANEL_IDS,
        policy=replace(
            BASE_POLICY,
            bootstrap_iterations=100,
            grouped_cv_repeats=1,
            score_representatives_per_family=2,
            score_validation_pairs=2,
            score_validation_triples=1,
            score_triple_seed_pairs=5,
            include_broad_phenome_scan=False,
        ),
        description_de=(
            "Werkzeug der Entwicklung: der vollständige Produktionspfad -- "
            "Validierung mit adjustierten Modellen, Kalibration, "
            "Mechanismusmodellen und Score-Suche sowie alle Folgestufen -- auf "
            "einer begrenzten Stichprobe. Nicht einfrierbar und nicht für "
            "Aussagen bestimmt; die Stichprobe ist keine Population."
        ),
        expert_only=True,
    ),
}


def resolve_preset_id(preset_id: str) -> str:
    """Map a retired identifier onto the configuration now in force."""

    name = str(preset_id)
    return RETIRED_PRESET_IDS.get(name, name)


def get_analysis_preset(preset_id: str) -> AnalysisPreset:
    resolved = resolve_preset_id(preset_id)
    try:
        return PRESETS[resolved]
    except KeyError as exc:
        raise KeyError(
            f"Unknown analysis preset {preset_id!r}; "
            f"available: {', '.join(sorted(PRESETS))}"
        ) from exc


def analysis_presets(*, include_expert: bool = True) -> tuple[AnalysisPreset, ...]:
    values = PRESETS.values()
    if include_expert:
        return tuple(values)
    return tuple(item for item in values if not item.expert_only)


__all__ = [
    "BASE_POLICY",
    "PRESETS",
    "PREVIEW_PRESET_ID",
    "RETIRED_PRESET_IDS",
    "PRODUCTION_PREVIEW_PRESET_ID",
    "PRODUCTION_PREVIEW_RECORDS",
    "PRODUCTION_PREVIEW_EXTERNAL_RECORDS",
    "STUDY_PRESET_ID",
    "analysis_presets",
    "get_analysis_preset",
    "resolve_preset_id",
]
