from pathlib import Path

import pytest

from vcg_biomarker.mimic.pilot_visual_audit import run


def test_audit_tab_identity_survives_inserted_views():
    from vcg_biomarker.mimic.pilot_visual_audit import audit_tab_indices
    names = ["pilot_results", "pilot_lab", "pilot_context", "pilot_ndat",
             "pilot_segment_matrix", "review_export", "review_import", "pilot_lead_measurements"]
    class Page:
        def __init__(self, name): self.name = name
        def objectName(self): return self.name
    class Tabs:
        def count(self): return len(names)
        def widget(self, i): return Page(names[i])
    assert dict((name, i) for i, name in audit_tab_indices(Tabs()))["lead_measurements"] == 7
    names.append("pilot_lab")
    with pytest.raises(ValueError, match="ambiguous"):
        audit_tab_indices(Tabs())


@pytest.mark.parametrize("method,case_index", [("svd", 0), ("missing", 0), ("kors", -1)])
def test_invalid_audit_selection_rejected_before_opening_workspace(tmp_path, method, case_index):
    with pytest.raises(ValueError, match="Invalid laboratory audit selection"):
        run(Path("nonexistent"), tmp_path, "light", True, method, case_index)
