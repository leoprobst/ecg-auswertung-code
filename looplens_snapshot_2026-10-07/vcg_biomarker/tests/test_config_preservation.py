"""A settings dialog may only write the settings it shows.

``config.save()`` serialises the whole project file from the in-memory object.
The dialog loads that object once, at open time, so any value edited elsewhere
in the meantime is reverted the next time somebody presses "Projekt pruefen".
Measured on 2026-08-28: ``workers`` set to 5 and ``os_memory_reserve_gb`` to
7.0 while a run was paused were back at 7 and 6.0 afterwards -- and the
reserve does not even have a field in that dialog, so nobody chose to change
it.
"""

from __future__ import annotations

import json

import pytest

from vcg_biomarker.mimic.config import MimicProjectConfig


@pytest.fixture()
def project(tmp_path):
    database_root = tmp_path / "physionet.org" / "files" / "mimiciv" / "3.1"
    (database_root / "hosp").mkdir(parents=True)
    (database_root / "icu").mkdir(parents=True)
    work_root = tmp_path / "work"
    work_root.mkdir()
    config = MimicProjectConfig.discover(database_root.parent, work_root)
    config.save()
    return config


def test_an_external_edit_survives_a_reload(project):
    """The reload is the whole mechanism; assert it directly."""

    payload = json.loads(project.project_file.read_text(encoding="utf-8"))
    payload["os_memory_reserve_gb"] = 7.0
    payload["workers"] = 5
    project.project_file.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = MimicProjectConfig.load(project.project_file)
    assert reloaded.os_memory_reserve_gb == 7.0
    assert reloaded.workers == 5


def test_saving_a_stale_object_is_what_reverts_the_file(project):
    """Names the defect, so a future change cannot quietly restore it."""

    stale = MimicProjectConfig.load(project.project_file)

    payload = json.loads(project.project_file.read_text(encoding="utf-8"))
    payload["os_memory_reserve_gb"] = 7.0
    project.project_file.write_text(json.dumps(payload), encoding="utf-8")

    stale.save()  # what the dialog used to do
    after = json.loads(project.project_file.read_text(encoding="utf-8"))
    assert after["os_memory_reserve_gb"] != 7.0, (
        "Wenn das je gleich wird, ist der Defekt behoben und dieser Test "
        "beschreibt ihn falsch"
    )

    # And the fix: re-read before saving.
    fresh = MimicProjectConfig.load(project.project_file)
    fresh.os_memory_reserve_gb = 7.0
    fresh.workers = 5
    fresh.save()
    fresh.workers = 6
    reloaded = MimicProjectConfig.load(fresh.project_file)
    reloaded.workers = 6
    reloaded.save()
    final = json.loads(project.project_file.read_text(encoding="utf-8"))
    assert final["os_memory_reserve_gb"] == 7.0
    assert final["workers"] == 6


def test_the_dialog_declares_which_settings_it_owns():
    """The list is the contract; an empty or missing one is the old bug."""

    import inspect

    from vcg_biomarker.gui.widgets import mimic_project_widget as module

    source = inspect.getsource(module)
    assert "_OWNED_SETTINGS" in source
    assert "_reload_config_preserving_external_edits" in source
    # The reload must happen before the fields are applied, or it undoes them.
    reload_at = source.index("self._reload_config_preserving_external_edits()")
    apply_at = source.index("self.config.workers = self.workers_spin.value()")
    assert reload_at < apply_at
