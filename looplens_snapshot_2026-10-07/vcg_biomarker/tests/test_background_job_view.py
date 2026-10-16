from types import SimpleNamespace as NS
from vcg_biomarker.gui.background_job_view import background_job_notice


def record(state='running', run='A131', stage='study-run'):
    return NS(state=NS(value=state), spec=NS(stage=stage, arguments={'run_id': run, 'preset_id': 'exhaustive_research'},
              job_id='job'), progress_message='STAFF III 31/520', heartbeat_at='2026-09-22T01:11:00+02:00', worker_pid=28092)


def test_active_runs_not_filtered_by_selection():
    text = background_job_notice([record(), record(run='A132', state='queued')])
    assert 'A131' in text and 'A132' in text and '31/520' in text
    assert '28092' in text and 'keinen Rechenfortschritt' in text


def test_completed_and_non_analysis_jobs_not_advertised_as_active():
    assert 'kein aktiver' in background_job_notice([record(state='completed'), record(stage='inventory')])


def test_missing_heartbeat_is_not_invented():
    r = record()
    r.heartbeat_at = None
    r.worker_pid = None
    text = background_job_notice([r])
    assert 'unbekannt' in text and 'nicht zugewiesen' in text


def test_paused_and_resuming_are_visible_without_claiming_compute():
    text = background_job_notice([record(state='paused'), record(run='A132', state='resuming')])
    assert 'Jobstatus: paused' in text and 'Jobstatus: resuming' in text
    assert 'Pausiert bedeutet nicht rechnend' in text
    assert 'Jobtabelle fehlen' not in text
