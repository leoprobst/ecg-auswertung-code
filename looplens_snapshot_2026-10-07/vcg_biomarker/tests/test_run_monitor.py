from audit_tools.run_monitor import PhaseRate, can_retry, current_job_id


def test_eta_uses_only_current_attempt_phase_and_never_paused_time():
    rate=PhaseRate()
    assert rate.update('attempt1:read',800,1000,0) is None
    assert rate.update('attempt1:read',900,1000,20)==20
    assert rate.update('attempt2:read',900,1000,200) is None
    assert rate.update('attempt2:merge',950,1000,220) is None
    assert rate.update('attempt2:merge',975,1000,240)==20
    assert rate.update('attempt2:merge',975,1000,500,running=False) is None
    assert rate.update('attempt2:merge',975,1000,600) is None


def test_manual_pause_cancel_and_repeat_failure_are_not_restarted():
    policy={'auto_restart_failed':True}
    job={'job_id':'job1','state':'failed','error_message':'OutOfMemoryException: test'}
    assert can_retry(job,policy,[])
    for state in ('paused','cancelled','running','pausing'):
        assert not can_retry(dict(job,state=state),policy,[])
    assert not can_retry(job,policy,['job1:OutOfMemoryException: test'])
    assert not can_retry(dict(job,error_message='Identity mismatch'),policy,[])


def test_monitor_ignores_incomplete_or_foreign_restart_record(tmp_path):
    root=tmp_path/'A155';root.mkdir()
    (root/'full_restarts.jsonl').write_text('{"run_id":"A155","job_id":"correct"}\n'
        '{"run_id":"other","job_id":"foreign"}\n{unfinished')
    assert current_job_id(root)=='correct'
