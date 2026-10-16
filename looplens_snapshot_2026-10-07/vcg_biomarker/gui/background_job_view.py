"""Project-wide activity notice independent of the selected scientific run."""


def background_job_notice(records):
    lines = []
    for record in records:
        state = getattr(record.state, 'value', record.state)
        if record.spec.stage != 'study-run' or state not in {'running', 'queued', 'pausing', 'cancelling', 'paused', 'resuming'}:
            continue
        arguments = record.spec.arguments
        run = arguments.get('run_id') or record.spec.job_id
        lines.append(f"{run} · {arguments.get('preset_id', 'Profil unbekannt')} · Jobstatus: {state}"
                     f"\n{record.progress_message or 'Noch keine Fortschrittsmeldung'}"
                     f"\nLetzte Jobmeldung: {record.heartbeat_at or 'unbekannt'} · Worker-PID: {record.worker_pid or 'nicht zugewiesen'}")
    if not lines:
        return 'Projektweit kein aktiver Analysejob gemeldet.'
    return ('Hintergrundanalysen – unabhängig von der unten gewählten Analyse:\n' + '\n\n'.join(lines)
            + '\nDer gespeicherte Jobstatus allein belegt keinen Rechenfortschritt. '
              'Ein eingefrorener Hintergrundlauf kann von der aktuellen Entwicklungsversion abweichen '
              'und wird dann in der Jobtabelle als eingefrorener Lauf nur zur Anzeige geführt. '
              'Pausiert bedeutet nicht rechnend. Keinen zweiten Lauf starten.')
