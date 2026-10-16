"""Bounded, read-only job telemetry preview; never infer process liveness."""
import json
import math
from datetime import datetime, timezone

from ..mimic.storage import content_hash

TAIL_BYTES = 128 * 1024


def telemetry_path(config, spec):
    if spec.stage == "study-run":
        root = config.runs_root.resolve()
        run = (root / str(spec.arguments.get("run_id", "mimic-development-guided"))).resolve()
        if run == root or not run.is_relative_to(root):
            raise ValueError("Run-Pfad ausserhalb des Ergebnisverzeichnisses")
        return run / "run.jsonl"
    return config.jobs_root / "telemetry" / content_hash({"job_id": spec.job_id})[:20] / "run.jsonl"


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def telemetry_preview(config, spec, *, now=None):
    """Read at most 128 KiB; no full-log scan, outcome query or write."""
    try:
        path = telemetry_path(config, spec)
        with path.open("rb") as handle:
            size = handle.seek(0, 2)
            offset = max(0, size - TAIL_BYTES)
            handle.seek(offset)
            data = handle.read(TAIL_BYTES)
    except (OSError, ValueError) as exc:
        return f"Keine lesbare SQL-/Ressourcen-Telemetrie: {exc}"
    if offset:
        # First line may begin in the middle of a UTF-8 character or JSON object.
        data = data.partition(b"\n")[2]
    records = []
    for line in data.splitlines():
        try:
            record = json.loads(line)
        except (ValueError, UnicodeError):
            continue  # A concurrent append can leave a partial last line.
        if isinstance(record, dict) and record.get("job_id") == spec.job_id:
            records.append(record)
    if not records or not records[-1].get("attempt_id"):
        return "Keine zuordenbare Messaufnahme im begrenzten Logende."
    attempt = records[-1]["attempt_id"]
    current = [r for r in records if r.get("attempt_id") == attempt]
    samples = [r for r in current if r.get("kind") == "sample"]
    header = f"Job: {spec.job_id}\nVersuch: {attempt}\nLogende: maximal {TAIL_BYTES // 1024} KiB\n"
    if not samples:
        return header + "Noch keine Ressourcen-Messaufnahme dieses Versuchs."
    sample = samples[-1]
    stamp = str(sample.get("at", ""))
    try:
        measured = datetime.fromisoformat(stamp)
        age = ((now or datetime.now(timezone.utc)) - measured).total_seconds()
    except (TypeError, ValueError):
        age = None
    freshness = ("Zeitbezug unbekannt" if age is None or age < 0 else
                 f"{age:.0f} s alt" + (" — VERALTET" if age > 90 else ""))
    lines = [header, f"Messaufnahme: {stamp} ({freshness})",
             "Historische Messung, kein Nachweis eines aktuell lebenden Prozesses; keine ETA."]
    for key, label, factor in (("rss_gb", "Prozessbaum-RAM (GiB)", 1),
                               ("tree_cpu_core_percent", "CPU-Kernaequivalente (ueberlebende Prozesse)", .01),
                               ("tree_read_bytes_per_second", "Prozess-Lesen (MiB/s, kein SSD-Durchsatz)", 1/1024**2),
                               ("tree_write_bytes_per_second", "Prozess-Schreiben (MiB/s, kein SSD-Durchsatz)", 1/1024**2)):
        value = _number(sample.get(key))
        lines.append(f"{label}: {value * factor:.2f}" if value is not None else f"{label}: unbekannt")
    queries = sample.get("active_queries")
    if isinstance(queries, list):
        lines.append("Zum Messzeitpunkt aktive SQL-Abfragen:")
        for query in queries[:10]:
            if not isinstance(query, dict):
                continue
            elapsed = _number(query.get("elapsed_seconds"))
            duration = f"{elapsed:.0f} s" if elapsed is not None else "Dauer unbekannt"
            lines.append(f"- {str(query.get('label', 'unbekannt'))[:200]}: {duration}")
        if not queries:
            lines.append("Keine in diesem Sample registriert (kein Stillstandsbeweis).")
        if len(queries) > 10:
            lines.append(f"Weitere {len(queries)-10} Abfragen im Log.")
    else:
        lines.append("SQL-Aktivitaet in dieser Loggeneration nicht aufgezeichnet.")
    return "\n".join(lines)
