r"""S3-MC-MED Wachprozess: startet den Treiber neu, wenn er ohne FERTIG.txt/STOP.txt verschwindet. Startet NIE ohne START_SIGNAL.txt.

- liest tasklist mit encoding="cp850", errors="replace" (Prozess gilt als lebend, wenn die PID als CSV-Feld erscheint)
- Neustart per WMI (start_wmi_s3.ps1), hoechstens 5 Neustarts je 60 Minuten, danach STOP.txt (Entscheidung K)
- beendet nie fremde Prozesse; beendet sich bei FERTIG.txt oder STOP.txt
Aufruf (per WMI gestartet): <local path> -I s3_wacht.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, os, json, time, subprocess, datetime as dt
from pathlib import Path

sys.dont_write_bytecode = True
HIER = Path(__file__).resolve().parent
ST = Path(str(data_path("s3.pipeline_dir")))
PY = str(data_path("s3.python_exe"))
PIDFILE, WPID = ST / 'treiber.pid', ST / 'wacht.pid'
STOP, FERTIG, SIGNAL, LOG = ST / 'STOP.txt', ST / 'FERTIG.txt', ST / 'START_SIGNAL.txt', ST / 'wacht.log'
INTERVALL = 60


def jetzt():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(*a):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(jetzt() + ' ' + ' '.join(str(x) for x in a) + '\n')


def lebt(pid):
    try:
        out = subprocess.run(['tasklist', '/FI', f'PID eq {int(pid)}', '/FO', 'CSV', '/NH'], capture_output=True).stdout.decode('cp850', errors='replace')
        return f'"{int(pid)}"' in out and 'python' in out.lower()
    except Exception as ex:
        log('tasklist-Fehler', ex)
        return True          # im Zweifel nicht neu starten (zwei Treiber auf einer Datenbank waeren schlimmer)


def starte_treiber():
    cmd = f'cmd.exe /c ""{PY}" -I "{HIER / "s3_gesamtlauf_treiber.py"}" lauf >> "{ST / "treiber.log"}" 2>&1"'
    r = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(HIER / 'start_wmi_s3.ps1'), '-CommandLine', cmd], capture_output=True)
    out = r.stdout.decode('cp850', errors='replace').strip()
    log('Neustart ausgeloest, cmd-PID', out, r.stderr.decode('cp850', errors='replace').strip()[:200])
    return r.returncode == 0


def main():
    ST.mkdir(parents=True, exist_ok=True)
    WPID.write_text(str(os.getpid()), encoding='utf-8')
    log('Wachprozess gestartet, PID', os.getpid())
    starts = []
    while True:
        if FERTIG.exists():
            log('FERTIG.txt: Wachprozess endet')
            return 0
        if STOP.exists():
            log('STOP.txt: Wachprozess endet:', STOP.read_text(encoding='utf-8', errors='replace')[:200].strip())
            return 3
        if not SIGNAL.exists():
            time.sleep(INTERVALL)          # ohne Startsignal wird nichts gestartet
            continue
        pid = None
        try:
            pid = int(PIDFILE.read_text().strip())
        except Exception:
            pass
        if pid is None or not lebt(pid):
            jetzt_t = time.time()
            starts = [t for t in starts if jetzt_t - t < 3600]
            if len(starts) >= 5:
                STOP.write_text(f'{jetzt()} Wachprozess: mehr als 5 Neustarts in 60 Minuten; Ursache im treiber.log pruefen.\n', encoding='utf-8')
                log('zu viele Neustarts, STOP gesetzt')
                return 3
            time.sleep(15)          # Auslaufen / Dateisperre der Datenbank abwarten und PID erneut lesen
            try:
                pid2 = int(PIDFILE.read_text().strip())
            except Exception:
                pid2 = None
            if pid2 is not None and lebt(pid2):
                continue
            if FERTIG.exists() or STOP.exists():
                continue
            if starte_treiber():
                starts.append(time.time())
                try:
                    wj = ST / 'wacht.json'
                    z = json.loads(wj.read_text(encoding='utf-8')) if wj.exists() else {}
                    z['neustarts'] = z.get('neustarts', 0) + 1
                    z['letzter'] = jetzt()
                    wj.write_text(json.dumps(z, indent=1), encoding='utf-8')
                except Exception as ex:
                    log('Zaehler nicht aktualisiert', ex)
                time.sleep(120)
        time.sleep(INTERVALL)


if __name__ == '__main__':
    sys.exit(main())
