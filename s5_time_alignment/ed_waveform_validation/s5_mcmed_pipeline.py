"""S5-MC-MED Pipeline-Treiber (nicht Teil der gebundenen Auswertung): fuehrt die Schritte der Spezifikation v1.6/v1.7 mit dem gebundenen Skript v17 aus.

Der Treiber veraendert weder das Skript noch die Konfiguration ausser den ERLAUBTEN Pilot-Aenderungen (rr_cv_max, abdeckung_A_min_s innerhalb der Spannen) und setzt zur Laufzeit
die Pfade der Spezifikation (v1.7) und der Basiskette; alle Start- und Bindungspruefungen des Skripts bleiben aktiv.

Schritte (jeder wiederaufnehmbar; Checkpoints: je Besuch eine Transaktion in der Extraktion, Zustandsdatei, Protokoll- und Ausgabedateien):
  1. Val-Extraktion und technische Pruefung je Erkennerkandidat (vorab festgelegte Liste KANDIDATEN), Protokollierung nur auf der vollstaendigen Val-Grundgesamtheit
  2. Einfrieren des nach technischer Guete besten Kandidaten (Uebereinstimmung >= 95 %, Ausbeute >= 0,80); sonst Halt (Messkette v2, Entscheidung K)
  3. Pilot (Val), 4. Train-Extraktion, 5. Bestaetigung (Train), 6. Ergebnisbericht (nur Aggregate)
Aufrufe (Python: <local path> -I):
  s5_mcmed_pipeline.py binden     Bindung (spec_lock_v16.json) erzeugen, einmalig
  s5_mcmed_pipeline.py lauf       Treiber (durch den Wachprozess oder per WMI gestartet)
  s5_mcmed_pipeline.py stand      Zustand ausgeben
Regeln: Testsplits werden nie geoeffnet (das Skript kennt nur train und val); keine Kennungen in Ausgaben; keine Loeschbefehle; MC-MED nur lokal.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, os, json, time, subprocess, importlib.util, datetime as dt, traceback, re, hashlib
from pathlib import Path

sys.dont_write_bytecode = True
HIER = Path(__file__).resolve().parent
AUSW = Path(str(data_path("s5.auswertung")))
PRIV = Path(str(data_path("s5.analyse_mc_med_s5")))
ST = PRIV / 'pipeline'
ZW = AUSW / 'koordination/abgaben/S5_MCMED_ZWISCHENSTAND.md'
SKRIPT = HIER / 's5_mcmed_auswertung_v17.py'
SPEC17 = AUSW / 'nachrechnung/SPEZIFIKATION_S5_MCMED_v1_7.md'
SPEC16 = AUSW / 'nachrechnung/SPEZIFIKATION_S5_MCMED_v1_6.md'
SPEC15 = AUSW / 'nachrechnung/SPEZIFIKATION_S5_MCMED_v1_5.md'
RAM = Path(str(data_path("s5.scanner_v10_1_tools_k_coord_ram_reservierung_py")))
PY = str(data_path("s5.venvs_auswertung_scripts_python_exe"))
STATE = ST / 'state.json'
STOP = ST / 'STOP.txt'
FERTIG = ST / 'FERTIG.txt'
PIDFILE = ST / 'treiber.pid'

# Vorab festgelegte Kandidatenliste (Spannen aus v1 5.1: rr_cv_max 0,3 bis 0,5; abdeckung_A_min_s 45 bis 54). Reihenfolge = Protokollreihenfolge; die Wahl erfolgt allein nach technischer Guete.
KANDIDATEN = [
    {'name': 'K1_cv0.50_abd48', 'rr_cv_max': 0.5, 'abdeckung_A_min_s': 48.0},
    {'name': 'K2_cv0.40_abd48', 'rr_cv_max': 0.4, 'abdeckung_A_min_s': 48.0},
    {'name': 'K3_cv0.30_abd48', 'rr_cv_max': 0.3, 'abdeckung_A_min_s': 48.0},
    {'name': 'K4_cv0.50_abd45', 'rr_cv_max': 0.5, 'abdeckung_A_min_s': 45.0},
    {'name': 'K5_cv0.50_abd54', 'rr_cv_max': 0.5, 'abdeckung_A_min_s': 54.0},
]


class Angehalten(Exception):
    pass


def jetzt():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(*a):
    print(jetzt(), *a, flush=True)


def lade(override=None):
    """Laedt das gebundene Skript v17 und setzt NUR Pfade der Spezifikation (v1.7, Basis v1.6 im letzten Slot) und erlaubte Konfigurationswerte."""
    sp = importlib.util.spec_from_file_location('s5v17', SKRIPT)
    A = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(A)
    A.SPEC_PATH = SPEC17
    A.BASIS5_SPEC_PATH = SPEC16
    for k, v in (override or {}).items():
        assert k in ('rr_cv_max', 'abdeckung_A_min_s'), k
        A.CFG[k] = v
    return A


def zustand():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding='utf-8'))
    return {'schritte': {}, 'neustarts': 0, 'fortschritt': {}}


def sichere(z):
    ST.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(z, indent=1, default=str), encoding='utf-8')
    os.replace(tmp, STATE)


def lebt(pid):
    try:
        out = subprocess.run(['tasklist', '/FI', f'PID eq {int(pid)}', '/FO', 'CSV', '/NH'], capture_output=True).stdout.decode('cp850', errors='replace')
        return f'"{int(pid)}"' in out
    except Exception:
        return False


# ------------------------------------------------------------------ Zwischenstand
class Anzeige:
    def __init__(self):
        self.z = zustand()
        self.letzte = 0.0
        self.rss_max = 0.0

    def schritt(self, name, status, info=''):
        s = self.z['schritte'].setdefault(name, {})
        if status == 'laeuft' and 'start' not in s:
            s['start'] = jetzt()
        s['status'] = status
        s['info'] = info
        if status in ('fertig', 'uebersprungen', 'halt'):
            s['ende'] = jetzt()
        self.z['aktuell'] = name if status == 'laeuft' else self.z.get('aktuell')
        sichere(self.z)
        self.schreibe(erzwingen=True)

    def fortschritt(self, d):
        try:
            import psutil
            rss = psutil.Process().memory_info().rss / 2 ** 30
            self.rss_max = max(self.rss_max, rss)
            self.z['rss_max_gb'] = round(self.rss_max, 2)
        except Exception:
            pass
        f = self.z.setdefault('fortschritt', {})
        f.update(d)
        f['zeit'] = jetzt()
        if 'begin_t' not in f or f.get('begin_name') != self.z.get('aktuell'):
            f['begin_t'] = time.time()
            f['begin_name'] = self.z.get('aktuell')
            f['begin_fertig'] = d.get('besuche_fertig', 0)
        done = d.get('besuche_fertig', 0) - f['begin_fertig']
        dt_s = max(1.0, time.time() - f['begin_t'])
        if done > 0:
            rate = done / dt_s
            f['besuche_pro_s'] = round(rate, 3)
            f['rest_h'] = round((d.get('von', 0) - d.get('besuche_fertig', 0)) / rate / 3600, 2)
        sichere(self.z)
        self.schreibe()

    def schreibe(self, erzwingen=False):
        if not erzwingen and time.time() - self.letzte < 240:
            return
        self.letzte = time.time()
        z = self.z
        wacht = ''
        try:
            z['neustarts'] = json.loads((ST / 'wacht.json').read_text(encoding='utf-8')).get('neustarts', 0)
        except Exception:
            pass
        wp = ST / 'wacht.pid'
        if wp.exists():
            try:
                w = int(wp.read_text().strip())
                wacht = f'PID {w} ({"lebt" if lebt(w) else "TOT"})'
            except Exception:
                wacht = 'unbekannt'
        zeilen = ['# S5-MC-MED Pipeline: Zwischenstand', '',
                  f'Stand {jetzt()} (laufend fortgeschrieben, nur Aggregate, keine Kennungen). Treiber PID {os.getpid()}, Wachprozess {wacht}, Neustarts {z.get("neustarts", 0)}, Speicherspitze des Treibers {z.get("rss_max_gb", "?")} GB.',
                  '', 'Bindung: Spezifikation v1.7 mit Skript v17 (Details `nachrechnung/S5_MCMED_BINDUNG.md`). Zustand und Logs: `<local path>` (`state.json`, `treiber.log`, `wacht.log`).', '',
                  '## Schritte', '', '| Schritt | Status | Beginn | Ende | Hinweis |', '|---|---|---|---|---|']
        for n, s in z['schritte'].items():
            zeilen.append(f"| {n} | {s.get('status')} | {s.get('start', '')} | {s.get('ende', '')} | {s.get('info', '')} |")
        f = z.get('fortschritt', {})
        if f:
            zeilen += ['', '## Fortschritt des laufenden Schritts', '',
                       f"{z.get('aktuell')}: {f.get('besuche_fertig', 0)} von {f.get('von', '?')} Besuchen, Lesefehler {f.get('lesefehler', 0)}, Besuche ohne Einheiten {f.get('ohne_einheiten', 0)}; "
                       f"Rate {f.get('besuche_pro_s', '?')} Besuche/s, geschaetzte Restdauer {f.get('rest_h', '?')} h (Stand {f.get('zeit', '')})."]
        if z.get('kandidaten'):
            zeilen += ['', '## Erkennerkandidaten (technische Guete auf der vollstaendigen Val-Grundgesamtheit)', '',
                       '| Kandidat | Uebereinstimmung innerhalb 5/min | Ausbeute | Kriterium | gewaehlt |', '|---|---:|---:|---|---|']
            for k in z['kandidaten']:
                zeilen.append(f"| {k['name']} | {k.get('anteil')} | {k.get('ausbeute')} | {k.get('erfuellt')} | {k.get('gewaehlt', '')} |")
        if z.get('hinweis'):
            zeilen += ['', '## Hinweis', '', z['hinweis']]
        zeilen += ['', '## Wiederaufnahme', '',
                   'Der Wachprozess startet den Treiber bei Absturz neu (nicht bei `STOP.txt` oder `FERTIG.txt`). Manuell: `STOP.txt` pruefen und die Ursache im Log beheben, dann '
                   '`s5_mcmed_pipeline.py lauf` per WMI (`start_wmi.ps1`) starten; alle Schritte setzen am Checkpoint fort (Extraktion je Besuch transaktional).']
        ZW.parent.mkdir(parents=True, exist_ok=True)
        tmp = ZW.with_suffix('.tmp')
        tmp.write_text('\n'.join(zeilen) + '\n', encoding='utf-8')
        os.replace(tmp, ZW)


# ------------------------------------------------------------------ RAM-Reservierung
def ram_aufruf(*args):
    r = subprocess.run([PY, '-I', str(RAM), *args], capture_output=True)
    return r.returncode, r.stdout.decode('cp850', errors='replace') + r.stderr.decode('cp850', errors='replace')


class Reservierung:
    """Reserviert vor dem Start (wartet ohne Start bei 'warten'), gibt am Ende frei. Eintraege verfallen ohnehin mit dem Ende des Treiber-Prozesses."""

    def __init__(self, gb, zweck, stunden):
        self.gb, self.zweck, self.stunden = gb, zweck, stunden
        self.rid = None

    def __enter__(self):
        while True:
            if STOP.exists():
                raise Angehalten('STOP.txt beim Warten auf RAM')
            rc, out = ram_aufruf('pruefen', '--gb', str(self.gb), '--seite', 'analyse')
            if rc == 0:
                rc, out = ram_aufruf('eintragen', '--gb', str(self.gb), '--seite', 'analyse', '--sitzung', 'S5-MC-MED Pipeline', '--pid', str(os.getpid()),
                                     '--stunden', str(self.stunden), '--zweck', self.zweck)
                m = re.search(r'eingetragen (\w+)', out)
                if rc == 0 and m:
                    self.rid = m.group(1)
                    log('RAM reserviert', self.gb, 'GB', self.rid)
                    return self
            log('RAM-Reservierung wartet:', out.strip()[:150])
            time.sleep(180)

    def __exit__(self, *a):
        if self.rid:
            ram_aufruf('freigeben', '--id', self.rid)
            log('RAM freigegeben', self.rid)


def warte_speicher(min_gb=1.6):
    import psutil
    while psutil.virtual_memory().available < min_gb * 2 ** 30:
        if STOP.exists():
            raise Angehalten('STOP.txt beim Warten auf Speicher')
        log(f'weniger als {min_gb} GB verfuegbar, warte')
        time.sleep(120)


def mit_speicherwarten(fn, *a, **k):
    """Ruft fn auf; bei 'weniger als 1,5 GB frei' (SystemExit des Skripts) wird gewartet und wiederholt, jede andere Verweigerung ist fatal."""
    while True:
        warte_speicher()
        try:
            return fn(*a, **k)
        except SystemExit as ex:
            if 'weniger als 1,5 GB' in str(ex):
                time.sleep(120)
                continue
            raise


# ------------------------------------------------------------------ Schritte
def hake(A, anz):
    """Fortschrittsmeldungen der Extraktion (JSON-Zeilen) abfangen; an diesen Stellen (nach dem Commit) darf angehalten werden."""
    orig = print

    def hook(*args, **kw):
        orig(*args, **kw)
        if args and isinstance(args[0], str) and args[0].startswith('{'):
            try:
                d = json.loads(args[0])
            except ValueError:
                return
            if 'besuche_fertig' in d:
                anz.fortschritt(d)
                if STOP.exists():
                    raise Angehalten('STOP.txt')
    A.print = hook


def extraktion(A, split, anz, name):
    anz.schritt(name, 'laeuft', f'Erkennerversion {A.erkenner_version()}')
    hake(A, anz)
    with Reservierung(2.0, f'S5-MC-MED Extraktion {split}', 14):
        res = mit_speicherwarten(A.modus_extrahieren, split)
    anz.schritt(name, 'fertig', f"{res['besuche']} Besuche neu, Lesefehler {res['lesefehler']}, ohne Einheiten {res['ohne_einheiten']}")
    return res


def kandidat_vorhanden(A, gg_sha):
    ev, ch, cg = A.erkenner_version(), A.code_hash(), A.cfg_hash()
    for e in A.lies_protokoll():
        if e.get('ev') == ev and e.get('vollstaendig') is True and e.get('code_sha256') == ch and e.get('cfg_sha256') == cg and (e.get('grundgesamtheit') or {}).get('sha256') == gg_sha:
            return e
    return None


def val_gg(A):
    c = A.Z.verbinden()
    try:
        A.tabellen_anlegen(c)
        return A.val_grundgesamtheit(c)
    finally:
        c.close()


def schritt_kandidaten(anz):
    z = anz.z
    z['kandidaten'] = z.get('kandidaten', [])
    for kd in KANDIDATEN:
        name = 'Kandidat ' + kd['name']
        A = lade({'rr_cv_max': kd['rr_cv_max'], 'abdeckung_A_min_s': kd['abdeckung_A_min_s']})
        gg = val_gg(A)
        e = kandidat_vorhanden(A, gg['sha256'])
        if e is None:
            extraktion(A, 'val', anz, name + ' Extraktion')
            anz.schritt(name + ' Pruefung', 'laeuft', 'technische Guete gegen Monitor-HF')
            mit_speicherwarten(A.modus_erkenner)
            e = kandidat_vorhanden(A, gg['sha256'])
            anz.schritt(name + ' Pruefung', 'fertig', '')
        else:
            anz.schritt(name, 'uebersprungen', 'bereits im Protokoll')
        k = e['kriterium']
        eintrag = {'name': kd['name'], 'ev': e['ev'][:8], 'anteil': round(k['anteil_innerhalb'], 4) if k.get('anteil_innerhalb') is not None else None,
                   'ausbeute': round(k['ausbeute'], 4) if k.get('ausbeute') is not None else None, 'erfuellt': k['kriterium_erfuellt']}
        z['kandidaten'] = [x for x in z['kandidaten'] if x['name'] != kd['name']] + [eintrag]
        sichere(z)
        anz.schreibe(erzwingen=True)


def schritt_einfrieren(anz):
    A0 = lade()
    if A0.FREEZE.exists():
        anz.schritt('Einfrieren', 'uebersprungen', 'freeze_v16.json vorhanden')
        return json.loads(A0.FREEZE.read_text(encoding='utf-8'))
    gg = val_gg(A0)
    best = A0.waehle_erkenner_eintrag(A0.lies_protokoll(), gg['sha256'])
    if best is None:
        anz.z['hinweis'] = 'HALT: Kein Erkennerkandidat erfuellt Uebereinstimmung (>= 95 % innerhalb 5/min) und Mindestausbeute (>= 0,80). Entscheidung K: Messkette v2 (AE-311 Punkt 4).'
        anz.schritt('Einfrieren', 'halt', 'kein Kandidat erfuellt das Kriterium')
        STOP.write_text('Kein Erkennerkandidat erfuellt das Kriterium; Entscheidung K (Messkette v2).\n', encoding='utf-8')
        raise Angehalten('kein Kandidat erfuellt das Kriterium')
    p = best['extraktionsparameter']
    A = lade({'rr_cv_max': p['rr_cv_max'], 'abdeckung_A_min_s': p['abdeckung_A_min_s']})
    anz.schritt('Einfrieren', 'laeuft', f"bester Kandidat {best['ev'][:8]}")
    A.modus_einfrieren('Pipeline: bester Kandidat nach technischer Guete (Uebereinstimmung, Ausbeute) auf der vollstaendigen Val-Grundgesamtheit')
    for k in anz.z.get('kandidaten', []):
        k['gewaehlt'] = 'ja' if best['ev'].startswith(k['ev']) else ''
    anz.schritt('Einfrieren', 'fertig', f"Erkennerversion {best['ev'][:8]}: rr_cv_max {p['rr_cv_max']}, abdeckung_A_min_s {p['abdeckung_A_min_s']}")
    return json.loads(A.FREEZE.read_text(encoding='utf-8'))


def frozen_override(fz):
    return {'rr_cv_max': fz['cfg']['rr_cv_max'], 'abdeckung_A_min_s': fz['cfg']['abdeckung_A_min_s']}


def auswertung(split, name, anz, fz, gb):
    A = lade(frozen_override(fz))
    ziel = A.AUS / f'{name}.json'
    if ziel.exists():
        anz.schritt(f'Auswertung {name}', 'uebersprungen', f'{ziel.name} vorhanden')
        return
    anz.schritt(f'Auswertung {name}', 'laeuft', f'Erkennerversion {A.erkenner_version()}')
    with Reservierung(gb, f'S5-MC-MED Auswertung {name}', 12):
        mit_speicherwarten(A.modus_auswerten, split, name)
    anz.schritt(f'Auswertung {name}', 'fertig', ziel.name)


def lauf():
    ST.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()), encoding='utf-8')
    anz = Anzeige()
    log('Treiber gestartet, PID', os.getpid())
    try:
        if FERTIG.exists():
            log('FERTIG.txt vorhanden, nichts zu tun')
            return 0
        if STOP.exists():
            log('STOP.txt vorhanden:', STOP.read_text(encoding='utf-8', errors='replace')[:200])
            return 3
        A0 = lade()
        fz = None
        if not A0.FREEZE.exists():
            schritt_kandidaten(anz)
            fz = schritt_einfrieren(anz)
        else:
            fz = json.loads(A0.FREEZE.read_text(encoding='utf-8'))
            anz.schritt('Einfrieren', 'uebersprungen', 'freeze_v16.json vorhanden')
        auswertung('val', 'pilot', anz, fz, 2.0)
        Atr = lade(frozen_override(fz))
        extraktion(Atr, 'train', anz, 'Train-Extraktion')
        auswertung('train', 'bestaetigung', anz, fz, 4.0)
        anz.schritt('Ergebnisbericht', 'laeuft', '')
        try:
            spec = importlib.util.spec_from_file_location('s5ber', HIER / 's5_mcmed_bericht.py')
            B = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(B)
            B.schreibe_bericht()
            anz.schritt('Ergebnisbericht', 'fertig', 'nachrechnung/S5_MCMED_ERGEBNIS.md')
        except Exception:
            log('Bericht fehlgeschlagen:', traceback.format_exc())
            anz.schritt('Ergebnisbericht', 'halt', 'Berichtsskript fehlgeschlagen (siehe Log); Auswertung ist fertig')
        FERTIG.write_text(jetzt(), encoding='utf-8')
        anz.z['hinweis'] = 'FERTIG: alle Schritte abgeschlossen.'
        sichere(anz.z)
        anz.schreibe(erzwingen=True)
        return 0
    except Angehalten as ex:
        log('angehalten:', ex)
        anz.z['hinweis'] = anz.z.get('hinweis') or f'Angehalten: {ex}'
        sichere(anz.z)
        anz.schreibe(erzwingen=True)
        return 3
    except SystemExit as ex:
        # Verweigerung durch die Startpruefungen oder Regeln des Skripts: fatal, kein Neustart
        STOP.write_text(f'{jetzt()} Verweigerung durch das Skript: {ex}\n', encoding='utf-8')
        log('FATAL (Verweigerung):', ex)
        anz.z['hinweis'] = f'HALT durch Verweigerung des Skripts: {str(ex)[:300]}'
        sichere(anz.z)
        anz.schreibe(erzwingen=True)
        return 4
    except Exception:
        log('Absturz:', traceback.format_exc())
        anz.z['hinweis'] = 'Absturz des Treibers (siehe treiber.log); der Wachprozess startet neu.'
        sichere(anz.z)
        anz.schreibe(erzwingen=True)
        return 1


def binden():
    A = lade()
    # Vorabpruefung der Kette: v1.6 nennt den Hash von v1.5
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(SPEC15) in SPEC16.read_text(encoding='utf-8'), 'v1.5-Hash steht nicht in v1.6'
    assert sha(SPEC16) in SPEC17.read_text(encoding='utf-8'), 'v1.6-Hash steht nicht in v1.7'
    assert sha(SKRIPT) in SPEC17.read_text(encoding='utf-8'), 'Skript-v17-Hash steht nicht in v1.7'
    A.modus_sperren()
    lk = json.loads(A.LOCK.read_text(encoding='utf-8'))
    print(json.dumps({k: v for k, v in lk.items() if k != 'cfg'}, indent=1))


if __name__ == '__main__':
    m = sys.argv[1] if len(sys.argv) > 1 else ''
    if m == 'binden':
        binden()
    elif m == 'lauf':
        sys.exit(lauf())
    elif m == 'stand':
        print(json.dumps(zustand(), indent=1, default=str))
    else:
        print(__doc__)
        sys.exit(2)
