"""S3/N3-M MC-MED Treiber (NICHT Teil der gebundenen Auswertung): Bindung, Annotationsvorbereitung und der losgeloeste Gesamtlauf des gebundenen Skripts s3_auswertung_v2_2.py.

Der Treiber aendert weder Skript noch Konfiguration noch Spezifikation. Er ruft nur die gebundenen Funktionen auf (modus_sperren, pruefe_start, gesamtlauf, laden, messe_entnahme,
annotation_stichprobe, annotation_bild, annotation_auswertung). Alle Start- und Bindungspruefungen des Skripts bleiben aktiv.

Aufrufe (Python: <local path> -I):
  binden              Bindung (spec_lock_s3_v2_2.json) ueber modus_sperren erzeugen, einmalig; ergaenzt den Verweis auf SHA256SUMS.txt der MC-MED-Version
  pruefen             Bereitschaft ausgeben (Bindung, SHA256SUMS, Signale, Annotation, RAM); veraendert nichts
  annotation          (Alternative, nicht noetig, wenn die HTML-Annotation im Ordner annotation existiert) Annotationsbilder vorbereiten; verlangt START_ANNOTATION.txt
  annotation_auswerten  Anwender-CSV gegen die Pipelinewerte auswerten (beschreibend, ohne Tor)
  start               START_SIGNAL.txt schreiben (nur nach Entscheidung K) und den Wachprozess losgeloest starten
  lauf                Treiber des Gesamtlaufs (durch den Wachprozess gestartet; verlangt START_SIGNAL.txt und die Annotation oder deren Verzicht)
  stand               Zustand ausgeben
  selbsttest          Tests an Kunstdaten in einem Arbeitsordner (kein Zugriff auf MC-MED)
Regeln: der Gesamtlauf startet NIE ohne START_SIGNAL.txt; Annotation vor dem Gesamtlauf (damit sie kaliumblind bleibt) oder ANNOTATION_VERZICHT.txt; keine Kennungen in Ausgaben;
keine Loeschbefehle; MC-MED nur lokal; RAM-Reservierung ueber ram_reservierung.py (Seite analyse; Deckel Tag 7 GB, Nacht 4 GB: Standard 4 GB).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, os, json, time, subprocess, importlib.util, datetime as dt, traceback, re, hashlib, tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HIER = Path(__file__).resolve().parent
SKRIPT = HIER / 's3_auswertung_v2_2.py'
RAM = Path(str(data_path("s3.ram_reservation_py")))
PY = str(data_path("s3.python_exe"))
DATEN = Path(str(data_path("s3.mcmed_data_dir")))
PRIVAT = Path(str(data_path("s3.work_dir")))


class Angehalten(Exception):
    pass


class Konfig:
    """Pfade und Schalter. Standard = Produktion; Tests setzen eigene Ordner und ram=False."""

    def __init__(self, daten=DATEN, privat=PRIVAT, welle=None, lock=None, spec=None, ram=True, gb=4.0, arbeiter=3, stunden=16, n_annot=100, n_annot_pool=300):
        self.daten = Path(daten)
        self.privat = Path(privat)
        self.welle = Path(welle) if welle else self.daten / 'waveforms'
        self.lock = Path(lock) if lock else None
        self.spec = Path(spec) if spec else None
        self.ram, self.gb, self.arbeiter, self.stunden = ram, gb, arbeiter, stunden
        self.n_annot, self.n_annot_pool = n_annot, n_annot_pool
        self.st = self.privat / 'pipeline'
        self.ann = self.privat / 'annotation'

    @property
    def stop(self):
        return self.st / 'STOP.txt'

    @property
    def fertig(self):
        return self.st / 'FERTIG.txt'

    @property
    def signal(self):
        return self.st / 'START_SIGNAL.txt'

    @property
    def signal_annot(self):
        return self.st / 'START_ANNOTATION.txt'

    @property
    def verzicht(self):
        return self.st / 'ANNOTATION_VERZICHT.txt'

    @property
    def anwender_csv(self):
        return self.ann / 'annotation_anwender.csv'

    @property
    def anwender_csvs(self):
        """Ergebnisdateien der Anwenderannotation: annotation_anwender.csv (Treiberpfad) oder der CSV-Export der HTML-Annotation (referenzannotation_s3_ergebnisse_*.csv)."""
        return ([self.anwender_csv] if self.anwender_csv.exists() else []) + sorted(self.ann.glob('referenzannotation_s3_ergebnisse_*.csv'))

    @property
    def state(self):
        return self.st / 'state.json'

    @property
    def pidfile(self):
        return self.st / 'treiber.pid'


def jetzt():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(*a):
    print(jetzt(), *a, flush=True)


def lade_skript(k=None):
    """Importiert das gebundene Skript ueber seinen Modulnamen von seinem Pfad (dieselbe Datei, deren Hash gebunden ist; als importierbares Modul, damit Worker-Prozesse es finden)
    und setzt NUR Pfade (Spezifikation, Sperrdatei, Daten) fuer Tests."""
    if str(HIER) not in sys.path:
        sys.path.insert(0, str(HIER))
    A = importlib.import_module(SKRIPT.stem)
    assert Path(A.__file__).resolve() == SKRIPT.resolve()
    if k is not None:
        if k.lock is not None:
            A.LOCK = k.lock
        if k.spec is not None:
            A.SPEC_PATH = k.spec
        A.PRIV = k.privat
    return A


def lock_pfad(A, k):
    return Path(k.lock) if k.lock else A.LOCK


def zustand(k):
    if k.state.exists():
        return json.loads(k.state.read_text(encoding='utf-8'))
    return {'schritte': {}, 'neustarts': 0}


def sichere(k, z):
    k.st.mkdir(parents=True, exist_ok=True)
    tmp = k.state.with_suffix('.tmp')
    tmp.write_text(json.dumps(z, indent=1, default=str), encoding='utf-8')
    os.replace(tmp, k.state)


# ------------------------------------------------------------------ RAM
def ram_aufruf(*args):
    r = subprocess.run([PY, '-I', str(RAM), *args], capture_output=True)
    return r.returncode, r.stdout.decode('cp850', errors='replace') + r.stderr.decode('cp850', errors='replace')


class Reservierung:
    """Reserviert vor dem Start (wartet ohne Start bei 'warten') und gibt am Ende frei; Eintraege verfallen mit dem Ende des Treiberprozesses."""

    def __init__(self, k, zweck):
        self.k, self.zweck, self.rid = k, zweck, None

    def __enter__(self):
        if not self.k.ram:
            return self
        while True:
            if self.k.stop.exists():
                raise Angehalten('STOP.txt beim Warten auf RAM')
            rc, out = ram_aufruf('pruefen', '--gb', str(self.k.gb), '--seite', 'analyse')
            if rc == 0:
                rc, out = ram_aufruf('eintragen', '--gb', str(self.k.gb), '--seite', 'analyse', '--sitzung', 'S3-MC-MED Gesamtlauf', '--pid', str(os.getpid()),
                                     '--stunden', str(self.k.stunden), '--zweck', self.zweck)
                m = re.search(r'eingetragen (\w+)', out)
                if rc == 0 and m:
                    self.rid = m.group(1)
                    log('RAM reserviert', self.k.gb, 'GB', self.rid)
                    return self
            log('RAM-Reservierung wartet:', out.strip()[:150])
            time.sleep(180)

    def __exit__(self, *a):
        if self.rid:
            ram_aufruf('freigeben', '--id', self.rid)
            log('RAM freigegeben', self.rid)


def warte_speicher(k, min_gb=1.6):
    if not k.ram:
        return
    import psutil
    while psutil.virtual_memory().available < min_gb * 2 ** 30:
        if k.stop.exists():
            raise Angehalten('STOP.txt beim Warten auf Speicher')
        log(f'weniger als {min_gb} GB verfuegbar, warte')
        time.sleep(120)


# ------------------------------------------------------------------ Bindung
def eingaben_verweis(A, k):
    """Verweis auf SHA256SUMS.txt der MC-MED-Version (Hash der Datei und die dort eingetragenen Hashes der Eingabedateien; es werden keine Dateiinhalte der Splits gelesen)."""
    sums = k.daten.parent / 'SHA256SUMS.txt'
    if not sums.exists():
        raise SystemExit('SHA256SUMS.txt fehlt: keine Bindung')
    soll = {}
    for line in sums.read_text(encoding='utf-8', errors='replace').splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            soll[parts[1].strip().lstrip('*')] = parts[0].lower()
    fehl = [f for f in A.EINGABEN if f'data/{f}' not in soll]
    if fehl:
        raise SystemExit('nicht in SHA256SUMS: ' + ', '.join(fehl))
    return {'datensatz': 'MC-MED 1.0.1', 'sha256sums_pfad': str(sums), 'sha256sums_sha256': hashlib.sha256(sums.read_bytes()).hexdigest(),
            'eingabedateien_laut_sha256sums': {f: soll[f'data/{f}'] for f in A.EINGABEN},
            'hinweis': 'Der Gesamtlauf prueft vor dem Laden alle Eingaben (einschliesslich Testsplit) gegen diese Datei; Abbruch bei Abweichung oder fehlender Datei.'}


def binden(k=None):
    k = k or Konfig()
    A = lade_skript(k)
    lk = lock_pfad(A, k)
    if lk.exists():
        raise SystemExit('Bindung existiert bereits: ' + str(lk))
    verweis = eingaben_verweis(A, k)                  # vorab pruefen, bevor irgendetwas geschrieben wird
    A.modus_sperren()                                 # das vorgesehene Verfahren des gebundenen Skripts
    d = json.loads(lk.read_text(encoding='utf-8'))
    d['eingaben'] = verweis                           # Verweis unmittelbar nach der Bindung; pruefe_start liest nur 'hashes'
    lk.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding='utf-8')
    fr = A.pruefe_start(lk, A.CFG, A.SPEC_PATH)       # Probe: die frisch geschriebene Bindung muss die Pruefung bestehen
    out = {kk: v for kk, v in d.items() if kk not in ('cfg',)}
    out['sperrdatei_sha256'] = fr.hash
    return out


# ------------------------------------------------------------------ Annotation
def annotation_csv_lesen(pfad):
    """Anwender-CSV: Spalten annot_id und t_mm (T-Amplitude in mm, Komma oder Punkt als Dezimalzeichen). Rueckgabe DataFrame annot_id, t_anwender (mV, 10 mm/mV)."""
    import pandas as pd
    d = pd.read_csv(pfad, sep=None, engine='python', dtype=str)
    d.columns = [c.strip().lower() for c in d.columns]
    if 'annot_id' not in d or 't_mm' not in d:
        raise ValueError('CSV braucht die Spalten annot_id und t_mm')
    d['t_anwender'] = pd.to_numeric(d['t_mm'].str.strip().str.replace(',', '.', regex=False), errors='coerce') / 10.0
    return d[['annot_id', 't_anwender']].dropna()


def annotation_bereit(k):
    return k.verzicht.exists() or bool(k.anwender_csvs)


def annotation_vorbereiten(k=None):
    """Kaliumblinde Referenzannotation (Spezifikation 10, ungebundener Treiberpfad, Bekanntes B2): nur Train und Val; zufaellige, kaliumunabhaengige Teilmenge der
    Serumentnahmen mit Block (die feste Ein-Entnahme-je-Person-Stichprobe des Skripts); Messung mit der gebundenen Kette; Zufallsstichprobe n Bloecke; Bilder ohne Kalium."""
    import numpy as np
    import pandas as pd
    k = k or Konfig()
    if not k.signal_annot.exists():
        raise SystemExit('START_ANNOTATION.txt fehlt: Annotationsvorbereitung nicht freigegeben')
    A = lade_skript(k)
    fr = A.pruefe_start(lock_pfad(A, k), A.CFG, A.SPEC_PATH)
    A.setze_bindung(fr)
    A.WAVE = k.welle
    A.eingaben_pruefen(k.daten, A.EINGABEN_OHNE_TEST)
    tr, va = A.split_csns(k.daten, 'train', fr), A.split_csns(k.daten, 'val', fr)
    menge = A.CSNMenge(sorted(set(tr) | set(va)), {**tr.zuordnung, **va.zuordnung}, False)
    c = A.verbinden(k.privat / 's3_annotation.duckdb')
    A.laden(c, k.daten, k.welle, menge, fr)
    D = A.entnahmen_tabelle(c, fr, A.CFG)
    Ds = A.draws_serum(D)
    seg = c.execute('SELECT * FROM seg').df()
    leer_p = pd.DataFrame(columns=['CSN', 'ta', 'tb', 'ka'])
    leer_h = pd.DataFrame(columns=['CSN', 'ta'])
    Kd = A.kandidaten(Ds, leer_p, leer_p, leer_h, seg, A.CFG)
    Kd = Kd[Kd.kand_pk]                                # nur die kaliumunabhaengige Zufallsstichprobe, nie die Kaliumkandidaten
    rng = np.random.default_rng(A.CFG['seeds']['annotation'])
    Kd = Kd.iloc[rng.permutation(len(Kd))[:k.n_annot_pool]]
    sg_ = A.segs_von(seg)
    arbeit = [(int(r.CSN), float(r.t_order), float(r.t_result) if r.t_result == r.t_result else float('nan'), sg_.get(int(r.CSN), []),
               {'rep': False, 'shift': False, 'cfg': A.CFG, 'wave': str(k.welle), 'bindung': dict(fr.abhaengig)}) for r in Kd.itertuples()]
    if k.arbeiter > 1:
        from multiprocessing import Pool
        with Pool(k.arbeiter) as p:
            res = list(p.imap(A.messe_entnahme, arbeit, chunksize=4))
    else:
        res = [A.messe_entnahme(a) for a in arbeit]
    M = pd.DataFrame(res)
    Dm = Ds.merge(M, on=['CSN', 't_order'], how='inner')
    Dm['c'] = Dm['c_d3'] if 'c_d3' in Dm else np.nan
    liste, schluessel = A.annotation_stichprobe(Dm, k.n_annot, None, A.CFG)
    k.ann.mkdir(parents=True, exist_ok=True)
    bilder, pipe = [], []
    for r in schluessel.itertuples():
        zeile = Dm[(Dm.CSN == r.CSN) & (np.isclose(Dm.c, r.c))].iloc[0]
        segs = [s for s in sg_.get(int(r.CSN), []) if s[3] <= r.c <= s[4]]
        if not segs:
            bilder.append('')
            continue
        rd = A.lies_roh(int(r.CSN), segs[0], r.c - 5, r.c + 5, k.welle)
        pn = k.ann / f'{r.annot_id}.png'
        A.annotation_bild(rd[1], rd[0], r.c, pn)
        bilder.append(pn.name)
        pipe.append({'annot_id': r.annot_id, 'CSN': int(r.CSN), 'c': float(r.c), 't_pipeline': float(zeile.get('t_amp_d3', np.nan)), 'valid': float(zeile.get('valid_d3', 0)),
                     'pol': float(zeile.get('pol_d3', np.nan))})
    liste['bild'] = bilder
    liste.to_csv(k.ann / 'annotation_liste.csv', index=False)
    pd.DataFrame({'annot_id': liste.annot_id, 't_mm': ''}).to_csv(k.ann / 'annotation_anwender_VORLAGE.csv', index=False)
    schluessel.to_csv(k.privat / 'annotation_schluessel_privat.csv', index=False)
    pd.DataFrame(pipe).to_csv(k.privat / 'annotation_pipeline_privat.csv', index=False)
    (k.ann / 'ANLEITUNG.txt').write_text(
        'Kaliumblinde Referenzannotation (nur Messung der T-Amplitude, kein Kaliumwert, keine Behandlung, keine Pipelinewerte).\n'
        'Je Bild (Datei <annot_id>.png, Raster 25 mm/s und 10 mm/mV): Hoehe des T-Gipfels ueber der PR-Strecke in mm, positiv, wenn T in der ueberwiegenden\n'
        'QRS-Richtung liegt, negativ sonst. Mehrere Schlaege: typischen Schlag nehmen. Nicht beurteilbar: Feld leer lassen.\n'
        'Ergebnis: Datei annotation_anwender.csv (Vorlage annotation_anwender_VORLAGE.csv, Spalten annot_id;t_mm, Komma oder Punkt als Dezimalzeichen) in diesem Ordner ablegen.\n',
        encoding='utf-8')
    c.close()
    return {'bloecke': int(len(liste)), 'ordner': str(k.ann), 'pool': int(len(M))}


def annotation_auswerten(k=None):
    import pandas as pd
    k = k or Konfig()
    A = lade_skript(k)
    user = annotation_csv_lesen(k.anwender_csv)
    pipe = pd.read_csv(k.privat / 'annotation_pipeline_privat.csv')
    pipe = pipe[(pipe.valid == 1)][['annot_id', 't_pipeline']].dropna()
    e = A.annotation_auswertung(user, pipe)
    e['n_anwender'] = int(len(user))
    e['n_pipeline_gueltig'] = int(len(pipe))
    e['hinweis'] = 'beschreibend, ohne Tor; Messrichtigkeit der Pipeline gegen die kaliumblinde Anwendermessung'
    out = k.st / 'annotation_ergebnis.json'
    k.st.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(A.mask_tree(e), indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    return e


# ------------------------------------------------------------------ Gesamtlauf
def bereitschaft(k=None):
    k = k or Konfig()
    A = lade_skript(k)
    r = {'sperrdatei_vorhanden': lock_pfad(A, k).exists(), 'start_signal': k.signal.exists(), 'annotation_csv': bool(k.anwender_csvs), 'annotation_verzicht': k.verzicht.exists(),
         'annotation_signal': k.signal_annot.exists(), 'stop': k.stop.exists(), 'fertig': k.fertig.exists(), 'sha256sums': (k.daten.parent / 'SHA256SUMS.txt').exists()}
    try:
        fr = A.pruefe_start(lock_pfad(A, k), A.CFG, A.SPEC_PATH)
        r['bindung_gueltig'] = True
        r['sperrdatei_sha256'] = fr.hash
    except SystemExit as ex:
        r['bindung_gueltig'] = False
        r['bindung_fehler'] = str(ex)
    r['start_erlaubt'] = bool(r['bindung_gueltig'] and r['start_signal'] and (r['annotation_csv'] or r['annotation_verzicht']) and not r['stop'] and r['sha256sums'])
    return r


def gesamtlauf_treiber(k=None):
    k = k or Konfig()
    k.st.mkdir(parents=True, exist_ok=True)
    if not k.signal.exists():
        log('START_SIGNAL.txt fehlt: Gesamtlauf nicht freigegeben')
        return 5
    if not annotation_bereit(k):
        log('Annotation fehlt (annotation_anwender.csv) und kein ANNOTATION_VERZICHT.txt: Gesamtlauf nicht freigegeben')
        return 5
    k.pidfile.write_text(str(os.getpid()), encoding='utf-8')
    z = zustand(k)
    log('Treiber gestartet, PID', os.getpid())
    try:
        if k.fertig.exists():
            log('FERTIG.txt vorhanden, nichts zu tun')
            return 0
        if k.stop.exists():
            log('STOP.txt vorhanden:', k.stop.read_text(encoding='utf-8', errors='replace')[:200])
            return 3
        A = lade_skript(k)
        z['schritte']['gesamtlauf'] = {'status': 'laeuft', 'start': z.get('schritte', {}).get('gesamtlauf', {}).get('start') or jetzt()}
        z['neustarts'] = z.get('neustarts', 0)
        sichere(k, z)
        warte_speicher(k)
        with Reservierung(k, 'S3-MC-MED Gesamtlauf (alle Splits)'):
            A.gesamtlauf(k.daten, k.welle, k.privat, lock_pfad(A, k), A.CFG, k.arbeiter, k.privat / 'auswertung_v2_2', A.SPEC_PATH)
        z['schritte']['gesamtlauf'].update({'status': 'fertig', 'ende': jetzt()})
        z['hinweis'] = 'FERTIG: Ausgabe unter ' + str(k.privat / 'auswertung_v2_2' / 'gesamt_maskiert.json')
        sichere(k, z)
        k.fertig.write_text(jetzt(), encoding='utf-8')
        return 0
    except Angehalten as ex:
        log('angehalten:', ex)
        z['hinweis'] = f'Angehalten: {ex}'
        sichere(k, z)
        return 3
    except (SystemExit, PermissionError, ValueError) as ex:
        # Verweigerung durch Bindungs-, Eingabe- oder Zeitvertragspruefung: fatal, kein Neustart
        k.stop.write_text(f'{jetzt()} Verweigerung durch das Skript: {type(ex).__name__}: {ex}\n', encoding='utf-8')
        log('FATAL (Verweigerung):', ex)
        z['hinweis'] = f'HALT durch Verweigerung des Skripts: {str(ex)[:300]}'
        sichere(k, z)
        return 4
    except Exception:
        log('Absturz:', traceback.format_exc())
        z['hinweis'] = 'Absturz des Treibers (siehe treiber.log); der Wachprozess startet neu.'
        sichere(k, z)
        return 1


def wacht_soll_starten(k):
    """Der Wachprozess startet den Treiber nur nach START_SIGNAL.txt und nur, wenn weder FERTIG noch STOP vorliegt."""
    return k.signal.exists() and not k.fertig.exists() and not k.stop.exists()


def start(k=None):
    """Startsignal (nur nach Entscheidung K): schreibt START_SIGNAL.txt und startet den Wachprozess losgeloest per WMI."""
    k = k or Konfig()
    b = bereitschaft(k)
    if not (b['bindung_gueltig'] and (b['annotation_csv'] or b['annotation_verzicht']) and b['sha256sums'] and not b['stop']):
        raise SystemExit('nicht bereit: ' + json.dumps({x: b[x] for x in ('bindung_gueltig', 'annotation_csv', 'annotation_verzicht', 'sha256sums', 'stop')}))
    k.st.mkdir(parents=True, exist_ok=True)
    k.signal.write_text(jetzt(), encoding='utf-8')
    cmd = f'cmd.exe /c ""{PY}" -I "{HIER / "s3_wacht.py"}" >> "{k.st / "wacht_start.log"}" 2>&1"'
    r = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(HIER / 'start_wmi_s3.ps1'), '-CommandLine', cmd], capture_output=True)
    return r.stdout.decode('cp850', errors='replace').strip()


# ------------------------------------------------------------------ Selbsttest (Kunstdaten, kein Zugriff auf MC-MED)
def selbsttest():
    import numpy as np
    import pandas as pd
    erg = []

    def T(name):
        def deko(fn):
            try:
                fn()
                erg.append((name, True, ''))
            except Exception as e:
                erg.append((name, False, repr(e)[:300] + ' ' + traceback.format_exc().splitlines()[-3][:150]))
            print('OK    ' if erg[-1][1] else 'FEHLER', name, erg[-1][2], flush=True)
            return fn
        return deko

    def muss(exc, fn, text=''):
        try:
            fn()
        except exc:
            return
        raise AssertionError('kein Abbruch: ' + text)

    with tempfile.TemporaryDirectory() as td:
        A0 = lade_skript()
        daten_root = Path(td) / 'ds'
        info = A0.kunst_datensatz(daten_root, n_visits=20, beta=0.02, n_beh=0)
        k = Konfig(daten=daten_root / 'data', privat=Path(td) / 'priv', lock=Path(td) / 'priv' / 'lock.json', spec=Path(td) / 'spec.md', ram=False, arbeiter=1, n_annot=5, n_annot_pool=20)
        k.spec.write_text('Spezifikation (Test)')
        k.privat.mkdir(parents=True, exist_ok=True)
        # kleine Testkonfiguration, nur fuer den Test (die Produktion nutzt CFG des Skripts); pruefe_start/gesamtlauf nutzen A.CFG, daher CFG im Skript-Objekt anpassen
        orig_laden = lade_skript

        def lade_test(kk=None):
            A = orig_laden(kk)
            A.CFG['bootstrap_n'] = 100
            A.CFG['nk2_n'] = 40
            A.CFG['min_schaetzung'] = {'paare': 10, 'personen': 5}
            A.CFG['nk']['n_min'] = 5
            return A

        globals()['lade_skript'] = lade_test

        @T('binden: Sperrdatei ueber modus_sperren, Verweis auf SHA256SUMS, Probe der Pruefung; zweites Binden verweigert')
        def _():
            out = binden(k)
            assert k.lock.exists() and out['eingaben']['sha256sums_sha256'] and set(out['eingaben']['eingabedateien_laut_sha256sums']) == set(A0.EINGABEN)
            assert out['hashes']['spezifikation'] and 'konfiguration' in out['hashes'] and out['gebunden'] and out['sperrdatei_sha256']
            muss(SystemExit, lambda: binden(k), 'zweites Binden')

        @T('binden ohne SHA256SUMS bricht ab (keine Sperrdatei)')
        def _():
            k2 = Konfig(daten=Path(td) / 'ohne' / 'data', privat=Path(td) / 'priv2', lock=Path(td) / 'priv2' / 'lock.json', spec=k.spec, ram=False)
            (Path(td) / 'ohne' / 'data').mkdir(parents=True)
            muss(SystemExit, lambda: binden(k2), 'SHA256SUMS fehlt')
            assert not k2.lock.exists()

        @T('Bereitschaft: ohne Signale nicht erlaubt; Gesamtlauf und Annotation verweigern ohne Freigabesignal')
        def _():
            b = bereitschaft(k)
            assert b['bindung_gueltig'] and not b['start_erlaubt'] and not b['start_signal']
            assert gesamtlauf_treiber(k) == 5 and not (k.privat / 's3_gesamt.duckdb').exists()
            muss(SystemExit, lambda: annotation_vorbereiten(k), 'Annotation ohne START_ANNOTATION')
            assert not (k.privat / 's3_annotation.duckdb').exists()
            muss(SystemExit, lambda: start(k), 'start ohne Annotation')

        @T('Annotationsvorbereitung: nur Train/Val, Bilder und Listen ohne Kalium, Schluessel und Pipelinewerte privat')
        def _():
            k.st.mkdir(parents=True, exist_ok=True)
            k.signal_annot.write_text('x')
            r = annotation_vorbereiten(k)
            liste = pd.read_csv(k.ann / 'annotation_liste.csv')
            assert r['bloecke'] == 5 and list(liste.columns) == ['annot_id', 'bild'] and all((k.ann / b).stat().st_size > 1000 for b in liste.bild)
            sch = pd.read_csv(k.privat / 'annotation_schluessel_privat.csv')
            assert not (set(sch.CSN) & set(info['test'])), 'Testbesuch in der Annotation'
            assert (k.ann / 'ANLEITUNG.txt').exists() and list(pd.read_csv(k.ann / 'annotation_anwender_VORLAGE.csv').columns) == ['annot_id', 't_mm']
            assert 't_pipeline' in pd.read_csv(k.privat / 'annotation_pipeline_privat.csv').columns
            assert not any('pipeline' in str(p) or 'schluessel' in str(p) for p in k.ann.iterdir()), 'private Dateien im Anwenderordner'

        @T('Anwender-CSV: Komma/Punkt, Semikolon, mm nach mV; Auswertung gegen Pipelinewerte')
        def _():
            pipe = pd.read_csv(k.privat / 'annotation_pipeline_privat.csv')
            r = np.random.default_rng(3)
            mm_ = (pipe.t_pipeline * 10 + r.normal(0, 0.2, len(pipe))).round(2).astype(str).str.replace('.', ',', regex=False)
            pd.DataFrame({'annot_id': pipe.annot_id, 't_mm': mm_}).to_csv(k.anwender_csv, sep=';', index=False)
            u = annotation_csv_lesen(k.anwender_csv)
            assert len(u) == len(pipe) and abs(u.t_anwender.iloc[0] - float(mm_.iloc[0].replace(',', '.')) / 10) < 1e-9
            muss(ValueError, lambda: annotation_csv_lesen(k.privat / 'annotation_pipeline_privat.csv'), 'falsche Spalten')
            e = annotation_auswerten(k)
            assert (k.st / 'annotation_ergebnis.json').exists() and e['n_anwender'] == len(pipe)

        @T('Gesamtlauf-Treiber: verweigert ohne Signal, laeuft mit Signal und Annotation; FERTIG; Ausgabe maskiert')
        def _():
            assert not wacht_soll_starten(k)
            assert gesamtlauf_treiber(k) == 5 and not k.fertig.exists()                  # Annotation vorhanden, aber kein START_SIGNAL
            b = bereitschaft(k)
            assert b['annotation_csv'] and not b['start_erlaubt']
            k.signal.write_text('x')
            assert bereitschaft(k)['start_erlaubt'] and wacht_soll_starten(k)
            rc = gesamtlauf_treiber(k)
            assert rc == 0 and k.fertig.exists() and not wacht_soll_starten(k), rc
            out = json.loads((k.privat / 'auswertung_v2_2' / 'gesamt_maskiert.json').read_text(encoding='utf-8'))
            js = json.dumps(out)
            assert not re.search(r'(?<![\d.])\d{8,}(?![\d.])', js) and out['h1']['primaer']['beurteilbar'] is not None
            assert gesamtlauf_treiber(k) == 0                                              # FERTIG vorhanden: nichts zu tun
            assert zustand(k)['schritte']['gesamtlauf']['status'] == 'fertig'

        @T('Annotationsbereitschaft: CSV-Export der HTML-Annotation oder annotation_anwender.csv oder Verzicht')
        def _():
            k4 = Konfig(daten=k.daten, privat=Path(td) / 'priv4', ram=False)
            assert not annotation_bereit(k4)
            k4.ann.mkdir(parents=True)
            (k4.ann / 'referenzannotation_s3_ergebnisse_20261017_0930.csv').write_text('a,b')
            assert annotation_bereit(k4) and len(k4.anwender_csvs) == 1
            k5 = Konfig(daten=k.daten, privat=Path(td) / 'priv5', ram=False)
            k5.st.mkdir(parents=True)
            k5.verzicht.write_text('x')
            assert annotation_bereit(k5) and not k5.anwender_csvs

        @T('Gesamtlauf-Treiber: Verweigerung durch das Skript (veraenderte Spezifikation) setzt STOP und startet nicht neu')
        def _():
            k3 = Konfig(daten=k.daten, privat=Path(td) / 'priv3', lock=Path(td) / 'priv3' / 'lock.json', spec=Path(td) / 'spec3.md', ram=False, arbeiter=1)
            k3.spec.write_text('Spezifikation (Test 3)')
            k3.privat.mkdir(parents=True)
            binden(k3)
            k3.st.mkdir(parents=True, exist_ok=True)
            k3.signal.write_text('x')
            k3.verzicht.write_text('x')
            k3.spec.write_text('veraendert nach der Bindung')
            assert gesamtlauf_treiber(k3) == 4 and k3.stop.exists() and not wacht_soll_starten(k3)
            assert not (k3.privat / 's3_gesamt.duckdb').exists()

        @T('RAM-Pruefung ueber ram_reservierung.py erreichbar (Rueckgabe 0 oder 3), ohne etwas einzutragen')
        def _():
            rc, out = ram_aufruf('pruefen', '--gb', '0.1', '--seite', 'analyse')
            assert rc in (0, 3), (rc, out)

        globals()['lade_skript'] = orig_laden
    n_ok = sum(1 for _, ok, _ in erg if ok)
    print(f'{n_ok} von {len(erg)} bestanden')
    return n_ok == len(erg)


if __name__ == '__main__':
    m = sys.argv[1] if len(sys.argv) > 1 else ''
    if m == 'binden':
        print(json.dumps(binden(), indent=1, ensure_ascii=False))
    elif m == 'pruefen':
        print(json.dumps(bereitschaft(), indent=1))
    elif m == 'annotation':
        print(json.dumps(annotation_vorbereiten(), indent=1))
    elif m == 'annotation_auswerten':
        print(json.dumps(annotation_auswerten(), indent=1, default=str))
    elif m == 'start':
        print(start())
    elif m == 'lauf':
        sys.exit(gesamtlauf_treiber())
    elif m == 'stand':
        print(json.dumps(zustand(Konfig()), indent=1, default=str))
    elif m == 'selbsttest':
        sys.exit(0 if selbsttest() else 1)
    else:
        print(__doc__)
        sys.exit(2)
