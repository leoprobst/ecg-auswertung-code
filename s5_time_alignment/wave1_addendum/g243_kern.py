"""AW-243 Kern: Beobachtungszeit nach Entlassung (S5-R4), neu geschrieben, ganzzahlige Sekunden.

Unabhaengig vom archivierten risk_set und von den AW-242-Skripten. Alle Zeiten relativ zur EKG-Zeit in
ganzen Sekunden (int64). Zwei Varianten der Beobachtungsspanne:
  'archiv'  : Spanne bis zum letzten Kontakt als Stundengrenzen-Differenz (followup_hours * 3600)
  'exakt'   : Spanne in Sekunden (last_observed_time - ecg_time)
Tod ist ein Datum (00:00 Uhr), kein Zeitpunkt.
"""
import re
import numpy as np
import pandas as pd

H = 3600
DAY = 86400
NOINF = np.iinfo(np.int64).max // 4  # Ersatz fuer 'kein Tod'


def secs(ts):
    """datetime64-Serie -> int64 Sekunden seit 1970 (NaT -> Minimum)."""
    a = pd.to_datetime(ts).to_numpy().astype('datetime64[s]')
    out = a.astype('int64')
    out[np.isnat(a)] = np.iinfo(np.int64).min
    return out, ~np.isnat(a)


def hour_boundary_diff(t0_s, t1_s):
    """Entspricht DuckDB DATE_DIFF('hour', t0, t1): Zahl der ueberschrittenen Stundengrenzen (Boden je Zeitpunkt)."""
    return np.floor_divide(t1_s, H) - np.floor_divide(t0_s, H)


def report_sinus(text):
    """Berichteter Sinusrhythmus ohne Verneinung (eigene Formulierung der Auswahlregel)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return False
    s = str(text).lower()
    pos = re.search(r'\bsinus\s+(rhythm|tachycardia|bradycardia|arrhythmia)\b', s) is not None
    neg = (re.search(r'\bno(?:\s+longer)?\s+sinus', s) is not None
           or re.search(r'sinus[^\n]{0,25}(?:has been replaced|not present)', s) is not None
           or re.search(r'\b(?:possible|probable|uncertain)[^\n]{0,20}\bsinus', s) is not None)
    return pos and not neg


def prepare(lm, last_obs_ts):
    """lm: DataFrame der Landmarken (ecg_time, dischtime, date_of_death, followup_hours, next_af_event_delta_hours).
    last_obs_ts: Serie 'letzter Kontakt' (Timestamp) in derselben Reihenfolge. Rueckgabe: DataFrame mit Sekundenspalten."""
    ecg, _ = secs(lm['ecg_time'])
    disch, _ = secs(lm['dischtime'])
    lo, lo_ok = secs(last_obs_ts)
    dod, dod_ok = secs(lm['date_of_death'])
    out = pd.DataFrame(index=lm.index)
    out['lm'] = disch - ecg
    out['fu_exakt'] = np.where(lo_ok, lo - ecg, np.iinfo(np.int64).min)
    out['fu_archiv'] = pd.to_numeric(lm['followup_hours'], errors='coerce').astype('float64').to_numpy() * H
    out['fu_hourdiff_nachgerechnet'] = np.where(lo_ok, hour_boundary_diff(ecg, lo) * H, np.nan)
    out['death'] = np.where(dod_ok, dod - ecg, np.nan)
    fut = pd.to_numeric(lm['next_af_event_delta_hours'], errors='coerce').astype('float64').to_numpy()
    out['future'] = np.rint(fut * H)
    return out


def risk_table(t, mode, horizon_days=365):
    """Zeilenweise Entscheidung. mode in {'archiv','exakt'}. Rueckgabe DataFrame mit Booleans/Zeiten (Sekunden)."""
    fu = t['fu_archiv'] if mode == 'archiv' else t['fu_exakt'].astype('float64')
    lm = t['lm'].astype('float64')
    dth = t['death']
    fut = t['future']
    hz = float(horizon_days * DAY)
    has_death = dth.notna()
    g6 = fut.isna() | ((fut <= fu) & (~has_death | (fut < dth)))
    g7 = (fu > lm) & (~has_death | (dth > lm))
    d_rel = np.where(has_death, dth - lm, np.inf)
    cens = np.minimum(fu - lm, d_rel)
    ev_t = fut - lm
    event = (ev_t > 0) & (ev_t <= cens) & (ev_t <= hz)
    event = event.fillna(False) if hasattr(event, 'fillna') else event
    time_s = np.where(event, ev_t, np.minimum(cens, hz))
    keep = g6 & g7 & np.isfinite(time_s) & (time_s > 0)
    return pd.DataFrame({'g6': g6, 'g7': g7, 'event': event.astype(bool), 'time_s': time_s, 'keep': keep,
                         'contact_rel': fu - lm, 'death_rel': d_rel}, index=t.index)


def census(rt, mask, horizon_days=365):
    """Endklassifikation der Risikopersonen: AF, Kontaktende, Todesende, Horizontende, Gleichstand (ohne AF)."""
    hz = float(horizon_days * DAY)
    r = rt[mask]
    ends = np.vstack([r['contact_rel'].to_numpy(float), r['death_rel'].to_numpy(float), np.full(len(r), hz)]).T
    mn = ends.min(axis=1)
    eq = (ends == mn[:, None])
    tie = eq.sum(axis=1) > 1
    noaf = ~r['event'].to_numpy(bool)
    res = dict(n=int(len(r)), af=int((~noaf).sum()),
               contact_end=int((noaf & ~tie & eq[:, 0]).sum()),
               death_end=int((noaf & ~tie & eq[:, 1]).sum()),
               horizon_end=int((noaf & ~tie & eq[:, 2]).sum()),
               tied=int((noaf & tie).sum()))
    assert res['af'] + res['contact_end'] + res['death_end'] + res['horizon_end'] + res['tied'] == res['n']
    res['no_af'] = res['n'] - res['af']
    return res


def loss_causes(t, rt):
    """Nullzeitverluste unter kompatiblen Landmarken (g6) ohne positive Beobachtungszeit (g7), hierarchisch."""
    lost = rt['g6'] & ~rt['g7']
    has_death = t['death'].notna()
    c_death = lost & has_death & (t['death'] <= t['lm'])
    rest = lost & ~c_death
    c_nocontact = rest & (rt['contact_rel'] <= 0)
    c_other = rest & ~c_nocontact
    return dict(lost=int(lost.sum()), death_on_or_before_discharge_day=int(c_death.sum()),
                no_contact_after_index_discharge=int(c_nocontact.sum()), other=int(c_other.sum()))


def kz(n):
    """Kleinzellenregel: Zaehlwerte 1 bis 9 als '<10'."""
    n = int(n)
    return '<10' if 0 < n < 10 else str(n)
