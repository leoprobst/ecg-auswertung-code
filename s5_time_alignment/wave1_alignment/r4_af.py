"""AW-242 S5-R4: AF-Beobachtungszeit als administrativ definierte Spanne. Ursachen der Nullzeitverluste, Zensus ohne
Stundenrundung, Kontaktquellen, Gleichstandsregel, Kontaktluecken-Varianten (post hoc).

Eingaben (nur Entwicklungsfolds train/validation, keine Testpartition):
 - first-eligible-landmarks.parquet (A155, AF-Landmarken v2, gesiegelt)
 - clinical_context_features.parquet (nur Spalten last_observed_time, hospital_expire_flag, ueber die Landmarken-Schluessel)
 - patient_clinical_timeline.parquet (last_hospital_discharge_time, last_ed_contact_time)
 - admissions.parquet, ecg_master_index.parquet (nur subject_id, ecg_time der Landmarken-Personen)
Ausgaben ohne Kennungen: nachrechnung/AW242/S5_R4_*.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, json, ast, re, time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import aw242_guard as G
G.start('r4')
import numpy as np, pandas as pd, duckdb

OUT = G.OUT
ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / 'nachrechnung/AW242'; PUB.mkdir(parents=True, exist_ok=True)
PP = Path(str(data_path("s5.mimic_validation_runs_a155_2026_10_04_d6e70e_publication_preparation_research_documented_af_landmark_20261012_v2")))
SRC = PP / 'executed_source/a155_research_documented_af.py'
CACHE = PP / 'first-eligible-landmarks.parquet'
EV = Path(str(data_path("s5.mimic_validation_evidence")))
STD = Path(str(data_path("s5.mimic_validation_standardized")))
MASTER = str(data_path("s5.mimic_validation_master_ecg_master_index_parquet"))
H = 3600.0


def load_pure():
    code = SRC.read_text(encoding='utf-8-sig')
    names = {'risk_set', 'report_sinus'}
    nodes = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 2
    ns = {'np': np, 'pd': pd, 're': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<AF archiviert>', 'exec'), ns)
    return ns['risk_set']


def kz(n):
    n = int(n)
    return '<10' if 0 < n < 10 else str(n)


def contact_chain_end(disch, intervals, g_days):
    """Ende der Kette zusammenhaengender Kontakte nach der Indexentlassung (Sekunden).

    intervals: (Beginn, Ende) von Aufnahmen und (t, t) von EKG; nur Kontakte nach der Indexentlassung zaehlen.
    Die Kette bricht beim ersten Abstand zwischen Kettenende und naechstem Kontaktbeginn von mehr als g_days.
    """
    e = disch
    lim = g_days * 86400.0
    for x, y in sorted(iv for iv in intervals if iv[0] > disch):
        if x - e <= lim:
            e = max(e, y)
        else:
            break
    return e


def census(frame, horizon, follow, landmark, death, ties='min'):
    """Endklassifikation wie AW-181: Kontaktende/Todesende/Horizontende/Gleichstand unter Personen ohne AF.

    follow, landmark, death in Stunden seit EKG; Kontakt = follow - landmark; Tod = death - landmark.
    """
    contact = follow - landmark
    dth = (death - landmark).fillna(np.inf)
    ends = pd.DataFrame({'contact': contact, 'death': dth, 'horizon': horizon * 24.0}, index=frame.index)
    mi = ends.min(axis=1)
    tie = ends.eq(mi, axis=0).sum(axis=1) > 1
    no = frame.event.eq(0)
    r = dict(n=len(frame), af=int(frame.event.sum()),
             contact_end=int((no & ~tie & ends.contact.eq(mi)).sum()), death_end=int((no & ~tie & ends.death.eq(mi)).sum()),
             horizon_end=int((no & ~tie & ends.horizon.eq(mi)).sum()), tied=int((no & tie).sum()))
    assert r['af'] + r['contact_end'] + r['death_end'] + r['horizon_end'] + r['tied'] == r['n']
    return r, contact, ends


def main():
    t0 = time.time()
    risk_set = load_pure()
    d = pd.read_parquet(CACHE)
    assert set(d.split.unique()) <= {'train', 'validation'} and not d.subject_id.duplicated().any()
    c = duckdb.connect(); c.execute("SET memory_limit='900MB'; SET threads=2")
    c.register('lm', d[['subject_id', 'study_id']])
    ctx = c.execute(f"SELECT x.subject_id, x.study_id, x.last_observed_time, x.hospital_expire_flag FROM read_parquet('{(EV / 'clinical_context_features.parquet').as_posix()}') x JOIN lm USING(subject_id, study_id)").df()
    tl = c.execute(f"SELECT t.subject_id, t.last_hospital_discharge_time, t.last_ed_contact_time FROM read_parquet('{(EV / 'patient_clinical_timeline.parquet').as_posix()}') t JOIN (SELECT DISTINCT subject_id FROM lm) USING(subject_id)").df()
    adm = c.execute(f"SELECT a.subject_id, a.admittime, a.dischtime FROM read_parquet('{(STD / 'admissions.parquet').as_posix()}') a JOIN (SELECT DISTINCT subject_id FROM lm) USING(subject_id)").df()
    ecgs = c.execute(f"SELECT m.subject_id, m.ecg_time FROM read_parquet('{MASTER}') m JOIN (SELECT DISTINCT subject_id FROM lm) USING(subject_id)").df()
    c.close()
    assert len(ctx) == len(d), (len(ctx), len(d))
    d = d.merge(ctx, on=['subject_id', 'study_id'], validate='one_to_one').merge(tl, on='subject_id', how='left', validate='many_to_one')
    d['ecg_time'] = pd.to_datetime(d.ecg_time); d['dischtime'] = pd.to_datetime(d.dischtime)
    d['last_observed_time'] = pd.to_datetime(d.last_observed_time)
    d['follow_exact'] = (d.last_observed_time - d.ecg_time).dt.total_seconds() / H
    # Paritaet: followup_hours = Stundengrenzen-Differenz (DuckDB DATE_DIFF('hour'))
    fl = lambda s: s.dt.floor('h')
    chk = ((fl(d.last_observed_time) - fl(d.ecg_time)).dt.total_seconds() / H)
    par = float((chk == pd.to_numeric(d.followup_hours, errors='coerce')).mean())
    assert par > 0.999, par
    res_all = {'parity_followup_hours_equals_hour_boundary_difference': par}
    rows_flow = []; rows_cause = []; rows_shift = []; rows_cens = []; rows_src = []; rows_gap = []; rows_tie = []; rows_dist = []

    d_exact = d.copy(); d_exact['followup_hours'] = d_exact.follow_exact
    for fold in ('train', 'validation'):
        part = d[d.split == fold]; part_e = d_exact[d_exact.split == fold]
        n_sel = len(part)
        landmark = (part.dischtime - part.ecg_time).dt.total_seconds() / H
        future = pd.to_numeric(part.next_af_event_delta_hours, errors='coerce').astype(float)
        follow_r = pd.to_numeric(part.followup_hours, errors='coerce').astype(float)
        follow_x = part.follow_exact
        death = (pd.to_datetime(part.date_of_death) - part.ecg_time).dt.total_seconds() / H
        # Tore 6 und 7 getrennt nachgerechnet (Paritaet zum archivierten risk_set unten)
        def gates(follow):
            g6 = future.isna() | (future.le(follow) & (death.isna() | future.lt(death)))
            g7 = follow.gt(landmark) & (death.isna() | death.gt(landmark))
            return g6.fillna(False), g7.fillna(False)
        g6r, g7r = gates(follow_r); g6x, g7x = gates(follow_x)
        for hz in (365,):
            risk_r, flow_r = risk_set(part, hz)
            risk_x, flow_x = risk_set(part_e[part_e.split == fold], hz)
            rows_flow += [dict(fold=fold, variante='archiviert (Stundengrenzen)', **{'gate': f['gate'], 'n': f['n_retained']}) for f in flow_r]
            rows_flow += [dict(fold=fold, variante='exakte Sekunden', **{'gate': f['gate'], 'n': f['n_retained']}) for f in flow_x]
            rows_flow.append(dict(fold=fold, variante='archiviert (Stundengrenzen)', gate='Endmenge mit time_days>0', n=len(risk_r)))
            rows_flow.append(dict(fold=fold, variante='exakte Sekunden', gate='Endmenge mit time_days>0', n=len(risk_x)))
            # eigene Torzaehlung muss gleich den archivierten sein
            assert int(g6r.sum()) == flow_r[5]['n_retained'] and int((g6r & g7r).sum()) == flow_r[6]['n_retained']
            assert int(g6x.sum()) == flow_x[5]['n_retained'] and int((g6x & g7x).sum()) == flow_x[6]['n_retained']
            # ---- Ursachen der Nullzeit (Original) ----
            comp = g6r  # kompatible Referenzen
            lost = comp & ~g7r
            death_before = lost & death.notna() & death.le(landmark)
            hosp = part.hospital_expire_flag.fillna(0).astype(float).gt(0)
            rem = lost & ~death_before
            no_later_exact = rem & (part.last_observed_time <= part.dischtime)
            rem2 = rem & ~no_later_exact
            rounded_zero = rem2 & follow_r.le(landmark)
            other = rem2 & ~rounded_zero
            rows_cause += [
                dict(fold=fold, horizon=hz, ursache='1 ausgewaehlte Landmarken (erste geeignete Entlassung)', n=n_sel),
                dict(fold=fold, horizon=hz, ursache='2 inkompatible Ereignis-/Beobachtungsreferenz (Tor 6)', n=int((~g6r).sum())),
                dict(fold=fold, horizon=hz, ursache='3 kompatibel (Tor 6 erfuellt)', n=int(comp.sum())),
                dict(fold=fold, horizon=hz, ursache='4 ohne positive Beobachtungszeit insgesamt (Nullzeitverlust)', n=int(lost.sum())),
                dict(fold=fold, horizon=hz, ursache='4a Sterbedatum am oder vor dem Entlassungstag (Tagesdatum 00:00 vor Entlassungszeit)', n=int(death_before.sum())),
                dict(fold=fold, horizon=hz, ursache='4a davon Krankenhaussterblichkeit der verknuepften Aufnahme', n=int((death_before & hosp).sum())),
                dict(fold=fold, horizon=hz, ursache='4b kein Kontakt nach der Indexentlassung (letzter Kontakt = Entlassung, auf die Sekunde)', n=int(no_later_exact.sum())),
                dict(fold=fold, horizon=hz, ursache='4c Kontakt nach Entlassung vorhanden, aber Stundengrenzen-Differenz ergibt keine positive Zeit', n=int(rounded_zero.sum())),
                dict(fold=fold, horizon=hz, ursache='4d sonstige', n=int(other.sum())),
                dict(fold=fold, horizon=hz, ursache='5 Risikopersonen (archiviert)', n=len(risk_r)),
                dict(fold=fold, horizon=hz, ursache='6 Risikopersonen (exakte Sekunden)', n=len(risk_x))]
            # Verschiebungsmatrix Null/Positiv (nur kompatible, ohne Tod am/vor Entlassungstag)
            base = comp & ~death_before
            pos_r = base & follow_r.gt(landmark); pos_x = base & follow_x.gt(landmark)
            rows_shift += [dict(fold=fold, horizon=hz, archiviert_positiv=a, exakt_positiv=b, n=int((base & (pos_r == a) & (pos_x == b)).sum()))
                           for a in (True, False) for b in (True, False)]
            # Kontaktzeit nach Entlassung (exakt) der archivierten Risikopersonen
            idx = risk_r.index
            post_exact = (part.loc[idx, 'last_observed_time'] - part.loc[idx, 'dischtime']).dt.total_seconds() / H
            for lab, lo, hi in (('0 h (letzter Kontakt = Entlassung)', -1e9, 0.0), ('>0 bis 1 h', 0.0, 1.0), ('>1 bis 24 h', 1.0, 24.0), ('>1 bis 7 Tage', 24.0, 168.0),
                                ('>7 bis 30 Tage', 168.0, 720.0), ('>30 bis 365 Tage', 720.0, 8760.0), ('>365 Tage', 8760.0, 1e12)):
                m = post_exact.gt(lo) & post_exact.le(hi) if lo > -1e8 else post_exact.le(hi)
                rows_dist.append(dict(fold=fold, horizon=hz, klasse=lab, n_risikopersonen_archiviert=int(m.sum())))
            # ---- Zensus ----
            for var, rk, fol in (('archiviert (Stundengrenzen)', risk_r, follow_r), ('exakte Sekunden', risk_x, follow_x)):
                lm = (rk.dischtime if 'dischtime' in rk else part.loc[rk.index, 'dischtime'])
                lm = (pd.to_datetime(lm) - pd.to_datetime(rk.ecg_time)).dt.total_seconds() / H
                fo = fol.loc[rk.index]
                dh = (pd.to_datetime(rk.date_of_death) - pd.to_datetime(rk.ecg_time)).dt.total_seconds() / H
                r, contact, ends = census(rk, hz, fo, lm, dh)
                rows_cens.append(dict(fold=fold, horizon=hz, variante=var, **r))
                if var.startswith('exakte') is False:
                    rows_tie.append(dict(fold=fold, regel='Gleichstand Kontakt/Tod/Horizont ohne AF', n=r['tied']))
            # Anteil Kontaktende im Original, die auf scheinbar positive Zeit unter 1 h beruhen
            ra = risk_r.index
            sp = (part.loc[ra, 'last_observed_time'] <= part.loc[ra, 'dischtime'])
            rows_cause.append(dict(fold=fold, horizon=hz, ursache='7 Risikopersonen (archiviert) mit exakt null Kontaktzeit nach Entlassung (positive Zeit nur durch Rundung)', n=int(sp.sum())))
            # ---- Kontaktquellen der archivierten Risikopersonen ----
            lo_ = part.loc[ra, 'last_observed_time']; di_ = part.loc[ra, 'dischtime']; lh_ = pd.to_datetime(part.loc[ra, 'last_hospital_discharge_time'])
            cat = np.where(lo_ <= di_, 'letzter Kontakt = Indexentlassung', np.where(lo_ == lh_, 'Entlassung einer spaeteren Aufnahme', np.where(lo_ > lh_, 'EKG nach der letzten Entlassung (ambulant/ED/sonstiges EKG)', 'sonstiges')))
            ed = pd.to_datetime(part.loc[ra, 'last_ed_contact_time'])
            for k_, v_ in pd.Series(cat).value_counts().items():
                rows_src.append(dict(fold=fold, horizon=hz, quelle=k_, n=int(v_)))
            rows_src.append(dict(fold=fold, horizon=hz, quelle='Zusatz: ED-Aufenthalt nach dem letzten Kontakt (nicht Kontaktquelle der Definition)', n=int((ed > lo_).sum())))
            # ---- Kontaktluecken-Varianten (Kette zusammenhaengender Kontakte) ----
            sub_ids = part.subject_id.to_numpy()
            ad = adm[adm.subject_id.isin(sub_ids)]
            ec = ecgs[ecgs.subject_id.isin(sub_ids)]
            ad_g = {k: (v.admittime.astype('datetime64[s]').astype('int64').to_numpy() / 1.0, v.dischtime.astype('datetime64[s]').astype('int64').to_numpy() / 1.0) for k, v in ad.groupby('subject_id')}
            ec_g = {k: v.ecg_time.astype('datetime64[s]').astype('int64').to_numpy() / 1.0 for k, v in ec.groupby('subject_id')}
            disch_s = part.dischtime.astype('datetime64[s]').astype('int64').to_numpy() / 1.0
            ecg_s = part.ecg_time.astype('datetime64[s]').astype('int64').to_numpy() / 1.0
            comp_arr = comp.to_numpy(); death_arr = death.to_numpy(); fut = future.to_numpy()
            for G_days in (None, 30, 90):
                ends_s = np.empty(len(part))
                for i, s in enumerate(sub_ids):
                    if G_days is None:
                        ends_s[i] = (part.last_observed_time.iat[i] - pd.Timestamp('1970-01-01')).total_seconds()
                        continue
                    ivs = []
                    if s in ad_g:
                        a0, a1 = ad_g[s]
                        ivs += list(zip(a0, a1))
                    if s in ec_g:
                        ivs += [(x, x) for x in ec_g[s]]
                    ends_s[i] = contact_chain_end(disch_s[i], ivs, G_days)
                follow = pd.Series((ends_s - ecg_s) / H, index=part.index)
                pos = follow.gt(landmark) & (death.isna() | death.gt(landmark))
                # AF-Ereignis zaehlt nur, wenn es vor Ketten-/Todes-/Horizontende liegt (kein Ausschluss der Person)
                keep = comp & pos
                fr = part.loc[keep].copy()
                cens_h = np.minimum(follow[keep] - landmark[keep], np.where(death[keep].notna(), death[keep] - landmark[keep], np.inf))
                ev_h = future[keep] - landmark[keep]
                ev = ev_h.gt(0) & ev_h.le(cens_h) & ev_h.le(hz * 24.0)
                fr['event'] = ev.astype(int)
                r, contact, ends = census(fr, hz, follow[keep], landmark[keep], death[keep])
                outside_chain = int(((ev_h.gt(0)) & ev_h.le(hz * 24.0) & ev_h.gt(cens_h)).sum())
                label = 'Original-Ende (letzter Kontakt, exakte Sekunden)' if G_days is None else f'zusammenhaengende Kontaktkette, Luecke hoechstens {G_days} Tage'
                rows_gap.append(dict(fold=fold, horizon=hz, variante=label, kompatible_landmarken=int(comp.sum()), ohne_positive_zeit=int((comp & ~pos).sum()), risikopersonen=r['n'],
                                     af=r['af'], kontaktende=r['contact_end'], todesende=r['death_end'], horizontende=r['horizon_end'], gleichstand=r['tied'],
                                     af_dokumentiert_nach_kettenende_bis_horizont=outside_chain,
                                     kontaktende_anteil_ohne_af=(r['contact_end'] / (r['n'] - r['af'])) if r['n'] - r['af'] else None))
    out = {'flow': rows_flow, 'cause': rows_cause, 'shift': rows_shift, 'census': rows_cens, 'source': rows_src, 'gap': rows_gap, 'dist': rows_dist, 'parity': res_all}
    (OUT / 'r4_raw.json').write_text(json.dumps(out, indent=1, default=str))
    for k_, v_ in out.items():
        if k_ != 'parity':
            pd.DataFrame(v_).to_csv(OUT / f'r4_{k_}_raw.csv', index=False)
    print('R4 fertig', round(time.time() - t0, 1), 's, RSS', G.peak())


if __name__ == '__main__':
    main()
