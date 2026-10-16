"""AW-243 Teil 1 und 3: Gegenrechnung S5-R4 und Tabelle 2 (Complete Case) mit exakter Zeit.

Eingaben (nur Entwicklungsfolds train/validation, die gesiegelten Landmarken):
  first-eligible-landmarks.parquet (A155 AF-Landmarken v2), patient_clinical_timeline.parquet (letzter Kontakt, nur ueber
  die Personenschluessel), ecg_master_index.parquet (nur subject_id, ecg_time der Landmarkenpersonen) und
  admissions.parquet (nur dischtime der Landmarkenpersonen) fuer die unabhaengige Herleitung des letzten Kontakts.
Aggregate ohne Kennungen: nachrechnung/AW243/. Private Rohausgabe: <local path>.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, json, time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import g243_guard as G
G.start('r4')
import numpy as np, pandas as pd, duckdb
import g243_kern as K

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / 'nachrechnung/AW243'; PUB.mkdir(parents=True, exist_ok=True)
PP = Path(str(data_path("s5.mimic_validation_runs_a155_2026_10_04_d6e70e_publication_preparation_research_documented_af_landmark_20261012_v2")))
CACHE = PP / 'first-eligible-landmarks.parquet'
EV = Path(str(data_path("s5.mimic_validation_evidence")))
STD = Path(str(data_path("s5.mimic_validation_standardized")))
MASTER = str(data_path("s5.mimic_validation_master_ecg_master_index_parquet"))
FOLDS = {'train': 'Entwicklung', 'validation': 'Interne Pruefung'}
PRED = ['p_loop_width_major', 'age', 'sex_male', 'machine_heart_rate_bpm']


def merge_small(items):
    """Klassen mit 1 bis 9 Personen werden mit der naechsten Klasse zusammengefasst (Kleinzellenregel, Komplemente nicht ableitbar)."""
    out = []; carry_lab = None; carry_n = 0
    for lab, n in items:
        if carry_lab is not None:
            lab = carry_lab + ' + ' + lab
            n += carry_n; carry_lab = None; carry_n = 0
        if 0 < n < 10:
            carry_lab, carry_n = lab, n
            continue
        out.append((lab, n))
    if carry_lab is not None and out:
        l0, n0 = out.pop(); out.append((l0 + ' + ' + carry_lab, n0 + carry_n))
    return out


def main():
    t0 = time.time()
    meta = json.loads((PP / 'first-eligible-landmarks.sha256.json').read_text())
    sha_now = G.sha(CACHE)
    assert sha_now == meta['sha256'], 'Landmarkendatei weicht von der Siegelung ab'
    d = pd.read_parquet(CACHE)
    assert set(d.split.unique()) == {'train', 'validation'} and not d.subject_id.duplicated().any()
    con = duckdb.connect(); con.execute("SET memory_limit='900MB'; SET threads=2")
    con.register('ids', d[['subject_id']])
    tl = con.execute(f"SELECT t.subject_id, t.last_observed_time, t.first_af_event_time FROM read_parquet('{(EV / 'patient_clinical_timeline.parquet').as_posix()}') t JOIN ids USING(subject_id)").df()
    me = con.execute(f"SELECT m.subject_id, MAX(m.ecg_time) AS last_ecg FROM read_parquet('{MASTER}') m JOIN ids USING(subject_id) GROUP BY 1").df()
    ad = con.execute(f"SELECT a.subject_id, MAX(a.dischtime) AS last_disch FROM read_parquet('{(STD / 'admissions.parquet').as_posix()}') a JOIN ids USING(subject_id) GROUP BY 1").df()
    con.close()
    d = d.merge(tl, on='subject_id', how='left', validate='one_to_one').merge(me, on='subject_id', how='left', validate='one_to_one').merge(ad, on='subject_id', how='left', validate='one_to_one')
    d['own_last'] = d[['last_ecg', 'last_disch']].max(axis=1)
    out = {'landmark_sha256': sha_now, 'n_landmarks': {f: int((d.split == f).sum()) for f in FOLDS}}
    # --- Herleitung des letzten Kontakts: Zeitlinie gegen eigene Herleitung ---
    eq = (d.own_last == d.last_observed_time)
    out['last_contact_parity'] = {f: dict(gleich=int(eq[d.split == f].sum()), abweichend=int((~eq[d.split == f]).sum()),
                                          zeitlinie_spaeter=int((d.last_observed_time > d.own_last)[d.split == f].sum()),
                                          zeitlinie_frueher=int((d.last_observed_time < d.own_last)[d.split == f].sum())) for f in FOLDS}
    t = K.prepare(d, d['last_observed_time'])
    # Archivspalte gegen Stundengrenzen-Nachrechnung
    out['archiv_followup_hours_equals_hour_boundary'] = float((t['fu_hourdiff_nachgerechnet'] == t['fu_archiv']).mean())
    cc_mask_all = d[PRED].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    rows = {}
    priv = {}
    for fold, lab in FOLDS.items():
        m_fold = (d.split == fold).to_numpy()
        tf = t[m_fold]
        res = {'selected': int(m_fold.sum())}
        for mode in ('archiv', 'exakt'):
            rt = K.risk_table(tf, mode)
            c_all = K.census(rt, rt.keep)
            c_cc = K.census(rt, rt.keep & cc_mask_all[m_fold].to_numpy())
            lc = K.loss_causes(tf, rt)
            res[mode] = dict(kompatibel=int(rt.g6.sum()), inkompatibel=int((~rt.g6).sum()), positive_zeit_und_kompatibel=int((rt.g6 & rt.g7).sum()),
                             risk=c_all, cc=c_cc, verluste=lc)
            # Beobachtungszeit nach Entlassung in Sekunden (Spanne Indexentlassung bis letzter Kontakt) der Risikopersonen
            obs = (rt.contact_rel[rt.keep]).to_numpy(float)
            res[mode]['obs_s_quantile'] = {q: float(np.quantile(obs, q)) for q in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)} if len(obs) else {}
            res[mode]['obs_s_klassen'] = {'<=0': int((obs <= 0).sum()), '1-3599 s': int(((obs >= 1) & (obs < 3600)).sum()),
                                           '3600-86399 s': int(((obs >= 3600) & (obs < 86400)).sum()), '1-7 d': int(((obs >= 86400) & (obs < 7 * 86400)).sum()),
                                           '7-30 d': int(((obs >= 7 * 86400) & (obs < 30 * 86400)).sum()), '30-365 d': int(((obs >= 30 * 86400) & (obs < 365 * 86400)).sum()),
                                           '>=365 d': int((obs >= 365 * 86400).sum())}
            priv[(fold, mode)] = rt
        # Bruecke Archiv -> exakt
        ra, rx = priv[(fold, 'archiv')], priv[(fold, 'exakt')]
        kept_a, kept_x = ra.keep, rx.keep
        both = kept_a & kept_x
        only_a = kept_a & ~kept_x
        only_x = ~kept_a & kept_x
        nocontact_exact = (tf['fu_exakt'] <= tf['lm'])
        res['bruecke'] = dict(archiv_risk=int(kept_a.sum()), exakt_risk=int(kept_x.sum()), in_beiden=int(both.sum()),
                              nur_archiv=int(only_a.sum()), davon_exakt_kein_kontakt_nach_entlassung=int((only_a & nocontact_exact).sum()),
                              davon_tor6_exakt_inkompatibel=int((only_a & ~rx.g6).sum()),
                              nur_exakt=int(only_x.sum()),
                              kontaktende_archiv_davon_ohne_kontakt_exakt=int(((ra.keep & ~ra.event & nocontact_exact)).sum()))
        # Anteil Kontaktende ohne AF
        for mode in ('archiv', 'exakt'):
            r = res[mode]['risk']; c = res[mode]['cc']
            res[mode]['anteil_kontaktende_ohne_af'] = r['contact_end'] / r['no_af']
            res[mode]['anteil_kontaktende_ohne_af_cc'] = c['contact_end'] / c['no_af']
        # Kein Kontakt nach Entlassung unter kompatiblen exakten Landmarken
        comp = rx.g6
        res['kein_kontakt_nach_entlassung_exakt_kompatibel'] = dict(n=int((comp & nocontact_exact).sum()), nenner=int(comp.sum()),
                                                                     anteil=float((comp & nocontact_exact).sum() / comp.sum()))
        rows[fold] = res
    out['folds'] = rows
    # Gate 5 (bekannte AF bis Entlassung) archiv gegen exakt an den gesiegelten Landmarken
    fs, _ = K.secs(d['first_af_event_time'])
    ecg_s, _ = K.secs(d['ecg_time'])
    ok_first = d['first_af_event_time'].notna().to_numpy()
    first_exact = np.where(ok_first, fs - ecg_s, np.nan)
    first_arch = pd.to_numeric(d['first_af_event_delta_hours'], errors='coerce').to_numpy(float) * 3600
    lm = t['lm'].to_numpy(float)
    pass_arch = np.isnan(first_arch) | (first_arch > lm)
    pass_exact = np.isnan(first_exact) | (first_exact > lm)
    out['gate5_landmarken'] = {f: dict(archiv_besteht=int(pass_arch[(d.split == f).to_numpy()].sum()), exakt_besteht=int(pass_exact[(d.split == f).to_numpy()].sum()),
                                       nur_archiv_besteht=int((pass_arch & ~pass_exact)[(d.split == f).to_numpy()].sum()),
                                       nur_exakt_besteht=int((~pass_arch & pass_exact)[(d.split == f).to_numpy()].sum())) for f in FOLDS}
    out['first_af_hour_parity'] = float(np.nanmean(np.where(ok_first, np.floor_divide(fs, 3600) - np.floor_divide(ecg_s, 3600) == pd.to_numeric(d['first_af_event_delta_hours'], errors='coerce').to_numpy(float), np.nan)))
    out['dauer_s'] = round(time.time() - t0, 1); out['spitzen_rss_gb'] = round(G.peak() / 2**30, 2)
    (G.OUT / 'r4_gegen_roh.json').write_text(json.dumps(out, indent=1, default=str))
    # Aggregat-CSV (Kleinzellenregel)
    lines = []
    for fold, lab in FOLDS.items():
        for mode in ('archiv', 'exakt'):
            r = rows[fold][mode]
            for stage, c in (('vor Vollstaendigkeit', r['risk']), ('Complete Case', r['cc'])):
                tie = c['tied']; horizon = c['horizon_end']
                coarse = 0 < tie < 10
                hz = f">{(horizon // 10) * 10 - (10 if coarse else 0)}" if coarse else str(horizon)
                lines.append(dict(fold=lab, variante=mode, stufe=stage, personen=c['n'], af=K.kz(c['af']), kontaktende=K.kz(c['contact_end']),
                                  todesende=K.kz(c['death_end']), horizontende=hz, gleichstand=K.kz(tie),
                                  anteil_kontaktende_ohne_af=round(c['contact_end'] / c['no_af'], 4)))
    pd.DataFrame(lines).to_csv(PUB / 'S5_NACHTRAG_tabelle2.csv', index=False)
    # Gegenrechnung: Fluss, Ursachen, Zensus je Fold und Variante (Kleinzellenregel; Gesamtzahl der Landmarken der internen Pruefung gerundet)
    g = []; kl = []
    for fold, lab in FOLDS.items():
        for mode in ('archiv', 'exakt'):
            r = rows[fold][mode]; c = r['risk']; v = r['verluste']
            sel = rows[fold]['selected']
            g.append(dict(fold=lab, variante=mode, ausgewaehlte_landmarken=(str(sel) if fold == 'train' else '>9745'),
                          kompatibel=(str(r['kompatibel']) if fold == 'train' or mode == 'archiv' else '>9745'),
                          ohne_positive_zeit=v['lost'], davon_tod_am_entlassungstag=v['death_on_or_before_discharge_day'],
                          davon_kein_kontakt_nach_entlassung=v['no_contact_after_index_discharge'], davon_sonstige=v['other'],
                          risikopersonen=c['n'], af=K.kz(c['af']), kontaktende=K.kz(c['contact_end']), todesende=K.kz(c['death_end']),
                          horizontende=(str(c['horizon_end']) if not (0 < c['tied'] < 10) else '>' + str((c['horizon_end'] // 10) * 10 - 10)),
                          gleichstand=K.kz(c['tied']), anteil_kontaktende_ohne_af=round(r['anteil_kontaktende_ohne_af'], 4)))
            for k_, n_ in merge_small(list(r['obs_s_klassen'].items())):
                kl.append(dict(fold=lab, variante=mode, spanne_nach_entlassung_bis_letzter_kontakt=k_, risikopersonen=n_))
    pd.DataFrame(g).to_csv(PUB / 'S5_NACHTRAG_gegenrechnung.csv', index=False)
    pd.DataFrame(kl).to_csv(PUB / 'S5_NACHTRAG_beobachtungsspanne_klassen.csv', index=False)
    print(json.dumps({'dauer': out['dauer_s'], 'rss': out['spitzen_rss_gb']}))


if __name__ == '__main__':
    main()
