"""AW-243 Teil 2: Zaehlung der Zuordnungs- und Wertaenderungen, wenn Stunden-/Tagesgrenzen-Differenzen exakt gerechnet werden.

Nur Entwicklungsfolds (train, validation); locked_test wird nicht gelesen. Nur Zaehlungen (keine Ergebnisgroessen der
Fragestellungen). Eingaben: clinical_context_features.parquet (ausgewaehlte Spalten), patient_clinical_timeline.parquet,
admissions.parquet, S5-Landmarkenpipeline (Features-Cache nur Spalte rhythm_context).
Ausgabe: nachrechnung/AW243/AUDIT_ZEIT_zaehlung.csv (Kleinzellenregel), <local path>.
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
G.start('audit', rss_gb=1.3)
import numpy as np, pandas as pd, duckdb
import g243_kern as K

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / 'nachrechnung/AW243'; PUB.mkdir(parents=True, exist_ok=True)
EV = str(data_path("s5.mimic_validation_evidence"))
STD = str(data_path("s5.mimic_validation_standardized"))
CACHE = str(data_path("s5.single_marker_feature_cache_dir")) + "/split=*/features.parquet"
SPL = ('train', 'validation')
H = 3600
rows = []


def add(werk, stelle, groesse, fold, n_ecg_oder_zeilen, **extra):
    rows.append(dict(werk=werk, stelle=stelle, groesse=groesse, fold=fold, n=n_ecg_oder_zeilen, **extra))


def s(x):
    a, ok = K.secs(x)
    return np.where(ok, a, np.nan).astype('float64')


def main():
    t0 = time.time()
    con = duckdb.connect(); con.execute("SET memory_limit='1000MB'; SET threads=2")
    cols = ['subject_id', 'study_id', 'split', 'ecg_time', 'admittime', 'dischtime', 'followup_hours', 'vital_status_followup_hours', 'death_delta_days',
            'length_of_stay_days', 'first_af_event_delta_hours', 'next_af_event_delta_hours', 'af_dx_prior_admission', 'af_dx_index_admission',
            'current_report_af_state', 'next_conduction_event_delta_hours', 'next_device_status_delta_hours', 'conduction_prior_evidence',
            'conduction_index_untimed', 'current_report_lbbb_state', 'current_report_pacemaker_state', 'machine_report',
            'machine_metadata_usable_for_primary']
    q = f"SELECT {','.join('c.'+x for x in cols)} FROM read_parquet('{EV}/clinical_context_features.parquet') c WHERE c.split IN ('train','validation')"
    d = con.execute(q).df()
    tl = con.execute(f"SELECT subject_id, last_observed_time, vital_status_known_until, first_af_event_time, first_af_report_time, date_of_death FROM read_parquet('{EV}/patient_clinical_timeline.parquet')").df()
    d = d.merge(tl, on='subject_id', how='left', validate='many_to_one')
    split_ids = set(d.subject_id.unique())
    adm = con.execute(f"SELECT subject_id, hadm_id, admittime, dischtime FROM read_parquet('{STD}/admissions.parquet')").df()
    adm = adm[adm.subject_id.isin(split_ids)].copy()
    out = {'zeilen_entwicklung': {f: int((d.split == f).sum()) for f in SPL}}
    ecg = s(d['ecg_time'])
    # ---- 0. Paritaet der Archivspalten mit Stundengrenzen-/Tagesgrenzen-Rechnung
    lo = s(d['last_observed_time']); vk = s(d['vital_status_known_until']); fa = s(d['first_af_event_time'])
    fl = lambda a: np.floor_divide(a, H)
    m = ~np.isnan(lo)
    out['paritaet_followup_hours'] = float(np.mean(fl(lo[m]) - fl(ecg[m]) == d.followup_hours.to_numpy(float)[m]))
    m = ~np.isnan(vk)
    out['paritaet_vital_status_followup_hours'] = float(np.mean(fl(vk[m]) - fl(ecg[m]) == d.vital_status_followup_hours.to_numpy(float)[m]))
    m = ~np.isnan(fa)
    out['paritaet_first_af_event_delta_hours'] = float(np.mean(fl(fa[m]) - fl(ecg[m]) == d.first_af_event_delta_hours.to_numpy(float)[m]))
    dd = pd.to_datetime(d['dischtime']); ad = pd.to_datetime(d['admittime'])
    los_cal = d['length_of_stay_days'].to_numpy(float)
    los_dates = (dd.dt.normalize() - ad.dt.normalize()).dt.days.to_numpy(float)
    mm = ~np.isnan(los_cal) & ~np.isnan(los_dates)
    out['paritaet_length_of_stay_days_kalendertage'] = float(np.mean(los_cal[mm] == los_dates[mm]))
    # Sekundenanteile der Zeitstempel (sollten 0 sein)
    fr = con.execute(f"SELECT count(*) FILTER (WHERE epoch_us(ecg_time)%1000000<>0), count(*) FILTER (WHERE epoch_us(admittime)%1000000<>0), count(*) FILTER (WHERE epoch_us(dischtime)%1000000<>0) FROM read_parquet('{EV}/clinical_context_features.parquet') WHERE split IN ('train','validation')").fetchone()
    fr2 = con.execute(f"SELECT count(*) FILTER (WHERE epoch_us(last_observed_time)%1000000<>0), count(*) FILTER (WHERE epoch_us(first_af_event_time)%1000000<>0) FROM read_parquet('{EV}/patient_clinical_timeline.parquet')").fetchone()
    out['zeitstempel_mit_sekundenbruchteilen'] = dict(ecg_time=int(fr[0]), admittime=int(fr[1]), dischtime=int(fr[2]), last_observed_time=int(fr2[0]), first_af_event_time=int(fr2[1]))

    fu_arch = d.followup_hours.to_numpy(float) * H
    fu_ex = lo - ecg
    vs_arch = d.vital_status_followup_hours.to_numpy(float) * H
    vs_ex = vk - ecg
    death_cal = d.death_delta_days.to_numpy(float)
    folds = {f: (d.split == f).to_numpy() for f in SPL}

    # ---- 1. neues AF (labels.py:_af_rules): followup_hours >= 24*H -> negativ/eligible
    cur_prior = ((d.current_report_af_state == 'positive').fillna(False).to_numpy() |
                 (d['first_af_report_time'].notna() & (d['first_af_report_time'] <= d['ecg_time'])).to_numpy() |
                 d.af_dx_prior_admission.fillna(False).astype(bool).to_numpy())
    idx_only = d.af_dx_index_admission.fillna(False).astype(bool).to_numpy()
    nxt = d.next_af_event_delta_hours.to_numpy(float)
    for days in (30, 90, 365):
        hh = days * 24
        within = (nxt > 0) & (nxt <= hh)
        dth = (death_cal >= 0) & (death_cal <= days - 1)
        reach = ~cur_prior & ~within & ~idx_only & ~dth
        a_neg = fu_arch >= hh * H; e_neg = fu_ex >= hh * H
        a_neg = a_neg & ~np.isnan(fu_arch); e_neg = e_neg & ~np.isnan(fu_ex)
        for f, mf in folds.items():
            r = reach & mf
            add('R1 (Endpunkt new_onset_af_%dd); S5 nutzt die Spalte direkt (siehe Gegenrechnung)' % days, 'labels.py:1409-1421 / evidence.py:4240', 'Zustand negativ vs. unbestimmt (followup_hours >= %d h)' % hh, f, int(r.sum()),
                aenderung_archiv_negativ_exakt_nicht=int((r & a_neg & ~e_neg).sum()), aenderung_archiv_nicht_exakt_negativ=int((r & ~a_neg & e_neg).sum()),
                tor_ueberhaupt_erreicht=int(r.sum()))
    # ---- 2. Leitungsfortschritt 1 Jahr
    cp = ((d.current_report_lbbb_state == 'positive').fillna(False).to_numpy() | (d.current_report_pacemaker_state == 'positive').fillna(False).to_numpy() |
          d.conduction_prior_evidence.fillna(False).astype(bool).to_numpy())
    ncd = d.next_conduction_event_delta_hours.to_numpy(float); ndv = d.next_device_status_delta_hours.to_numpy(float)
    within = (ncd > 0) & (ncd <= 8760); dev = (ndv >= 0) & (ndv <= 8760)
    dth = (death_cal >= 0) & (death_cal <= 364)
    reach = ~cp & ~within & ~d.conduction_index_untimed.fillna(False).astype(bool).to_numpy() & ~dev & ~dth
    a_neg = (fu_arch >= 8760 * H) & ~np.isnan(fu_arch); e_neg = (fu_ex >= 8760 * H) & ~np.isnan(fu_ex)
    for f, mf in folds.items():
        r = reach & mf
        add('PUB-13 (Referenzlabel cond1y); Stufenakten (conduction_progression_1y)', 'labels.py:1563-1576 / evidence.py:4240', 'Zustand negativ vs. unbestimmt (followup_hours >= 8760 h)', f, int(r.sum()),
            aenderung_archiv_negativ_exakt_nicht=int((r & a_neg & ~e_neg).sum()), aenderung_archiv_nicht_exakt_negativ=int((r & ~a_neg & e_neg).sum()), tor_ueberhaupt_erreicht=int(r.sum()))
    # ---- 3. Mortalitaet: vital_status_followup_hours >= 24*H nur bei death_delta_days IS NULL
    for days in (30, 90, 365):
        nulldeath = np.isnan(death_cal)
        a_neg = (vs_arch >= days * 24 * H) & ~np.isnan(vs_arch); e_neg = (vs_ex >= days * 24 * H) & ~np.isnan(vs_ex)
        for f, mf in folds.items():
            r = nulldeath & mf
            add(('PUB-13 (Referenzlabel mort30); A155-Assoziationsdomaene outcome_30d_mortality' if days == 30 else 'A155-Assoziationsdomaene outcome_%dd_mortality; thesis_tests (8760 h)' % days), 'labels.py:1911-1938, association_domains.py:515-580 / evidence.py:4242',
                'negativ/eligible vs. unbestimmt (vital_status_followup_hours >= %d h)' % (days * 24), f, int(r.sum()),
                aenderung_archiv_ja_exakt_nein=int((r & a_neg & ~e_neg).sum()), aenderung_archiv_nein_exakt_ja=int((r & ~a_neg & e_neg).sum()), tor_ueberhaupt_erreicht=int(r.sum()))
    # Grenzfaelle Tagespraezision des Todes (kein Fehler, Konvention): death_delta_days == Horizont
    for days in (30, 90, 365):
        for f, mf in folds.items():
            add('PUB-13 (Tod 365 d, death_day), Referenzlabel mort30, A155-Mortalitaetsendpunkte', 'evidence.py:4244, labels.py:1915; PUB-13 k1_auswertung.py:1018', 'Tod genau am Kalendertag %d nach dem EKG-Tag (Tagespraezision, Grenzfall)' % days, f,
                int(((death_cal == days) & mf).sum()), hinweis='Konvention Kalendertage; Todeszeitpunkt innerhalb des Tages unbekannt')
    # ---- 4. Verweildauer (Kalendertage)
    ex_los = (dd - ad).dt.total_seconds().to_numpy(float) / 86400.0
    mm = ~np.isnan(los_cal) & ~np.isnan(ex_los)
    for f, mf in folds.items():
        r = mm & mf
        diff = los_cal - ex_los
        add('A155-Assoziationsdomaene outcome_length_of_stay (kein Werk der sechs)', 'evidence.py:4246 / association_domains.py:593-600', 'length_of_stay_days (Tagesgrenzen) vs. exakte Tage', f, int(r.sum()),
            ungleich_floor_exakt=int((r & (los_cal != np.floor(ex_los))).sum()), abweichung_ueber_0_5_tage=int((r & (np.abs(diff) > 0.5)).sum()),
            abweichung_ueber_0_99_tage=int((r & (np.abs(diff) > 0.99)).sum()))
    # ---- 5. 30-Tage-Wiederaufnahme
    a = adm.sort_values(['subject_id', 'admittime', 'hadm_id']).copy()
    a['next_admit'] = a.groupby('subject_id')['admittime'].shift(-1)
    a = a[a.next_admit.notna()]
    cal = (a.next_admit.dt.normalize() - a.dischtime.dt.normalize()).dt.days
    ex_d = (a.next_admit - a.dischtime).dt.total_seconds() / 86400.0
    flag_arch = (cal >= 0) & (cal <= 30); flag_ex = (ex_d >= 0) & (ex_d <= 30)
    spl_by_subject = d.drop_duplicates('subject_id').set_index('subject_id')['split']
    a['split'] = a.subject_id.map(spl_by_subject)
    for f in SPL:
        r = (a.split == f).to_numpy()
        add('A155-Assoziationsdomaene outcome_readmission_30d (kein Werk der sechs)', 'evidence.py:3897 / association_domains.py:605-615', 'Wiederaufnahme innerhalb 30 Tage (Tagesgrenzen) vs. exakt', f, int(r.sum()),
            archiv_ja_exakt_nein=int((r & flag_arch.to_numpy() & ~flag_ex.to_numpy()).sum()), archiv_nein_exakt_ja=int((r & ~flag_arch.to_numpy() & flag_ex.to_numpy()).sum()),
            davon_naechste_aufnahme_vor_entlassung=int((r & (ex_d < 0).to_numpy()).sum()))
    # ---- 6. survival.py: Kalendertage bis letzter Kontakt (Zensur am Horizont)
    lo_dt = pd.to_datetime(d['last_observed_time']); ec_dt = pd.to_datetime(d['ecg_time'])
    cal_last = (lo_dt.dt.normalize() - ec_dt.dt.normalize()).dt.days.to_numpy(float)
    ex_last = fu_ex / 86400.0
    for days in (30, 90, 365):
        a_ = cal_last >= days; e_ = ex_last >= days
        for f, mf in folds.items():
            add('A155-Forschungsmodule und Stufenakten (survival.follow_up; kein Werk der sechs)', 'survival.py:206-215', 'days_to_last_contact >= %d (Kalendertage) vs. exakt' % days, f, int(mf.sum()),
                archiv_ja_exakt_nein=int((mf & a_ & ~e_).sum()), archiv_nein_exakt_ja=int((mf & ~a_ & e_).sum()))
    # ---- 7. S5: Tor 5 (first_af_event_delta_hours > Landmarke) auf Kandidatenebene und Auswahl je Person
    cache = con.execute(f"SELECT subject_id, study_id, rhythm_context FROM read_parquet('{CACHE}', hive_partitioning=true) WHERE split IN ('train','validation')").df()
    c2 = d.merge(cache, on=['subject_id', 'study_id'], how='inner', validate='one_to_one')
    lm_s = s(c2['dischtime']) - s(c2['ecg_time'])
    f_arch = c2.first_af_event_delta_hours.to_numpy(float) * H
    f_ex = s(c2['first_af_event_time']) - s(c2['ecg_time'])
    base = (c2.machine_metadata_usable_for_primary.eq(True) & c2.machine_report.map(K.report_sinus) & c2.rhythm_context.eq('regular')
            & c2.af_dx_prior_admission.eq(False) & c2.af_dx_index_admission.eq(False) & c2.current_report_af_state.eq('not_mentioned')).to_numpy()
    base &= (lm_s >= 0) & (lm_s <= 168 * H)
    out['kandidaten_ecg_ohne_tor5'] = {f: int((base & (c2.split == f).to_numpy()).sum()) for f in SPL}
    p_a = base & (np.isnan(f_arch) | (f_arch > lm_s)); p_e = base & (np.isnan(f_ex) | (f_ex > lm_s))

    def choose(mask):
        z = c2.loc[mask, ['subject_id', 'study_id', 'split']].copy()
        z['dischtime'] = c2.loc[mask, 'dischtime']; z['ecg_time'] = c2.loc[mask, 'ecg_time']
        z = z.sort_values(['subject_id', 'dischtime', 'ecg_time', 'study_id'], ascending=[True, True, False, True], kind='stable').drop_duplicates('subject_id')
        return z.set_index('subject_id')
    ca, ce = choose(p_a), choose(p_e)
    for f in SPL:
        A_ = ca[ca.split == f]; E_ = ce[ce.split == f]
        both = A_.index.intersection(E_.index)
        diff_study = int((A_.loc[both, 'study_id'] != E_.loc[both, 'study_id']).sum())
        add('S5 (Landmarkenauswahl)', 'a155_research_af_landmark_revision002.py:14-17 / evidence.py:4238', 'Tor 5 first_af_event_delta_hours > Landmarke: Kandidaten-EKG, die Ergebnis wechseln', f,
            int((base & (c2.split == f).to_numpy()).sum()), ekg_nur_archiv_besteht=int((p_a & ~p_e & (c2.split == f).to_numpy()).sum()), ekg_nur_exakt_besteht=int((~p_a & p_e & (c2.split == f).to_numpy()).sum()),
            personen_ausgewaehlt_archiv=len(A_), personen_ausgewaehlt_exakt=len(E_), personen_nur_archiv=len(A_.index.difference(E_.index)),
            personen_nur_exakt=len(E_.index.difference(A_.index)), personen_anderes_ekg_gewaehlt=diff_study)
    out['auswahl_paritaet_archiv'] = {f: int((ca.split == f).sum()) for f in SPL}
    out['dauer_s'] = round(time.time() - t0, 1); out['rss_gb'] = round(G.peak() / 2**30, 2)
    (G.OUT / 'audit_roh.json').write_text(json.dumps(out, indent=1, default=str))
    df = pd.DataFrame(rows)
    df.to_csv(G.OUT / 'audit_zaehlung_roh.csv', index=False)
    print(json.dumps(out, default=str))


if __name__ == '__main__':
    main()
