"""AW-255 S3 Machbarkeit (MC-MED, nur Train): reine Zaehlung von Verfuegbarkeit und Zeitstruktur. KEINE Assoziation Elektrolyt-EKG.

Voraussetzung: s3_laden.py (privater Bestand <local path>) und Tabelle seg (Lead-II-Segmentkoepfe aus s5_privat.duckdb, Train).
Ausgabe: maskierte Zaehlungen (Kleinzellenregel: 1 bis 9 als "<10") als JSON; keine Kennungen.
Aufruf: <local path> -I s3_zaehlung.py <ausgabe.json>

Definitionen (vor der Rechnung festgelegt, Auftrag AW-255):
  Entnahme = je Besuch und Auftragszeit (Order_time) ein Wert; Entnahmezeit selbst ist in MC-MED nicht enthalten (nur Order_time und Result_time).
  Kalium primaer: Komponente POTASSIUM (Serum/Plasma). Sensitivitaet "alle": zusaetzlich POC:POTASSIUM, ISTAT und POTASSIUM.WHOLE BLD.
  Zensierte Werte ('>10.0') als 10,0 gefuehrt, '<' Werte entfallen; widerspruechliche Werte derselben Auftragszeit entfallen.
  Paar = zwei Entnahmen desselben Besuchs mit Abstand der Auftragszeiten von 1 bis 12 h. "konsekutiv": je erster Entnahme nur die naechste Entnahme.
  Zeitfenster EKG: D1 = +-15 min um die Auftragszeit; D2 = +-15 min um die Mitte zwischen Auftrags- und Ergebniszeit; D3 = von 15 min vor der
  Auftragszeit bis 15 min nach der Ergebniszeit (grosszuegig, Entnahmezeit unbekannt). Zeitlich ausreichend = mindestens 10 s Ableitung II in einem Segment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, json
sys.dont_write_bytecode = True
import duckdb
from pathlib import Path

W = Path(str(data_path("s3.work_dir")))
c = duckdb.connect(str(W / 's3.duckdb'))
c.execute("SET memory_limit='700MB'; SET threads=2")
q = lambda s: c.execute(s).fetchall()
one = lambda s: c.execute(s).fetchone()
if 'seg' not in [r[0] for r in q('show tables')]:
    c.execute(("ATTACH '" + str(data_path("s3.s5_private_duckdb")) + "' AS s5 (READ_ONLY)"))
    c.execute("create table seg as select h.CSN, h.seg, h.fs, h.n, h.t0, h.t1 from s5.hd2 h join sp using(CSN)")
    c.execute("create table hrmon as select n.CSN, n.t, n.Value hr from s5.num n join sp using(CSN) where n.Source='Monitor' and n.Measure='HR'")


def maske(x):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, int) and 1 <= x <= 9:
        return '<10'
    if isinstance(x, dict):
        return {k: maske(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [maske(v) for v in x]
    return x


def mk_draws(name, comps):
    inl = ",".join(f"'{x}'" for x in comps)
    c.execute(f"""create or replace table {name} as with x as (select CSN, t_order, t_result, cname, cval,
        try_cast(regexp_replace(cval,'^[<>]','') as double) k, (cval like '>%') cens from labs where cname in ({inl})
        and try_cast(regexp_replace(cval,'^[<>]','') as double) is not null and cval not like '<%')
        select CSN, t_order, min(t_result) t_result, count(distinct k) nk, avg(k) k, bool_or(cens) cens, string_agg(distinct cname,'|') src from x group by CSN, t_order""")
    c.execute(f"delete from {name} where nk>1")


def mk_cov(tbl):
    c.execute(f"""create or replace table {tbl}_cov as
      with d as (select *, coalesce((t_order+t_result)/2, t_order+3600) mid from {tbl}),
      o as (select d.CSN, d.t_order,
        greatest(0, least(s.t1, d.t_order+900) - greatest(s.t0, d.t_order-900)) o1,
        greatest(0, least(s.t1, d.mid+900) - greatest(s.t0, d.mid-900)) o2,
        greatest(0, least(s.t1, greatest(coalesce(d.t_result,d.t_order+3600),d.t_order)+900) - greatest(s.t0, least(d.t_order,coalesce(d.t_result,d.t_order))-900)) o3
        from d join seg s on s.CSN=d.CSN and s.t1>=least(d.t_order,coalesce(d.t_result,d.t_order))-900 and s.t0<=greatest(coalesce(d.t_result,d.t_order+3600),d.t_order)+900)
      select d.CSN, d.t_order, d.t_result, d.mid, d.k, d.cens, coalesce(max(o.o1),0) mx_d1, coalesce(max(o.o2),0) mx_d2, coalesce(max(o.o3),0) mx_d3
      from d left join o on o.CSN=d.CSN and o.t_order=d.t_order group by d.CSN,d.t_order,d.t_result,d.mid,d.k,d.cens""")


def mk_pairs(n):
    c.execute(f"""create or replace table {n}_pairs as select a.CSN, a.t_order ta, b.t_order tb, a.k ka, b.k kb, b.k-a.k dk, (b.t_order-a.t_order)/3600 gap_h
      from {n} a join {n} b on a.CSN=b.CSN and b.t_order>a.t_order where (b.t_order-a.t_order)/3600 between 1 and 12""")
    c.execute(f"""create or replace table {n}_cons as select * from (select a.CSN, a.t_order ta, b.t_order tb, a.k ka, b.k kb, b.k-a.k dk, (b.t_order-a.t_order)/3600 gap_h,
       row_number() over (partition by a.CSN, a.t_order order by b.t_order) rn from {n} a join {n} b on a.CSN=b.CSN and b.t_order>a.t_order) where rn=1 and gap_h between 1 and 12""")
    for kind in ('pairs', 'cons'):
        t = f"{n}_{kind}"
        c.execute(f"""create or replace table {t}_c as select p.*, v.MRN, a.mx_d1 a1,b.mx_d1 b1,a.mx_d2 a2,b.mx_d2 b2,a.mx_d3 a3,b.mx_d3 b3 from {t} p
          join {n}_cov a on a.CSN=p.CSN and a.t_order=p.ta join {n}_cov b on b.CSN=p.CSN and b.t_order=p.tb join visit v on v.CSN=p.CSN""")


R = {'version': 's3_zaehlung v1 (AW-255)', 'population': 'MC-MED split_random_train (Test nie geoeffnet)'}
R['besuche_train'] = one('select count(*), count(distinct MRN) from visit')
R['besuche_mit_ableitung_II'] = one('select count(distinct CSN) from seg')

# --- Frage 1: Labore
R['labore'] = {}
for name, comps in {'kalium_serum': ['POTASSIUM'], 'kalium_poc_istat': ['POC:POTASSIUM, ISTAT'], 'kalium_vollblut': ['POTASSIUM.WHOLE BLD'],
                    'natrium': ['SODIUM'], 'natrium_poc': ['POC:SODIUM, ISTAT'], 'calcium_gesamt_mgdl': ['CALCIUM'],
                    'calcium_ionisiert': ['CALCIUM, IONIZED'], 'calcium_ionisiert_poc': ['POC:CALCIUM,ION, ISTAT'], 'magnesium_mgdl': ['MAGNESIUM'],
                    'bikarbonat_co2': ['CO2'], 'phosphat': ['PHOSPHORUS'], 'ph_venoes': ['PH.VENOUS'], 'ph_arteriell': ['PH.ARTERIAL'],
                    'laktat_poc': ['POC:LACTATE, ISTAT'], 'kreatinin': ['CREATININE'], 'troponin_hs': ['HIGH SENSITIVITY TROPONIN']}.items():
    inl = ",".join(f"'{x}'" for x in comps)
    r = one(f"select count(*), count(distinct CSN), sum((try_cast(cval as double) is not null)::int), quantile_cont((t_result-t_order)/60,[.05,.5,.95]) from labs where cname in ({inl})")
    R['labore'][name] = {'zeilen': r[0], 'besuche': r[1], 'numerisch': r[2], 'ergebnis_minus_auftrag_min_q05_50_95': r[3]}
R['zeitstempel'] = {'vorhanden': ['Order_time', 'Result_time'], 'entnahmezeit': 'nicht vorhanden',
                    'anteil_poc_istat_ergebnis_vor_auftrag': one("select avg(((t_result-t_order)<0)::int) from labs where cname='POC:POTASSIUM, ISTAT'")[0],
                    'anteil_k_auftrag_vor_roomed': one("select avg((l.t_order<v.r)::int) from labs l join visit v on v.CSN=l.CSN where l.cname='POTASSIUM' and v.r is not null")[0]}

# --- Frage 2 und 3: Kaliumpaare und EKG-Abdeckung (zeitlich)
mk_draws('kdraw_serum', ['POTASSIUM'])
mk_draws('kdraw_all', ['POTASSIUM', 'POC:POTASSIUM, ISTAT', 'POTASSIUM.WHOLE BLD'])
R['kalium'] = {}
for n in ('kdraw_serum', 'kdraw_all'):
    mk_cov(n)
    mk_pairs(n)
    S = {}
    S['entnahmen'], S['besuche'] = one(f'select count(*), count(distinct CSN) from {n}')
    S['besuche_mit_mind_2_entnahmen'] = one(f'select count(*) from (select CSN from {n} group by 1 having count(*)>=2)')[0]
    S['entnahmen_mit_ekg_zeitlich'] = dict(zip(['D1', 'D2', 'D3'], one(f'select sum((mx_d1>=10)::int), sum((mx_d2>=10)::int), sum((mx_d3>=10)::int) from {n}_cov')))
    for kind in ('cons', 'pairs'):
        t = f'{n}_{kind}_c'
        P = {'paare': one(f'select count(*), count(distinct CSN), count(distinct MRN) from {t}')}
        for thr in (0.0, 0.5, 1.0):
            Q = {'alle_paare_besuche_personen': one(f'select count(*), count(distinct CSN), count(distinct MRN) from {t} where abs(dk)>={thr}'),
                 'richtung_abfall_anstieg': one(f'select count(*) filter (where dk<=-{max(thr, 1e-9)}), count(*) filter (where dk>={max(thr, 1e-9)}) from {t} where abs(dk)>={thr}')}
            for d in (1, 2, 3):
                Q[f'ekg_zeitlich_beidseits_D{d}_paare_besuche_personen'] = one(f'select count(*), count(distinct CSN), count(distinct MRN) from {t} where abs(dk)>={thr} and a{d}>=10 and b{d}>=10')
            P[f'dK_ab_{thr}'] = Q
        S[kind] = P
    S['dK_konsekutiv_quantile_05_25_50_75_95'] = one(f'select quantile_cont(dk,[.05,.25,.5,.75,.95]) from {n}_cons')[0]
    S['abstand_h_konsekutiv_quantile_05_50_95'] = one(f'select quantile_cont(gap_h,[.05,.5,.95]) from {n}_cons')[0]
    S['ekg_anteil_D1_nach_rang'] = q(f"""with r as (select cv.*, row_number() over (partition by CSN order by t_order) rn, count(*) over (partition by CSN) cnt from {n}_cov cv)
       select case when cnt=1 then 'einzige' when rn=1 then 'erste_von_mind_2' else 'spaetere' end g, count(*), avg((mx_d1>=10)::int) from r group by 1""")
    S['erste_entnahme_K_bei_abfall_ab_1'] = one(f'select avg((ka>=5.5)::int), avg((ka>=6.0)::int), count(*) from {n}_cons where dk<=-1')
    S['erste_entnahme_K_bei_anstieg_ab_1'] = one(f'select avg((ka<=3.5)::int), count(*) from {n}_cons where dk>=1')
    R['kalium'][n] = S

# --- Frage 4: Extremfaelle
CL = "('insulin_regular','glukose_bolus','calcium_iv','bikarbonat_iv','albuterol','k_binder')"
c.execute("""create or replace table trt as select CSN, t_admin t,
  case when regexp_matches(upper(pname),'INSULIN REGULAR|INSULIN, REGULAR') then 'insulin_regular'
       when regexp_matches(upper(pname),'INSULIN') then 'insulin_andere'
       when regexp_matches(upper(pname),'DEXTROSE 50|D50W|DEXTROSE 20|D20W') then 'glukose_bolus'
       when regexp_matches(upper(pname),'CALCIUM GLUC|CALCIUM CHLORIDE') then 'calcium_iv'
       when regexp_matches(upper(pname),'SODIUM BICARBONATE (8.4|IVPB)|BICARBONATE IV|D5W WITH SODIUM BICARBONATE') then 'bikarbonat_iv'
       when regexp_matches(upper(pname),'ALBUTEROL') then 'albuterol'
       when regexp_matches(upper(pname),'POLYSTYRENE|ZIRCONIUM|PATIROMER|LOKELMA') then 'k_binder'
       when regexp_matches(upper(pname),'FUROSEMIDE|BUMETANIDE|TORSEMIDE') then 'schleifendiuretikum'
       when regexp_matches(upper(pname),'POTASSIUM CHLORIDE|POTASSIUM PHOSPHATE|POTASSIUM, SODIUM PHOSPHATES|POTASSIUM CHLORID') then 'kalium_gabe'
       when regexp_matches(upper(pname),'MAGNESIUM SULFATE') then 'magnesium_iv' end cls
  from orders where Order_type='Medications' and t_admin is not null""")
c.execute("delete from trt where cls is null")
R['behandlungen_gesamt_mit_gabezeit'] = q('select cls, count(*), count(distinct CSN) from trt group by 1 order by 2 desc')
R['extremwerte'] = {}
for n in ('kdraw_serum', 'kdraw_all'):
    E = {}
    for thr in (6.0, 6.5):
        c.execute(f"""create or replace table ext as select e.CSN, e.t_order, e.k, cv.mx_d1, cv.mx_d2, cv.mx_d3, v.MRN,
           (select min(t_order) from {n} x where x.CSN=e.CSN and x.t_order>e.t_order) tnext_any,
           (select min(t_order) from {n} x where x.CSN=e.CSN and x.t_order>e.t_order+3600 and x.t_order<=e.t_order+12*3600) tnext_1_12
           from {n} e join {n}_cov cv using(CSN,t_order) join visit v using(CSN) where e.k>={thr}""")
        c.execute(f"""create or replace table ext3 as select e.*, kn.k k_next, cn.mx_d1 nx1, cn.mx_d3 nx3,
            (select count(*) from trt t where t.CSN=e.CSN and t.t>e.t_order-3600 and t.t<=e.tnext_1_12 and t.cls in {CL}) n_trt
            from ext e left join {n} kn on kn.CSN=e.CSN and kn.t_order=e.tnext_1_12 left join {n}_cov cn on cn.CSN=e.CSN and cn.t_order=e.tnext_1_12""")
        X = {'entnahmen_besuche_personen': one('select count(*), count(distinct CSN), count(distinct MRN) from ext')}
        for d in (1, 2, 3):
            X[f'ekg_zeitlich_D{d}_entnahmen_besuche'] = one(f'select count(*), count(distinct CSN) from ext where mx_d{d}>=10')
        X['spaeterer_wert_beliebig_besuche'] = one('select count(distinct CSN) from ext where tnext_any is not null')[0]
        X['spaeterer_wert_1_12h_besuche'] = one('select count(distinct CSN) from ext where tnext_1_12 is not null')[0]
        for d in (1, 3):
            X[f'ekg_D{d}_und_spaeterer_wert_1_12h_besuche'] = one(f'select count(distinct CSN) from ext3 where mx_d{d}>=10 and tnext_1_12 is not null')[0]
            X[f'ekg_D{d}_an_beiden_enden_besuche'] = one(f'select count(distinct CSN) from ext3 where mx_d{d}>=10 and nx{d}>=10')[0]
            X[f'ekg_D{d}_an_beiden_enden_mit_dokumentierter_k_senkender_behandlung'] = one(f'select count(distinct CSN) from ext3 where mx_d{d}>=10 and nx{d}>=10 and n_trt>0')[0]
        X['spaeterer_wert_1_12h_dK_quantile_10_50_90'] = one('select quantile_cont(k_next-k,[.1,.5,.9]) from ext3 where tnext_1_12 is not null')[0]
        X['spaeterer_wert_1_12h_mit_behandlung_ohne_behandlung'] = one('select count(*) filter (where n_trt>0), count(*) filter (where n_trt=0) from ext3 where tnext_1_12 is not null')
        X['ohne_behandlung_und_spaeterer_wert_unter_5_5'] = one('select count(*) from ext3 where tnext_1_12 is not null and n_trt=0 and k_next<5.5')[0]
        E[f'K_ab_{thr}'] = X
    R['extremwerte'][n] = E

# --- Frage 6: Stoergroessen bei konsekutiven Serumpaaren
for n in ('kdraw_serum', 'kdraw_all'):
    t = f'{n}_cons_c'
    c.execute(f"""create or replace table {t}_t as select p.*,
       (select count(*) from trt x where x.CSN=p.CSN and x.t>p.ta-1800 and x.t<=p.tb and x.cls in {CL}) n_kl,
       (select count(*) from trt x where x.CSN=p.CSN and x.t>p.ta-1800 and x.t<=p.tb and x.cls='kalium_gabe') n_kgabe,
       (select median(hr) from hrmon h where h.CSN=p.CSN and h.t between p.ta-900 and p.ta+900) hr_a,
       (select median(hr) from hrmon h where h.CSN=p.CSN and h.t between p.tb-900 and p.tb+900) hr_b from {t} p""")
    S = {}
    S['paare'] = one(f'select count(*) from {t}_t')[0]
    S['mit_k_senkender_behandlung_zwischen'] = one(f'select count(*) filter (where n_kl>0) from {t}_t')[0]
    S['abfall_ab_1_gesamt_mit_behandlung'] = one(f'select count(*) filter (where dk<=-1), count(*) filter (where dk<=-1 and n_kl>0) from {t}_t')
    S['anstieg_ab_1_gesamt_mit_kaliumgabe'] = one(f'select count(*) filter (where dk>=1), count(*) filter (where dk>=1 and n_kgabe>0) from {t}_t')
    for cl in ('insulin_regular', 'calcium_iv', 'bikarbonat_iv', 'albuterol', 'schleifendiuretikum', 'kalium_gabe'):
        S[f'paare_mit_{cl}'] = one(f"select count(*) from {t}_t p where exists (select 1 from trt x where x.CSN=p.CSN and x.t>p.ta-1800 and x.t<=p.tb and x.cls='{cl}')")[0]
    for d in (1, 3):
        for thr in (0.5, 1.0):
            S[f'ekg_zeitlich_D{d}_dK_ab_{thr}'] = one(f"""select count(*) filter (where a{d}>=10 and b{d}>=10 and abs(dk)>={thr}),
                count(*) filter (where a{d}>=10 and b{d}>=10 and abs(dk)>={thr} and n_kl=0 and n_kgabe=0),
                count(*) filter (where a{d}>=10 and b{d}>=10 and abs(dk)>={thr} and n_kl>0),
                count(*) filter (where a{d}>=10 and b{d}>=10 and abs(dk)>={thr} and hr_a is not null and hr_b is not null),
                count(*) filter (where a{d}>=10 and b{d}>=10 and abs(dk)>={thr} and abs(hr_b-hr_a)>=10) from {t}_t""")
    S['dHF_monitor_quantile_05_25_50_75_95_und_n'] = one(f'select quantile_cont(hr_b-hr_a,[.05,.25,.5,.75,.95]), count(*) from {t}_t where hr_a is not null and hr_b is not null')
    S['reihenfolge_felder_ekg_zeitlich'] = 'n_gesamt, ohne_behandlung_und_kaliumgabe, mit_behandlung, mit_HF_beidseits, abs_dHF_ab_10'
    R['stoergroessen_' + n] = S

# --- weitere Elektrolyte (beliebige Paare 1 bis 12 h, EKG zeitlich D1)
weit = {'natrium': (['SODIUM', 'POC:SODIUM, ISTAT', 'SODIUM.WHOLE BLD'], 5.0), 'calcium_gesamt': (['CALCIUM'], 0.5),
        'calcium_ionisiert': (['CALCIUM, IONIZED', 'POC:CALCIUM,ION, ISTAT'], 0.10), 'magnesium': (['MAGNESIUM'], 0.3),
        'bikarbonat_co2': (['CO2'], 4.0), 'phosphat': (['PHOSPHORUS'], 1.0), 'ph': (['PH.VENOUS', 'PH.ARTERIAL', 'PH.SER/PLAS', 'POC:PH, ISTAT'], 0.05)}
R['weitere_elektrolyte'] = {}
for name, (cl, thr) in weit.items():
    inl = ",".join(f"'{x}'" for x in cl)
    c.execute(f"""create or replace table e_{name} as with x as (select CSN,t_order,t_result,try_cast(cval as double) k from labs where cname in ({inl}) and try_cast(cval as double) is not null)
       select CSN,t_order,min(t_result) t_result,avg(k) k from x group by 1,2""")
    c.execute(f"""create or replace table e_{name}_p as select a.CSN, a.t_order ta, b.t_order tb, b.k-a.k dk from e_{name} a join e_{name} b
       on a.CSN=b.CSN and b.t_order>a.t_order and (b.t_order-a.t_order)/3600 between 1 and 12""")
    c.execute(f"""create or replace table cv as with p as (select * from e_{name}_p where abs(dk)>={thr}), ends as (select CSN, ta t from p union select CSN, tb from p),
      o as (select e.CSN, e.t, max(greatest(0, least(s.t1, e.t+900) - greatest(s.t0, e.t-900))) mx from ends e join seg s on s.CSN=CAST(e.CSN as BIGINT) and s.t1>=e.t-900 and s.t0<=e.t+900 group by 1,2)
      select p.CSN, coalesce(oa.mx,0) a, coalesce(ob.mx,0) b from p left join o oa on oa.CSN=p.CSN and oa.t=p.ta left join o ob on ob.CSN=p.CSN and ob.t=p.tb""")
    R['weitere_elektrolyte'][name] = {'schwelle': thr, 'messungen_besuche': one(f'select count(*), count(distinct CSN) from e_{name}'),
                                       'paare_1_12h_besuche': one(f'select count(*), count(distinct CSN) from e_{name}_p'),
                                       'paare_ab_schwelle_besuche': one(f'select count(*), count(distinct CSN) from e_{name}_p where abs(dk)>={thr}'),
                                       'paare_ab_schwelle_mit_ekg_D1_beidseits': one('select count(*) from cv where a>=10 and b>=10')[0]}

out = Path(sys.argv[1]) if len(sys.argv) > 1 else W / 'zaehlung_s3.json'
out.write_text(json.dumps(maske(R), indent=1, ensure_ascii=False, default=str), encoding='utf-8')
print('geschrieben', out)
