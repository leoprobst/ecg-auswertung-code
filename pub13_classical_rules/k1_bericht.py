"""PUB-13 K1 (AW-236): erzeugt aus den Einheitenergebnissen (<local path><modus>\\results.json) einen Markdown-Bericht
mit allen Urteilen, Schaetzern, Intervallen und Sekundaeranalysen. Anzahlen gehen durch die Kleinzellenregel des Auswertungsskripts
(`masked_copy`), es erscheinen keine Kennungen. Dieses Werkzeug rechnet nichts und gehoert nicht zum eingefrorenen Skript.

Aufruf: python -I k1_bericht.py <modus dev|test> <ausgabe.md> [Ergebnisordner]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k1_auswertung as K  # noqa: E402

ORDER = ["R02", "R03", "R04", "R06", "R07", "R14", "R20_A1", "R20_A2", "R09", "R18", "R08_VTVF", "R08_Tod", "R19"]
KL = {"haelt": "hält", "haelt nicht": "hält nicht", "unentschieden": "unentschieden"}


def f(x, nd=3):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    if isinstance(x, str):
        return x
    if isinstance(x, float) and x in (float("inf"), float("-inf")):
        return "∞" if x > 0 else "-∞"
    return f"{x:.{nd}f}"


def kl(c):
    return KL.get(c, c)


def hint_txt(h):
    return "; ".join(h) if h else "-"


def sect_e(uid, u, L):
    if u.get("status") != "ok":
        L.append(f"Status: {u.get('status')}")
        return
    a, c = u["abdeckung"], u["counts"]
    L.append(f"Population {c['n_pop']} Personen (von {a['n_index']} Index-EKG; Regel nicht auswertbar in der Basis: {a['n_regel_NULL_in_basis']}), "
             f"verifiziert {c['n_verified']}, Referenzpositive unter den Verifizierten {c['cases']}.")
    L.append("")
    L.append("| Komponente | Behauptung | Schätzer | Schranken (α/m) | 95 % unadjustiert | Klasse | Hinweise |")
    L.append("|---|---|---|---|---|---|---|")
    for k in u["judge"]["components"]:
        L.append(f"| {k['id']} | {k['claim']} | {f(k['estimate'])} | [{f(k['lo'])}; {f(k['hi'])}] (α/m={k['alpha_c']:.5f}) | [{f(k['lo95'])}; {f(k['hi95'])}] | {kl(k['klasse'])} | {hint_txt(k['hinweise'])} |")
    L.append("")
    L.append(f"Konkordanzvariante: Einheitsklasse {kl(u['concordance']['variante_klasse'])}. **Endurteil: {kl(u['final']['klasse'])}** (Primär {kl(u['final']['primaer'])}); Hinweise: {hint_txt(u['final']['hinweise'])}.")
    kp = u["judge"].get("kipp")
    if kp:
        cl = {g: v["unit_klasse"] for g, v in kp.items() if g != "_robust"}
        L.append(f"Kipppunktraster 3 × 3 (κ₊ | κ₋): Klassen {dict(sorted(cl.items()))}; robust = {kp['_robust']}.")
    m = u["metrics"]
    for t, mm in m.items():
        L.append(f"- Variante {t}: Sens {f(mm['sens']['est'])} [{f(mm['sens']['lo95'])}; {f(mm['sens']['hi95'])}], Spez {f(mm['spec']['est'])} [{f(mm['spec']['lo95'])}; {f(mm['spec']['hi95'])}], "
                 f"PPV {f(mm['ppv']['est'])}, NPV {f(mm['npv']['est'])}, Prävalenz (gewichtet) {f(mm['prev']['est'])}, LR+ {f(mm['lrp']['est'], 2)}, Trivialschranke {f(mm['trivial_bound']['est'])}, ESS-Anteil {f(mm['ess_frac']['est'], 2)}; Worst-Case-Schranken Sens [{f(mm['manski']['sens_lo'])}; {f(mm['manski']['sens_hi'])}], Spez [{f(mm['manski']['spec_lo'])}; {f(mm['manski']['spec_hi'])}]"
                 + (f"; Positivrate gewichtet {f(mm['p_t']['est'])}" if 'p_t' in mm else "")
                 + (f", beobachtet {f(mm['p_t_beobachtet']['est'])}" if 'p_t_beobachtet' in mm else "")
                 + ("; 95 %-Intervalle der Worst-Case-Schranken " + ", ".join(f"{k} [{f(v.get('lo95'))}; {f(v.get('hi95'))}]" for k, v in mm.get("manski_intervalle", {}).items()) if mm.get("manski_intervalle") else ""))
    for hk, hv in (u.get("hinweisurteile") or {}).items():
        L.append(f"- Hinweisurteil {hk}: {f(hv['estimate'], 2)} [{f(hv['lo'], 2)}; {f(hv['hi'], 2)}] -> {kl(hv['klasse'])} ({hv['herkunft']})")
    dg = u.get("diagnostik") or {}
    if dg.get("gewichtet"):
        L.append(f"- Verifikationsmodell: c-Statistik {f(dg['c_statistik'])}, ESS-Anteil {f(dg['ess_frac'], 2)}, größte standardisierte Differenz {f(dg['smd_max_ohne_gewichtung'], 2)} → {f(dg['smd_max_nach_gewichtung'], 3)} nach Gewichtung ({dg['n_smd_ueber_0_1']} von {dg['n_spalten']} Spalten über 0,1), Ridge-Rückfall: {dg['ridge']}")
    L.append("")
    L.append("Sensitivitäten (ohne Urteil, B = 1.000):")
    for nm, s in (u.get("sensitivitaeten") or {}).items():
        if s.get("status") != "ok":
            L.append(f"- {nm}: {s.get('status')}")
            continue
        for t, mm in s["metrics"].items():
            L.append(f"- {nm} ({t}): n={s['n_pop']}, verifiziert {s['n_verified']}, Fälle {s['cases']}; Sens {f(mm['sens']['est'])} [{f(mm['sens']['lo95'])}; {f(mm['sens']['hi95'])}], Spez {f(mm['spec']['est'])} [{f(mm['spec']['lo95'])}; {f(mm['spec']['hi95'])}]")
        for sx in ("F", "M"):  # K2 (feste Schwellen), geschlechtsgetrennt gerechnet
            gs = (s.get("geschlecht") or {}).get(sx)
            if not gs:
                continue
            if "metrics" not in gs:
                L.append(f"- {nm}, Geschlecht {sx}: {gs.get('status')}")
                continue
            for t, mm in gs["metrics"].items():
                L.append(f"- {nm}, Geschlecht {sx} ({t}): n={gs['n_pop']}, Fälle {gs['cases']}; Sens {f(mm['sens']['est'])} [{f(mm['sens']['lo95'])}; {f(mm['sens']['hi95'])}], Spez {f(mm['spec']['est'])} [{f(mm['spec']['lo95'])}; {f(mm['spec']['hi95'])}]")
        for k, v in ((s.get("geschlecht") or {}).get("interaktion_F_minus_M") or {}).items():
            L.append(f"- {nm}, Differenz Frauen minus Männer {k}: {f(v['diff_est'])} [{f(v['lo95'])}; {f(v['hi95'])}] (explorativ)")
    g = u.get("geschlecht") or {}
    for sx in ("F", "M"):
        s = g.get(sx)
        if s and "metrics" in s:
            for t, mm in s["metrics"].items():
                L.append(f"- Geschlecht {sx} ({t}): n={s['n_pop']}, Fälle {s['cases']}; Sens {f(mm['sens']['est'])} [{f(mm['sens']['lo95'])}; {f(mm['sens']['hi95'])}], Spez {f(mm['spec']['est'])} [{f(mm['spec']['lo95'])}; {f(mm['spec']['hi95'])}]")
    for k, v in (g.get("interaktion_F_minus_M") or {}).items():
        L.append(f"- Differenz Frauen minus Männer {k}: {f(v['diff_est'])} [{f(v['lo95'])}; {f(v['hi95'])}] (explorativ)")


def sect_a(uid, u, L):
    L.append(f"Behauptung: {u['claim']}, θ = {f(u['theta'], 4)}. Kohorte: {u['kohorte']['n_ed_brustschmerz_troponin']} Personen (ED, Brustschmerz, Troponin), Normal-EKG ohne Zusatz {u['kohorte']['n_normal']}, mit Zusatz {u['kohorte']['n_normal_mit_zusatz']}.")
    for nm in ("primaer", "konkordanz"):
        p = u[nm]
        L.append(f"- {nm}: n={p['n']}, Ereignisse {p['events']}, Schätzer {f(p['estimate'], 4)}, Schranken (α/m={p['alpha_c']}) [{f(p['lo'], 4)}; {f(p['hi'], 4)}], 95 % [{f(p['lo95'], 4)}; {f(p['hi95'], 4)}], Klasse {kl(p['klasse'])}")
    L.append(f"**Endurteil: {kl(u['final']['klasse'])}**; Hinweise: {hint_txt(u['final']['hinweise'])}.")
    for k, v in (u.get("sekundaer") or {}).items():
        L.append(f"- Sekundär {k}: n={v['n']}, Ereignisse {v['events']}, Schätzer {f(v['estimate'], 4)} [{f(v['lo95'], 4)}; {f(v['hi95'], 4)}]")
    for sx, v in (u.get("geschlecht") or {}).items():
        if sx == "interaktion_F_minus_M":
            L.append(f"- Differenz Frauen minus Männer: {f(v['diff_est'], 4)} [{f(v['lo95'], 4)}; {f(v['hi95'], 4)}] (explorativ)")
            continue
        L.append(f"- Geschlecht {sx}: n={v['n']}, Ereignisse {v['events']}, Schätzer {f(v['estimate'], 4)} [{f(v['lo95'], 4)}; {f(v['hi95'], 4)}]")


def sect_p(uid, u, L):
    k = u["kohorte"]
    L.append(f"Index {k['n_index']}, Regel auswertbar {k['n_regel_auswertbar']}, Population der Behauptung {k['n_claim_pop']}, davon exponiert {k['n_exponiert_claim']}.")
    L.append("")
    L.append("| Fassung | Endpunkt | HR | Schranken (α/m=0,00625) | 95 % | n | Ereignisse | exponiert | Klasse |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for var in ("primaer", "konkordanz"):
        for oc, v in u[var].items():
            if v.get("status") == "ok":
                L.append(f"| {var} | {oc} | {f(v['hr'])} | [{f(v['lo'])}; {f(v['hi'])}] | [{f(v['lo95'])}; {f(v['hi95'])}] | {v['n']} | {v['events']} | {v['n_exposed']} | {kl(v['klasse'])}{' (' + v['hinweis'] + ')' if v.get('hinweis') else ''} |")
            else:
                L.append(f"| {var} | {oc} | {v.get('status')} | | | {v.get('n', '')} | | | |")
    L.append("")
    L.append(f"**Endurteil: {kl(u['final']['klasse'])}** (Primär {kl(u['final']['primaer'])}, Konkordanz {kl(u['final']['konkordanz']) if u['final'].get('konkordanz') else 'n/a'}); Hinweise: {hint_txt(u['final']['hinweise'])}.")
    L.append("")
    L.append("Sekundäranalysen (ohne Urteil):")
    for nm, d in (u.get("sekundaer") or {}).items():
        if nm == "or_sekundaer_endpunkte":
            for k2, v in d.items():
                L.append(f"- OR {k2}: " + (f"{f(v['OR'], 2)} [{f(v['lo95'], 2)}; {f(v['hi95'], 2)}], n={v['n']}, Ereignisse {v['events']}" if v.get("status") == "ok" else str(v.get("status"))))
        elif nm == "geschlecht":
            for oc, v in d.items():
                for sx in ("F", "M"):
                    x = v.get(sx, {})
                    if x.get("status") == "ok":
                        L.append(f"- Geschlecht {sx} {oc}: HR {f(x['hr'])} [{f(x['lo95'])}; {f(x['hi95'])}], n={x['n']}, Ereignisse {x['events']}")
                if "interaktion_HR_F_durch_HR_M" in v:
                    i = v["interaktion_HR_F_durch_HR_M"]
                    L.append(f"- Interaktion {oc}: HR_F/HR_M {f(i['ratio'], 2)} [{f(i['lo95'], 2)}; {f(i['hi95'], 2)}]")
        elif nm == "absolutes_risiko_365_tage":
            for oc, x in d.items():
                L.append(f"- Absolutes 365-Tage-Risiko {oc} (Kaplan-Meier, dieselbe Auswahl wie Cox): exponiert {f(x.get('exponiert'), 4)}, nicht exponiert {f(x.get('nicht_exponiert'), 4)}; Personen mit Risikozeit 0 ausgeschlossen: {x.get('n_risikozeit_null')}")
        else:
            for oc, x in d.items():
                if x.get("status") == "ok":
                    L.append(f"- {nm} {oc}: HR {f(x['hr'])} [{f(x['lo95'])}; {f(x['hi95'])}], n={x['n']}, Ereignisse {x['events']}")
                else:
                    L.append(f"- {nm} {oc}: {x.get('status')}")


def sect_r08(uid, u, L):
    L.append(f"Index {u['kohorte']['n_index']}, Regel auswertbar {u['kohorte']['n_regel_auswertbar']}, QTc > 500 ms (Bazett) positiv {u['kohorte']['n_positiv']}.")
    L.append("")
    L.append("| Fassung | Ergebnis | RR | Schranken (α/m=0,00833) | 95 % | a/n1 | c/n0 | Klasse |")
    L.append("|---|---|---|---|---|---|---|---|")
    for var in ("primaer", "konkordanz"):
        for oc, v in u[var].items():
            if v.get("status") == "ok":
                L.append(f"| {var} | {oc} | {f(v['rr'])} | [{f(v['lo'])}; {f(v['hi'])}] | [{f(v['lo95'])}; {f(v['hi95'])}] | {v['a']}/{v['n1']} | {v['c']}/{v['n0']} | {kl(v['klasse'])}{' (' + v['hinweis'] + ')' if v.get('hinweis') else ''} |")
            else:
                L.append(f"| {var} | {oc} | {v.get('status')} | | | | | |")
    L.append("")
    L.append(f"**Endurteil: {kl(u['final']['klasse'])}** (Primär {kl(u['final']['primaer'])}, Konkordanz {kl(u['final']['konkordanz'])}); Hinweise: {hint_txt(u['final']['hinweise'])}.")
    L.append("")
    for oc, d in u["sekundaer"].items():
        for nm, v in d.items():
            if nm == "geschlecht":
                continue
            if isinstance(v, dict) and v.get("status") == "ok":
                lo = v.get("lo95")
                L.append(f"- {oc} / {nm}: RR {f(v['rr'])} [{f(lo)}; {f(v.get('hi95'))}]" + (f", n={v['n']}" if "n" in v else ""))
            elif isinstance(v, dict):
                L.append(f"- {oc} / {nm}: {v.get('status')}")
    for oc, v in u["sekundaer"].items():
        gsx = v if False else None


def sect_r19(uid, u, L):
    for nm, v in u["varianten"].items():
        if v.get("status") == "ok":
            extra = ""
            if "risk_f_bei_150" in v:
                extra = f"; Risiko Frauen/Männer bei 150 ms: {f(v['risk_f_bei_150'], 4)}/{f(v['risk_m_bei_150'], 4)}; verifiziert {v['n_verified']}, Fälle {v['cases']}"
            L.append(f"- {nm}: Population {v['n_pop']}; Δ = {f(v['delta'], 1)} ms [{f(v['lo'], 1)}; {f(v['hi'], 1)}] (α/m=0,00833), 95 % [{f(v['lo95'], 1)}; {f(v['hi95'], 1)}], Schnittpunktstatus {v['delta_status']}, Wiederholungen ohne eindeutigen Schnittpunkt {v['bootstrap_kein_eindeutiger_schnittpunkt']}{(', Anpassungsfehler ' + str(v['bootstrap_fit_fehlgeschlagen'])) if 'bootstrap_fit_fehlgeschlagen' in v else ''}, Klasse {kl(v['klasse'])} ({v['anzeige']}){extra}")
        else:
            L.append(f"- {nm}: {v.get('status')}")
    L.append(f"**Endurteil: {kl(u['final']['klasse'])}**; Hinweise: {hint_txt(u['final']['hinweise'])}.")
    for nm, v in (u.get("sekundaer") or {}).items():
        L.append(f"- Sekundär {nm}: " + (f"Δ = {f(v['delta'], 1)} ms [{f(v.get('lo95'), 1)}; {f(v.get('hi95'), 1)}], n={v['n_pop']}" if v.get("status") == "ok" else str(v.get("status"))))


def build(mode, base):
    res = json.loads((Path(base) / mode / "results.json").read_text(encoding="utf-8"))
    res = {k: K.masked_copy(v) for k, v in res.items()}
    L = [f"# K1 Ergebnisbericht {mode} (automatisch aus `results.json` erzeugt; Kleinzellenregel angewendet)", ""]
    L.append("| Einheit | Familie | Endurteil | Primär | Replikation | Hinweisklassen |")
    L.append("|---|---|---|---|---|---|")
    for uid in ORDER:
        u = res.get(uid)
        if not u:
            continue
        fin = u.get("final") or {}
        rep = (u.get("replikation") or {}).get("urteil", "")
        L.append(f"| {uid} | {u.get('family', '')} | {kl(fin.get('klasse', u.get('status', 'n/a')))} | {kl(fin.get('primaer', ''))} | {rep} | {hint_txt(fin.get('hinweise', []))} |")
    L.append("")
    for uid in ORDER:
        u = res.get(uid)
        if not u:
            continue
        L.append(f"## {uid}")
        L.append("")
        if u.get("status") != "ok":
            L.append(f"Status: {u.get('status')}")
        elif u.get("family") == "E":
            sect_e(uid, u, L)
        elif u.get("family") == "A":
            sect_a(uid, u, L)
        elif u.get("family") == "P":
            sect_p(uid, u, L)
        elif uid.startswith("R08"):
            sect_r08(uid, u, L)
        elif uid == "R19":
            sect_r19(uid, u, L)
        rp = u.get("replikation")
        if rp:
            L.append("")
            L.append(f"Replikation: Entwicklung {kl(rp.get('entwicklung', ''))}, Test {kl(rp.get('test', ''))} → **{rp.get('urteil', rp.get('status'))}**")
            for k, v in (rp.get("effekt_differenz") or {}).items():
                if "differenz" in v:
                    L.append(f"- Effektdifferenz Test minus Entwicklung ({k}, {v.get('skala')}): {f(v['differenz'])} [{f(v['lo95'])}; {f(v['hi95'])}], verträglich: {v.get('vertraeglich')}")
                else:
                    L.append(f"- Effektdifferenz ({k}): {v.get('status')}")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    base = sys.argv[3] if len(sys.argv) > 3 else str(data_path("pub13.aw236"))
    Path(out).write_text(build(mode, base), encoding="utf-8")
    print("geschrieben", out)
