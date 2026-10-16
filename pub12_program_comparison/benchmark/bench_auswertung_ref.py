"""Unabhaengiger zweiter Rechenweg (Kontrolle K5) fuer Uebereinstimmungskennzahlen.

Reines Python, nur Standardbibliothek (math). Keine Abhaengigkeit von numpy, pandas oder scipy.
Quantile (Typ 7), Raenge (Mittelrang) und Summen (math.fsum) sind selbst implementiert.
"""
import math


# ---------------------------------------------------------------- Hilfsfunktionen

def _mittel(x):
    return math.fsum(x) / len(x)


def ref_quantil(werte, p):
    """Quantil Typ 7: h=(n-1)*p, lineare Interpolation zwischen x[floor(h)] und x[ceil(h)]."""
    s = sorted(werte)
    n = len(s)
    if n == 0:
        return None
    h = (n - 1) * p
    lo = int(math.floor(h))
    hi = int(math.ceil(h))
    if lo == hi:
        return s[lo]
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def ref_median(werte):
    return ref_quantil(werte, 0.5)


def _mittelraenge(x):
    """Raenge 1..n, Gleichstaende erhalten den mittleren Rang."""
    n = len(x)
    idx = sorted(range(n), key=lambda i: x[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        mr = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[idx[k]] = mr
        i = j + 1
    return r


def _pearson(x, y):
    n = len(x)
    if n < 2:
        return None
    mx = _mittel(x)
    my = _mittel(y)
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = math.fsum((a - mx) ** 2 for a in x)
    syy = math.fsum((b - my) ** 2 for b in y)
    if sxx == 0.0 or syy == 0.0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _gname(g):
    gf = float(g)
    return str(int(gf)) if gf.is_integer() else str(g)


# ---------------------------------------------------------------- 1. Paarkennzahlen

def ref_paar(a, b, schwelle=120.0, nah_lo=100.0, nah_hi=140.0, gitter=(2, 5, 10, 20, 40)):
    n = len(a)
    if n != len(b):
        raise ValueError("a und b muessen gleich lang sein")
    d = [float(x) - float(y) for x, y in zip(a, b)]
    out = {"n": n}
    if n == 0:
        return out
    bias = _mittel(d)
    ad = [abs(v) for v in d]
    out["bias"] = bias
    out["mae"] = _mittel(ad)
    out["rmse"] = math.sqrt(math.fsum(v * v for v in d) / n)
    if n > 1:
        sd = math.sqrt(math.fsum((v - bias) ** 2 for v in d) / (n - 1))
    else:
        sd = None
    out["sd"] = sd
    out["median_abs"] = ref_median(ad)
    out["median"] = ref_median(d)
    for name, p in (("q025", 0.025), ("q05", 0.05), ("q25", 0.25),
                    ("q75", 0.75), ("q95", 0.95), ("q975", 0.975)):
        out[name] = ref_quantil(d, p)
    out["iqr"] = out["q75"] - out["q25"]
    if sd is None:
        out["loa_normal_lo"] = None
        out["loa_normal_hi"] = None
    else:
        out["loa_normal_lo"] = bias - 1.96 * sd
        out["loa_normal_hi"] = bias + 1.96 * sd
    for g in gitter:
        out["innerhalb_" + _gname(g)] = sum(1 for v in ad if v <= g) / n

    # Kreuztabelle an der Schwelle (>=)
    A = B = C = D = 0
    for x, y in zip(a, b):
        px = x >= schwelle
        py = y >= schwelle
        if px and py:
            A += 1
        elif px and not py:
            B += 1
        elif (not px) and py:
            C += 1
        else:
            D += 1
    out["zellen"] = {"A": A, "B": B, "C": C, "D": D}
    out["d120"] = (B + C) / n
    nenn = 2 * A + B + C
    out["ppa"] = (2 * A / nenn) if nenn != 0 else None
    nenn = 2 * D + B + C
    out["npa"] = (2 * D / nenn) if nenn != 0 else None
    po = (A + D) / n
    pe = ((A + B) * (A + C) + (C + D) * (B + D)) / (n * n)
    out["kappa"] = ((po - pe) / (1 - pe)) if pe != 1 else None

    # Naehe der Schwelle
    n_nah = 0
    disk = 0
    for x, y in zip(a, b):
        m = (x + y) / 2.0
        if nah_lo <= m < nah_hi:
            n_nah += 1
            if (x >= schwelle) != (y >= schwelle):
                disk += 1
    out["n_nah"] = n_nah
    out["d120_nah"] = (disk / n_nah) if n_nah > 0 else None

    # Lin-CCC mit Populationsmomenten
    ma = _mittel(a)
    mb = _mittel(b)
    va = math.fsum((x - ma) ** 2 for x in a) / n
    vb = math.fsum((y - mb) ** 2 for y in b) / n
    cov = math.fsum((x - ma) * (y - mb) for x, y in zip(a, b)) / n
    nenn = va + vb + (ma - mb) ** 2
    out["lin_ccc"] = (2 * cov / nenn) if nenn != 0 else None

    # Spearman |d| gegen Paarmittel
    m_all = [(x + y) / 2.0 for x, y in zip(a, b)]
    out["spearman_absd_mean"] = _pearson(_mittelraenge(ad), _mittelraenge(m_all))
    return out


# ---------------------------------------------------------------- 2. Wilson mit Kish

def ref_wilson_kish(x, w, z=1.959963984540054):
    sw = math.fsum(w)
    sw2 = math.fsum(v * v for v in w)
    p = math.fsum(wi * xi for wi, xi in zip(w, x)) / sw
    n = sw * sw / sw2
    nenner = 1.0 + z * z / n
    zentrum = (p + z * z / (2.0 * n)) / nenner
    halb = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / nenner
    return {"p": p, "n_eff": n, "lo": zentrum - halb, "hi": zentrum + halb}


# ---------------------------------------------------------------- 3. Holm

def ref_holm(pwerte, alpha=0.05):
    m = len(pwerte)
    ordnung = sorted(range(m), key=lambda i: pwerte[i])
    res = [False] * m
    for k, i in enumerate(ordnung, start=1):
        if pwerte[i] <= alpha / (m - k + 1):
            res[i] = True
        else:
            break
    return res


# ---------------------------------------------------------------- 4. Baender

def ref_band(wert, tabelle):
    if tabelle == "diskordanz":
        pr = 100.0 * wert
        if pr < 1:
            return "kaum diskordant"
        if pr < 5:
            return "gering"
        if pr < 20:
            return "deutlich"
        return "überwiegend verschieden zugeordnet"
    if tabelle == "zeit":
        if wert <= 4:
            return "innerhalb der Abtastauflösung"
        if wert <= 10:
            return "klein"
        if wert <= 20:
            return "mittel"
        return "groß"
    if tabelle == "winkel":
        if wert <= 5:
            return "klein"
        if wert <= 10:
            return "mittel"
        if wert <= 20:
            return "deutlich"
        return "groß"
    if tabelle == "verstaerkung":
        pr = 100.0 * wert
        if pr < 1:
            return "praktisch unverändert"
        if pr < 5:
            return "gelegentlich verändert"
        if pr < 20:
            return "häufig verändert"
        return "überwiegend verändert"
    if tabelle == "ausfall":
        pr = 100.0 * wert
        if pr <= 1:
            return "selten"
        if pr <= 5:
            return "gelegentlich"
        if pr <= 10:
            return "häufig"
        return "Selektion wahrscheinlich"
    raise ValueError("unbekannte Tabelle: %r" % (tabelle,))


# ---------------------------------------------------------------- 5. gewichtetes Mittel

def ref_gewichtet_mittel(werte, w):
    return math.fsum(wi * xi for wi, xi in zip(w, werte)) / math.fsum(w)


# ---------------------------------------------------------------- Selbsttest

def _nah(x, y, tol=1e-12):
    return abs(x - y) <= tol


def _selbsttest():
    # Beispiel 1 (von Hand): d = [-2,-15,2,2,-10]
    a = [100, 110, 120, 130, 150]
    b = [102, 125, 118, 128, 160]
    r = ref_paar(a, b)
    assert r["n"] == 5
    assert _nah(r["bias"], -4.6)
    assert _nah(r["mae"], 6.2)
    assert _nah(r["rmse"], math.sqrt(67.4))
    assert _nah(r["sd"], math.sqrt(57.8))          # (337 - 5*4.6^2)/4
    assert r["median"] == -2 and r["median_abs"] == 2
    assert _nah(r["q025"], -14.5) and _nah(r["q05"], -14.0)
    assert _nah(r["q25"], -10.0) and _nah(r["q75"], 2.0)
    assert _nah(r["q95"], 2.0) and _nah(r["q975"], 2.0)
    assert _nah(r["iqr"], 12.0)
    assert _nah(r["loa_normal_lo"], -4.6 - 1.96 * math.sqrt(57.8))
    assert _nah(r["loa_normal_hi"], -4.6 + 1.96 * math.sqrt(57.8))
    assert _nah(r["innerhalb_2"], 0.6) and _nah(r["innerhalb_5"], 0.6)
    assert _nah(r["innerhalb_10"], 0.8) and _nah(r["innerhalb_20"], 1.0)
    assert _nah(r["innerhalb_40"], 1.0)
    assert r["zellen"] == {"A": 2, "B": 1, "C": 1, "D": 1}
    assert _nah(r["d120"], 0.4)
    assert _nah(r["ppa"], 2 / 3) and _nah(r["npa"], 0.5)
    assert _nah(r["kappa"], 1 / 6)                  # po=.6, pe=.52
    assert r["n_nah"] == 4 and _nah(r["d120_nah"], 0.5)
    assert _nah(r["lin_ccc"], 609.6 / 677.0)        # cov 304.8; var 296 und 359.84; dm^2 21.16
    assert _nah(r["spearman_absd_mean"], 1 / math.sqrt(80))

    # Beispiel 2: Gleichstaende, Nenner 0, Randfaelle
    r2 = ref_paar([130.0, 140.0, 150.0], [130.0, 140.0, 150.0])
    assert _nah(r2["bias"], 0.0) and _nah(r2["sd"], 0.0)
    assert r2["zellen"] == {"A": 3, "B": 0, "C": 0, "D": 0}
    assert _nah(r2["ppa"], 1.0) and r2["npa"] is None      # 2D+B+C = 0
    assert r2["kappa"] is None                              # pe == 1
    assert r2["n_nah"] == 1 and _nah(r2["d120_nah"], 0.0)   # nur m=130 liegt in [100,140)
    assert ref_paar([200.0], [210.0])["d120_nah"] is None   # n_nah == 0
    assert _nah(r2["lin_ccc"], 1.0)
    # Quantil und Median
    assert _nah(ref_quantil([1, 2, 3, 4], 0.5), 2.5)
    assert _nah(ref_quantil([1, 2, 3, 4], 0.25), 1.75)
    assert ref_median([5, 1, 3]) == 3
    # Mittelrang
    assert _mittelraenge([2, 15, 2, 2, 10]) == [2.0, 5.0, 2.0, 2.0, 4.0]


def _selbsttest_rest():
    # Beispiel 3: Wilson/Kish, Holm, Baender, gewichtetes Mittel
    w = [1.0, 1.0, 2.0, 2.0]
    x = [1, 0, 1, 0]
    k = ref_wilson_kish(x, w)
    assert _nah(k["p"], 0.5)
    assert _nah(k["n_eff"], 36.0 / 10.0)            # (6)^2 / (1+1+4+4)
    z = 1.959963984540054
    n = 3.6
    zen = (0.5 + z * z / (2 * n)) / (1 + z * z / n)
    hal = z * math.sqrt(0.25 / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert _nah(k["lo"], zen - hal) and _nah(k["hi"], zen + hal)
    assert _nah(k["lo"] + k["hi"], 2 * zen)
    assert ref_holm([0.001, 0.04, 0.03, 0.2]) == [True, False, False, False]
    # 0.001 <= .05/4 ja; 0.03 <= .05/3=.01667 nein -> Abbruch
    assert ref_holm([0.01, 0.02, 0.04]) == [True, True, True]
    # .01<=.0167, .02<=.025, .04<=.05
    assert ref_holm([0.02, 0.0001, 0.5]) == [True, True, False]   # .0001<=.0167, .02<=.025, .5>.05
    assert _nah(ref_gewichtet_mittel([1, 2, 3], [1, 1, 2]), 9 / 4)
    assert ref_band(0.0099, "diskordanz") == "kaum diskordant"
    assert ref_band(0.01, "diskordanz") == "gering"
    assert ref_band(0.05, "diskordanz") == "deutlich"
    assert ref_band(0.2, "diskordanz") == "überwiegend verschieden zugeordnet"
    assert ref_band(4, "zeit") == "innerhalb der Abtastauflösung"
    assert ref_band(4.1, "zeit") == "klein" and ref_band(10, "zeit") == "klein"
    assert ref_band(20, "zeit") == "mittel" and ref_band(20.5, "zeit") == "groß"
    assert ref_band(5, "winkel") == "klein" and ref_band(10, "winkel") == "mittel"
    assert ref_band(20, "winkel") == "deutlich" and ref_band(21, "winkel") == "groß"
    assert ref_band(0.0, "verstaerkung") == "praktisch unverändert"
    assert ref_band(0.03, "verstaerkung") == "gelegentlich verändert"
    assert ref_band(0.1, "verstaerkung") == "häufig verändert"
    assert ref_band(0.25, "verstaerkung") == "überwiegend verändert"
    assert ref_band(0.01, "ausfall") == "selten"
    assert ref_band(0.03, "ausfall") == "gelegentlich"
    assert ref_band(0.1, "ausfall") == "häufig"
    assert ref_band(0.11, "ausfall") == "Selektion wahrscheinlich"


if __name__ == "__main__":
    import pprint
    _selbsttest()
    _selbsttest_rest()
    print("Selbsttest bestanden")
    pprint.pprint(ref_paar([100, 110, 120, 130, 150], [102, 125, 118, 128, 160]), sort_dicts=False)
    pprint.pprint(ref_wilson_kish([1, 0, 1, 0], [1.0, 1.0, 2.0, 2.0]), sort_dicts=False)
