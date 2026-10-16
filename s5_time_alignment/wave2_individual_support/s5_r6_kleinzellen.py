# -*- coding: utf-8 -*-
"""AW-245 S5-R6 Nachbearbeitung (Version 1.0): Kleinzellenregel auf S5_R6_kennzahlen.csv. Zelle n=3 mit Wert 1 bis 9 wird '<10', die Komplementzelle n=4 bis 5 nicht ausgewiesen,
dafuer eine zusammengefasste Zeile n=3 bis 5 (Summe der beiden). Liest und schreibt nur die Aggregatdatei; rechnet nichts neu."""
import pandas as pd
from pathlib import Path
p = Path(__file__).resolve().parents[2] / "nachrechnung/AW245/S5_R6_kennzahlen.csv"
d = pd.read_csv(p, dtype=object)
cols = ["ident_p<0.05_und_loo<=15", "ident_p<0.05_und_loo<=15_und_abs_offset<=60"]
b = d[d.satz == "beobachtet"].set_index("gruppe")
neu = d[d.satz == "beobachtet"].iloc[0:0].copy()
zeile = {c: str(int(b.loc["n=3", c]) + int(b.loc["n=4 bis 5", c])) if str(b.loc["n=3", c]).isdigit() else b.loc["n=3", c] for c in cols}
for c in cols:
    if str(b.loc["n=3", c]).isdigit() and 1 <= int(b.loc["n=3", c]) <= 9:
        d.loc[(d.satz == "beobachtet") & (d.gruppe == "n=3"), c] = "<10"
        d.loc[(d.satz == "beobachtet") & (d.gruppe == "n=4 bis 5"), c] = "nicht ausgewiesen (Komplement)"
r = d[(d.satz == "beobachtet") & (d.gruppe == "n=3")].iloc[0].copy()
r["gruppe"] = "n=3 bis 5"
for c in d.columns[2:]:
    if c in cols:
        r[c] = zeile[c]
    else:
        try:
            r[c] = str(int(b.loc["n=3", c]) + int(b.loc["n=4 bis 5", c]))
        except Exception:
            r[c] = b.loc["n=3", c]
d = pd.concat([d, pd.DataFrame([r])], ignore_index=True)
d.to_csv(p, index=False, encoding="utf-8")
print(d[d.satz == "beobachtet"][["gruppe", "personen", "identifizierbar"] + cols].to_string())
