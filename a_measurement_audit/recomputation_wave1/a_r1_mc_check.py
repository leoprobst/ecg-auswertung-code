"""AW-240 / A-R1: Monte-Carlo-Streuung der Bootstrap-Grenzen der Gegenrechnung (5 weitere Seeds, je 10.000 Ziehungen)."""
import sys
sys.dont_write_bytecode = True
import numpy as np, pyarrow.parquet as pq
import a_r1_gegenrechnung as G

t = pq.read_table(G.RAHMEN, columns=["wall_mm", "v6"]).to_pandas()
x, y = t["wall_mm"].to_numpy(float), t["v6"].to_numpy(float)
p = np.load(G.PAARE)
rx = 1.0 - (np.mean(p["dx"].astype(float) ** 2) / 2) / x.var(ddof=1)
res = []
for s in (240111, 240112, 240113, 240114, 240115):
    r_b, b_b, c_b = G.bootstrap(x, y, rx, G.N_BOOT, s)
    res.append([*G.pi(r_b), *G.pi(b_b), *G.pi(c_b)])
res = np.array(res)
nm = ["r_lo", "r_hi", "OLS_lo", "OLS_hi", "OLSRX_lo", "OLSRX_hi"]
for i, n in enumerate(nm):
    print(f"{n:9s} Mittel {res[:, i].mean(): .5f}  SD ueber Seeds {res[:, i].std(ddof=1):.5f}  Min {res[:, i].min(): .5f}  Max {res[:, i].max(): .5f}")
