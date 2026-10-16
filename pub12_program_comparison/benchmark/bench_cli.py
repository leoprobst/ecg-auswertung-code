"""Command line of the benchmark scaffold (no computation).

Call:  <local path> werkzeuge/benchmark/bench_cli.py <command> ...

  liste                      datasets and recording counts (lists only, no signals)
  schema <target_folder>     writes ausgabeschema.csv and messgroessen.csv
  rauchtest <dataset> <id>   one recording -> BRAVEHEART input file under <local path>
  export <dataset> <ids.txt> <target_folder>   batch (not in use yet, resource check beforehand)

BRAVEHEART is not run (see bench_braveheart.bh_aufruf_platzhalter).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_braveheart as bb  # noqa: E402
import bench_datasets as bd  # noqa: E402
import bench_schema as bs  # noqa: E402

RAUCHTEST_ROOT = Path(str(data_path("pub12.benchmark_rauchtest")))


def cmd_liste() -> int:
    for k, d in bd.DATENSAETZE.items():
        try:
            n = len(bd.aufnahmen(k))
        except Exception as e:  # list is missing
            n = f"Error: {type(e).__name__}"
        print(f"{k:10s} {d.anzeige:16s} recordings={n}  local signals={d.signale_vorhanden}  12 leads={d.zwoelf_ableitungen}")
    return 0


def cmd_schema(ziel: str) -> int:
    for p in bs.schreibe_schema(Path(ziel)):
        print(p)
    return 0


def _hea_physikalisch(basis: Path) -> np.ndarray:
    """Independent re-implementation of load_wfdb_dat.m: (raw - zero) / gain, format 16, interleaved."""
    zeilen = [z for z in (basis.with_suffix(".hea")).read_text(encoding="utf-8").splitlines() if not z.startswith("#")]
    kopf = zeilen[0].split()
    n_sig = int(kopf[1])
    gain, null, namen = [], [], []
    for z in zeilen[1:1 + n_sig]:
        t = z.split()
        g = t[2]
        if "(" in g:
            gain.append(float(g[:g.index("(")]))
            null.append(float(g[g.index("(") + 1:g.index(")")]))
        else:
            gain.append(float("".join(c for c in g if c.isdigit())))
            null.append(0.0)
        namen.append(t[8].upper())
    roh = np.fromfile(basis.with_suffix(".dat"), dtype="<i2").reshape(-1, n_sig).astype(np.float64)
    phys = (roh - np.array(null)) / np.array(gain)
    idx = [namen.index(n) for n in bb.SPALTEN_UNFORMATTED]
    return phys[:, idx]


def cmd_rauchtest(datensatz: str, aufnahme: str) -> int:
    basis = bd.wfdb_basis(datensatz, aufnahme)
    ziel_dir = RAUCHTEST_ROOT / f"{datensatz}_{aufnahme}"
    if ziel_dir.exists() and any(ziel_dir.iterdir()):
        print(f"Target folder not empty: {ziel_dir}")
        return 2
    ziel = ziel_dir / f"{datensatz}_{aufnahme}.txt"
    man = bb.konvertiere(basis, ziel)
    signale, fs, _ = bb.lese_wfdb_mv(basis)
    man["roundtrip_text"] = bb.pruefe_unformatted_roundtrip(ziel, signale, fs)
    unabh = _hea_physikalisch(basis)
    man["abweichung_gegen_nachbildung_load_wfdb_dat_mV"] = float(np.max(np.abs(unabh - signale)))
    man["amplitude_min_max_mV"] = [float(signale.min()), float(signale.max())]
    man["bemerkung"] = "BRAVEHEART not run; conversion and self-check only"
    (ziel_dir / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(man, indent=2))
    return 0


def cmd_export(datensatz: str, ids_datei: str, zielordner: str) -> int:
    raise NotImplementedError(
        "Batch export is released only with an assignment and a resource check (ram_reservierung.py)."
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    b = argv[1]
    if b == "liste":
        return cmd_liste()
    if b == "schema" and len(argv) == 3:
        return cmd_schema(argv[2])
    if b == "rauchtest" and len(argv) == 4:
        return cmd_rauchtest(argv[2], argv[3])
    if b == "export" and len(argv) == 5:
        return cmd_export(argv[2], argv[3], argv[4])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
