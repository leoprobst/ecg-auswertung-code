"""Environment cross-check. Runs LUDB (200) and PT (40) natively with the wrapper ll_lauf.py in the .venv of the scanner project
(numpy 2.4.6 / scipy 1.18.0 as in requirements-lock-win-py312.txt) into a separate result directory. The main run uses the
analysis venv (numpy 2.5.3 / scipy 1.18.1). Comparison only as a count (bit-identical yes/no) by ll_umgebungsvergleich.py.
Call: <python of the .venv> -B ll_umgebungstest.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ll_lauf  # noqa: E402


def aufg(lauf, limit=None):
    t = [("ludb", str(i), "nativ", None) for i in range(1, 201)]
    t += [("ptbxl", r, "nativ", None) for r in ll_lauf.lies_liste("PT")]
    return t


if __name__ == "__main__":
    ll_lauf.mp.freeze_support()
    ll_lauf.aufgaben = aufg
    sys.argv = [sys.argv[0], "--lauf", "haupt", "--workers", "1", "--lauf-id", "AW219_umgebung_venv",
                "--out", str(data_path("pub12.benchmark_ergebnisse_looplens_umgebungstest"))]
    ll_lauf.main()
