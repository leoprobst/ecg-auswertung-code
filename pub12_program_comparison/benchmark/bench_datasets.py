"""Dataset register of the ECG/VCG benchmark.

Listing only. This module does not open signal files. At most it reads
the RECORDS lists or the PTB-XL metadata table (no signals).
Signals and person-level identifiers belong in <local path> and
<local path>, never in this <cloud folder> folder.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import csv
from dataclasses import dataclass
from pathlib import Path

DB_ROOT = Path(str(data_path("pub12.database")))
# PTB-XL 1.0.3 with signals (checked, read only): local copy in the project folder
PTBXL_LOKAL = Path(str(data_path("pub12.samples_ptb_x")))


@dataclass(frozen=True)
class Datensatz:
    kennung: str          # short key, e.g. "ludb"
    anzeige: str          # name and version
    wurzel: Path          # directory with the WFDB files (or metadata)
    fs_nativ: int         # native sampling rate (Hz), 0 = differs per file
    zwoelf_ableitungen: bool   # BRAVEHEART needs at least 8, VCG needs 12
    signale_vorhanden: str     # "ja" / "nein" / "pruefen"
    bemerkung: str


DATENSAETZE: dict[str, Datensatz] = {
    "ludb": Datensatz(
        "ludb", "LUDB 1.0.1",
        DB_ROOT / "lobachevsky-university-electrocardiography-database-1.0.1",
        500, True, "ja",
        "200 recordings, 12 leads, 10 s, WFDB format 16, expert marks per lead in data/<id>.<lead>",
    ),
    "ptbxl": Datensatz(
        "ptbxl", "PTB-XL 1.0.3",
        DB_ROOT / "research-diagnostic-identifiability-20261013" / "ptb-xl-1.0.3",
        500, True, "ja",
        "Metadata under wurzel; signals (records500, 21,799 recordings, checked) local under PTBXL_LOKAL. "
        "records100 is not used.",
    ),
    "ptbxlplus": Datensatz(
        "ptbxlplus", "PTB-XL+ 1.0.1",
        DB_ROOT / "ptb-xl-plus-1.0.1",
        500, True, "nein",
        "Features and labels (ECGDeli, 12SL, Uni-G), no signals of its own; same persons as PTB-XL.",
    ),
    "qtdb": Datensatz(
        "qtdb", "QTDB 1.0.0",
        DB_ROOT / "qt-database-1.0.0",
        250, False, "ja",
        "Two leads, WFDB format 212, 250 Hz. The BRAVEHEART loader requires at least 8 leads "
        "and format 16/32/61: QTDB cannot be loaded by BRAVEHEART (single-lead tools only).",
    ),
}


def liste_ludb() -> list[str]:
    """Recording numbers from RECORDS (e.g. 'data/1' -> '1')."""
    pfad = DATENSAETZE["ludb"].wurzel / "RECORDS"
    zeilen = [z.strip() for z in pfad.read_text(encoding="utf-8").splitlines() if z.strip()]
    return [z.split("/")[-1] for z in zeilen]


def liste_qtdb() -> list[str]:
    pfad = DATENSAETZE["qtdb"].wurzel / "RECORDS"
    return [z.strip() for z in pfad.read_text(encoding="utf-8").splitlines() if z.strip()]


def liste_ptbxl() -> list[str]:
    """ecg_id from the metadata table. No person identifier (patient_id) is read."""
    pfad = PTBXL_LOKAL / "ptbxl_database.csv"
    with pfad.open(encoding="utf-8", newline="") as f:
        leser = csv.DictReader(f)
        return [z["ecg_id"] for z in leser]


def liste_ptbxlplus() -> list[str]:
    """PTB-XL+ has no recording list of its own, the ecg_id follows PTB-XL."""
    return liste_ptbxl()


LISTER = {
    "ludb": liste_ludb,
    "ptbxl": liste_ptbxl,
    "ptbxlplus": liste_ptbxlplus,
    "qtdb": liste_qtdb,
}


def aufnahmen(kennung: str) -> list[str]:
    return LISTER[kennung]()


def wfdb_basis(kennung: str, aufnahme: str) -> Path:
    """Path without extension to the WFDB files of a recording (only where signals are stored locally)."""
    d = DATENSAETZE[kennung]
    if kennung == "ludb":
        return d.wurzel / "data" / aufnahme
    if kennung == "qtdb":
        return d.wurzel / aufnahme
    if kennung == "ptbxl":
        # records500/<group>/<id:05d>_hr, group = ecg_id // 1000 * 1000 (5 digits), as filename_hr in ptbxl_database.csv
        n = int(aufnahme)
        return PTBXL_LOKAL / "records500" / f"{n // 1000 * 1000:05d}" / f"{n:05d}_hr"
    raise NotImplementedError(
        f"{kennung}: signal path not bound (signals not available locally or path open)."
    )
