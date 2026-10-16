"""Test des Treibers der Testergaenzung (Kunstdaten, kein Datenzugriff auf MIMIC)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import k1_testergaenzung_treiber as T  # noqa: E402


def test_selbsttest_kunstdaten():
    T.selftest()
