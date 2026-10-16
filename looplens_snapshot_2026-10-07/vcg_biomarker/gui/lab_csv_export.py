"""Atomic local CSV replacement; leave an existing export intact on failure."""
import csv
import os
from pathlib import Path
import tempfile


def write_csv_atomic(path, rows, fieldnames):
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8-sig',
                dir=target.parent, prefix='.vcg-export-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            # Spreadsheet applications must not execute imported provenance as
            # formulas. Genuine numeric negatives remain numeric.
            writer.writerows({key: "'" + value if isinstance(value, str)
                              and value.startswith(('=', '+', '-', '@', '\t', '\r', '\n'))
                              else value for key, value in row.items()} for row in rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
