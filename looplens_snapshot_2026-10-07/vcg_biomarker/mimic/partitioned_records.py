"""Endpoint-bounded tabular buffers for validation (not model serialization).

This removes the full-run Python-dictionary copy. It does not claim that the
downstream wide-table finalizer is out-of-core; its peak must still be measured.
"""
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .storage import atomic_write_parquet, file_sha256


class PartitionedRecords(Sequence):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.pending = []
        self.parts = []
        self.spilled_rows = 0
        self.checksums = {}

    def __len__(self):
        return self.spilled_rows + len(self.pending)

    def append(self, row):
        self.pending.append(row)

    def extend(self, rows):
        self.pending.extend(rows)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if start >= self.spilled_rows:
                return self.pending[start - self.spilled_rows:stop - self.spilled_rows:step]
            raise ValueError("Historical dictionary slices forbidden; use to_frame")
        index = key if key >= 0 else len(self) + key
        if index < self.spilled_rows:
            raise ValueError("Historical dictionary indexing forbidden; use to_frame")
        return self.pending[index - self.spilled_rows]

    def flush(self, path=None):
        if not self.pending:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = Path(path) if path is not None else self.root / f"part-{len(self.parts):05d}.parquet"
        atomic_write_parquet(pd.DataFrame(self.pending), path)
        self.checksums[path] = file_sha256(path)
        self.parts.append(path)
        self.spilled_rows += len(self.pending)
        self.pending.clear()

    def adopt_verified_partition(self, path, count, sha256):
        if self.pending:
            raise ValueError("Cannot adopt a checkpoint with uncommitted rows")
        self.parts.append(Path(path))
        self.checksums[Path(path)] = sha256
        self.spilled_rows += int(count)

    def to_frame(self):
        self.flush()
        if not self.parts:
            return pd.DataFrame()
        # Arrow columns, not another full-run list of Python dictionaries.
        # Promote null-only columns from sparse endpoints to their actual type.
        tables = []
        for path in self.parts:
            if file_sha256(path) != self.checksums[path]:
                raise RuntimeError(f"Validation partition changed after commit: {path}")
            tables.append(pq.read_table(path))
        joined = pa.concat_tables(tables, promote_options="permissive")
        return joined.to_pandas()


def records_frame(records):
    return records.to_frame() if isinstance(records, PartitionedRecords) else pd.DataFrame(records)


def flush_record_groups(groups):
    for records in groups.values():
        if isinstance(records, PartitionedRecords):
            records.flush()
