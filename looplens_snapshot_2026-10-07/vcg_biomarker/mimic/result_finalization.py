"""One-at-a-time validation export; no simultaneous sanitized table copies.

This bounds additional export memory to one table, not yet to one row group.
Scientific FDR/selection must be completed before passing a table here.
"""
from pathlib import Path

from .storage import atomic_write_parquet


def finalize_result_tables(sources, root, transform, *, retain=()):
    """Consume sources destructively so finished tables can be reclaimed.

    A source can be a ready dataframe or a zero-argument factory. Never call
    multiple factories in advance. A failed write propagates and no completed
    stage manifest is issued by this function.
    """
    root = Path(root)
    retained, counts, paths = {}, {}, []
    for name in tuple(sources):
        source = sources.pop(name)
        frame = source() if callable(source) else source
        del source
        frame = transform(name, frame)
        path = root / f"{name}.parquet"
        atomic_write_parquet(frame, path)
        counts[name] = len(frame)
        paths.append(str(path.resolve()))
        if name in retain:
            retained[name] = frame
        del frame
    return retained, counts, paths
