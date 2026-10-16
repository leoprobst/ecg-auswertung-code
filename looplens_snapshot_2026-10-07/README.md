# LoopLens snapshot of 7 October 2026 (frozen research state)

This folder holds the measurement software that produced the ECG and vectorcardiogram (VCG) marker values analysed in the four preprints of this repository. It is the frozen research state, not maintained software: nothing here is updated, supported or meant for clinical use. It is published so that the analysis code in the sibling folders can be read and checked against the code that generated its inputs.

License of this folder: PolyForm Noncommercial License 1.0.0 (file `LICENSE`, required notice "Copyright 2026 Leo Probst"). The analysis scripts outside this folder are MIT-licensed (root `LICENSE`).

## Contents

| Path | What it is |
|---|---|
| `vcg_biomarker/` | The Python package of the release of 7 October 2026 (875 Python files: signal processing, delineation, VCG reconstruction, marker calculation, the MIMIC extraction and analysis stages, tests). Byte-identical to the release folder. |
| `pyproject.toml`, `requirements-lock-win-py312.txt`, `requirements-simulation-optional.txt` | Tool configuration and the frozen dependency list of the validated environment (Windows, CPython 3.12). Byte-identical. |
| `vcg_biomarker/requirements*.txt`, `vcg_biomarker/mimic/result_equivalence.json`, `vcg_biomarker/mimic/resources//SOURCE.json`, `vcg_biomarker/mimic/resources/clinical_utility_assessment.schema.json` | Configuration read by the package. Byte-identical. |
| `patch_2026-10-11/vcg_biomarker/` | The files that differ in the executed runtime snapshot. In `mimic/`: the seven files changed on 11 October 2026 (`association_domains.py`, `broad_axis_adjustment.py`, `broad_axis_stage.py`, `frame_checkpoints.py`, `future_phenome.py`, `phenome.py`, `result_equivalence.json`). In `tests/`: 16 test files and fixtures that exist only in the executed copy and the one test file that differs (`test_validation_cache_streaming.py`). Byte-identical to the executed files. |

Everything in `vcg_biomarker/` and `patch_2026-10-11/` is unmodified, including comments (some in German). The files `README.md` and `LICENSE` of this folder were added for the repository.

## Two states, three hashes

The marker values used in the preprints were computed by the runtime snapshot of the run. It equals the release folder except for the seven files in `patch_2026-10-11/`, which belong to late analysis stages (association, phenome, broad-axis adjustment). The code that defines a marker (signal processing, delineation, reconstruction, marker calculation, feature extraction) is identical in both states.

| Quantity | SHA-256 |
|---|---|
| Source-tree hash of the 875 Python files of `vcg_biomarker/`, release state of 7 October 2026 (prefix `9bdad0ce`) | `9bdad0ce4372718f11d2d35feae42031b4ba15bb057a83132d6216f83903899d` |
| Source-tree hash of the executed runtime snapshot (prefix `d80fc3c3`), 891 Python files | `d80fc3c3bec26f815f686b9195917716edf006a54a7caa147ba2f7ba1c76ba89` |
| Feature-code hash (code that defines a marker: the `core` tree plus six dataset modules); equal for the release state, the executed snapshot and the value recorded by the extraction run (prefix `c166bd00`) | `c166bd00403dcde9f154d6c50f15ae16791029d9b507c4a4094e6f96215789b9` |

The source-tree hash is the SHA-256 of the canonical JSON list of `{relative_path, sha256}` entries of all `*.py` files below `vcg_biomarker/` (sorted by path, `__pycache__` excluded). `verify_snapshot_hash.py` in the repository root recomputes it: without options it reproduces the release-state hash above from this folder.

The hash of the executed snapshot (891 Python files) covers 17 test files that exist only in the executed copy (16 new files and one changed file). They are synthetic unit tests and code fixtures without data and are published in `patch_2026-10-11/vcg_biomarker/tests/`. Copying the contents of `patch_2026-10-11/vcg_biomarker/` over `vcg_biomarker/` gives the executed state, and `python verify_snapshot_hash.py --with-patch` then prints the executed-snapshot hash above.

## Installing

The package is imported as `vcg_biomarker`. Put this folder on the Python path (or install the dependencies from `requirements-lock-win-py312.txt` and run from this folder):

```
cd looplens_snapshot_2026-10-07
python -m pip install -r requirements-lock-win-py312.txt
python -c "import vcg_biomarker"
```

The analysis scripts in the sibling folders import `vcg_biomarker...` where they need the package; add this folder to `PYTHONPATH`. To reproduce the executed state, copy the files of `patch_2026-10-11/vcg_biomarker/` over `vcg_biomarker/` first.

## What is deliberately not in this folder

| Excluded | Reason |
|---|---|
| Documentation and run notes (`*.md` in the release root and in the package, 108 files) | Internal working notes, not source code. |
| `result.csv` (9 MB), `source_fixture_manifest.json`, `config_hash.txt`, `info.txt`, `.schema_uniformity.json` | Result table, manifests and run notes of the release folder, not source or required configuration. |
| `audit_tools/` (208 files, mostly Python), `scripts/` (6 files) | Operations and audit helpers around the package (run supervision, resource policy, benchmarks); not needed to compute a marker. |
| `research_refresh_20260911/`, `research_pathology_round_20260913/`, `research_score_focus_20260914/`, `research_pre_run_extensions_20260914/` | Research reports, JSON result files and designs, no source code. |
| `assets/` (14 files), `vcg_biomarker/gui/resources/heart3d/heart_model.npz` and its attribution note | Images, fonts and a binary heart model used by the graphical interface only. |
| `vcg_biomarker/gui/vcglab/reference_limits.json` | Percentile table derived from a restricted-access database. The graphical VCG lab reads it through `load_reference()` in `gui/vcglab/widgets.py` and works without it (the function returns None, and the reference columns stay empty). To create one from your own data: take your own set of reference ECGs, compute the beat measures of `gui/vcglab/measures.py` for each (Kors reconstruction, origin-corrected median beat), and write a JSON file with a `_meta` block (`source`, `pipeline`, `n`, `n_female`, `n_male`, `date`) and, for each measure name used by the lab, an object with `n`, `p2_5`, `p5`, `p50`, `p95`, `p97_5`. Save it as `reference_limits.json` next to `widgets.py`. |
| `vcg_biomarker/mimic/resources/ccsr//*.csv`, `.../phecodex/*.csv` (5 files, about 25 MB) | Third-party code-system tables (AHRQ CCSR, PheCodeX); redistribute from the original sources, see the `SOURCE.json` files. |
| `vcg_biomarker/tests/fixtures/*.txt` | Non-Python test fixture. |
| `.simulation-deps/`, `.pytest_cache/`, all `__pycache__/` | Vendored third-party packages and caches. |

## No data, no warranty

The folder contains no patient data, no ECG waveforms and no study results. Several tests and modules contain synthetic example values and default location names for a local database layout; they are original code and were not edited. The software is provided as is (see `LICENSE`). The measurement core is a research implementation; its use on clinical decisions is not intended and not validated.
