# Analysis code for four ECG/VCG preprints

This repository holds the analysis scripts behind four preprints by Leo Probst, together with the frozen research state of the measurement software that produced their inputs. It contains code only: no patient data, no ECG waveforms, no record-level or individual-level values, and no result tables. All numbers reported in the manuscripts are in the manuscripts and their supplements.

Status: prepared for release v1.0.0; the archived version (Zenodo) does not exist yet.

## The four works

| Folder | Working title | What the code does |
|---|---|---|
| `a_measurement_audit/` | Repeat variation and definition dependence of ECG/VCG markers: an exploratory audit of one implementation (LoopLens 10.1) | Repeat variation of a marker catalog in routine-care ECG pairs, dependence on the reconstruction matrix, boundary errors against annotated recordings, replication on a held-out partition, external pairs in PTB-XL. |
| `b_qrs_t_coupling/` | Does QRS-T loop coupling add to the QRS-T angle? Frechet distance and reduced ejection fraction in hospital ECGs, with named measurement-error scenarios | Association of a Frechet-distance marker with reduced left ventricular ejection fraction, measurement-error corrections, sensitivity analyses. |
| `s5_time_alignment/` | Heart-rate-based clock alignment in intensive-care ECGs, serial-ECG matching and administrative follow-up in MIMIC-IV | Alignment of bedside heart-rate records and ECG-derived rates, simulation of known offsets, matching, follow-up counts, waveform-based validation in an emergency-department dataset. |
| `pub12_program_comparison/` | Same name, different threshold call: QRS and QTc classification across ECG programs and public datasets | Threshold classification by several programs on the same recordings, repeat-versus-program disagreement, definition audit. |
| `shared_support/` | (not a manuscript) | Helper modules that the scripts of several works import (statistics core, catalog continuation, tests of the variant code). |
| `looplens_snapshot_2026-10-07/` | (not a manuscript) | Frozen research state of the measurement software, see below. |

Each work folder has its own `README.md` that lists the scripts one by one with the manuscript section or table that depends on their output, the SHA-256 of the original and of the published file, and the kind of cleaning applied.

## Two licenses

* The analysis scripts of the work folders, `shared_support/`, `repo_config.py`, `werkzeuge/` and `verify_snapshot_hash.py` are licensed under the MIT License (`LICENSE` in this folder).
* `looplens_snapshot_2026-10-07/` is licensed under the PolyForm Noncommercial License 1.0.0 (`looplens_snapshot_2026-10-07/LICENSE`, required notice "Copyright 2026 Leo Probst"). It is a frozen research state, not maintained software.

## The measurement software

All marker values come from LoopLens 10.1. The research state that produced them is published in `looplens_snapshot_2026-10-07/`: the release of 7 October 2026 byte-identical, plus `patch_2026-10-11/` with the files that differ in the executed run. Hashes, the difference between the two states and everything that was left out are described in `looplens_snapshot_2026-10-07/README.md`; `python verify_snapshot_hash.py` recomputes them. Later development of LoopLens is not part of this repository.

## Data access

None of the datasets is redistributed. Use your own copies and point the scripts at them (see "Running the scripts").

| Dataset | Used in | Access |
|---|---|---|
| MIMIC-IV-ECG 1.0 (doi 10.13026/4nqg-sb35) | A, B, S5 | open on PhysioNet, ODbL 1.0 |
| MIMIC-IV 3.1 (doi 10.13026/kpb9-mt58) | A, B, S5 | PhysioNet credentialed health data license 1.5.0 |
| MIMIC-IV-ECHO 1.0 (doi 10.13026/nrjh-5r77) | B | PhysioNet credentialed health data license 1.5.0 |
| MC-MED 1.0.1 (doi 10.13026/wvyw-g663) | S5 | PhysioNet credentialed, data use agreement required |
| PTB-XL 1.0.3 (doi 10.13026/kfzx-aw45), PTB-XL+ 1.0.1 (doi 10.13026/g6h6-7g88) | A, PUB-12 | open on PhysioNet, CC BY 4.0 |
| LUDB 1.0.1 (doi 10.13026/eegm-h675) | A, B, PUB-12 | open on PhysioNet, ODC-By 1.0 |
| PTB Diagnostic ECG Database 1.0.0 (doi 10.13026/C28C71), Autonomic Aging 1.0.0 (doi 10.13026/2hsy-t491), ECGRDVQ 1.0.0 (doi 10.13026/C2HP45), ECGCIPA 1.0.0 (doi 10.13026/C2967M) | A | open on PhysioNet |
| EchoNext 1.1.1 (doi 10.13026/7cfw-d091) | PUB-12 | registered users, PhysioNet restricted health data license 1.5.0 |

Credentialed and restricted datasets require the credentialing and data use agreements of PhysioNet. Only aggregate results derived from them appear in the manuscripts.

## Dependencies

Python 3.12. `requirements.txt` lists the third-party packages the scripts import, with the versions of the environment in which they were prepared. The measurement package needs its own frozen list (`looplens_snapshot_2026-10-07/requirements-lock-win-py312.txt`).

```
python -m pip install -r requirements.txt
```

## Running the scripts

The scripts expect your own copies of the data and of the LoopLens output tables. They do not contain paths.

1. Copy `config.example.toml` to `config.toml` and fill in the `[paths]` table. Keys start with the work they belong to (for example `pub12.ptbxl_dir`, `a.mimic_ecg_dir`, `b.development_table`); the key name and the script that uses it tell what is expected. Optionally set `ECG_CODE_CONFIG` to the location of the file.
2. Put the repository root on `PYTHONPATH` (and `looplens_snapshot_2026-10-07` for the measurement package). Importing `repo_config` does this for the cleaned scripts; the unchanged originals need it set by hand.
3. Run a script by its file or, for scripts with package-relative imports, as a module from the repository root, for example `python -m werkzeuge.a_testreplikation.aw163_run`. The scripts were written in a working tree with other folder names. The compatibility package `werkzeuge/` and `import_map.json` map the original names (`werkzeuge.<original folder>.<module>`) to the folders of this repository, so every import resolves without editing the scripts.
4. Where a script needs LoopLens output, it expects the LoopLens output table in long format with one row per recording and measure. The long format used by the benchmark scripts is defined in `pub12_program_comparison/benchmark/bench_schema.py` (columns `lauf_id`, `datensatz`, `aufnahme`, `cluster_id`, `ableitung`, `bezug`, `schlag_nr`, `werkzeug`, `werkzeug_version`, `konfig_hash`, `arm`, `vorverarbeitung`, `messgroesse`, `native_bezeichnung`, `einheit`, `wert`, `status`, `statusgrund`). The other scripts read parquet or CSV tables of marker values per recording written by the measurement package; the header of each script names the columns it uses.

The scripts were written for a specific local data layout. They are published for inspection and adaptation. Their results depend on data that cannot be shared, so no script is expected to run end to end without adaptation. Expected numbers are in the manuscripts and supplements.

Reference values. Several scripts check their results against expected values. Values that are published in a manuscript stay in the code. Values that are not published are not in this repository: the scripts read them from a local file (`reference_values.json`, or the file named by the key `reference_values`) through `repo_config.check_reference` and `repo_config.reference`. Without that file the checks are skipped with a note.

## Original files, cleaned files and translated files

Most analysis scripts were specified, hashed and then run. The work READMEs mark every file with one of three statuses and list the SHA-256 of the original and of the published file:

* original, unchanged: the file that ran, byte for byte, German comments included. Its SHA-256 equals the recorded one.
* cleaned original: the file that ran with only these edits: local paths replaced by `data_path(...)`, user names and cloud-folder names removed, long dashes replaced by hyphens, unpublished expected values moved to the local reference file, and unpublished numbers removed from texts. The logic is unchanged; the work README says which of these edits each file received, and the edits were checked by a token-level comparison with the original.
* translated: scripts that were not hash-bound, with comments, docstrings and messages in English and the same edits as above. Data-schema labels (column names, status labels) stay as they are in the data.

Identifiers such as "AW-..." and "BEF-..." in data keys, default run labels and original docstrings are internal run and finding numbers of the research project; they carry no meaning outside it and stay because the scripts' logic depends on them.

## What is not in this repository

* Data of any kind, and every result file.
* Test files of the analysis code that embed expected values, download helpers, and the packaging of the manuscripts.
* The later development of the measurement software.

## How to cite

DOI: to be assigned on release.

Until then cite the preprints (see the manuscripts) and this repository by its name and commit.
