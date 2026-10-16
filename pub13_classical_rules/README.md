# pub13_classical_rules

Working title of the manuscript: Testing classical ECG rules on routine machine measurements: only QTc above 500 ms met its criteria in both internal partitions

Scripts behind the pre-specified test of ten classical ECG rules (thirteen test units) on routine machine measurements of MIMIC-IV-ECG and its replication on CODE-15%. The evaluation script is the frozen version 3.2; the tests use synthetic data only. The folder is flat because the evaluation script checks the hashes of its helper files in its own folder at start.

## Data and inputs

MIMIC-IV-ECG 1.0 (machine measurements), MIMIC-IV 3.1 (hosp and icu modules: outcomes, diagnoses, deaths), CODE-15% (replication of two units). The explicit diagnosis code lists of the freeze are included as `codelists/codelisten_k1.csv` (codes and titles only, no patient rows). None of these data is part of this repository. Set the locations in `config.toml` (see the root `README.md`); the keys used by this folder start with `pub13.` in `config.example.toml`.

## Scripts

11 files: 8 original and unchanged, 3 cleaned originals, 0 translated. The status column uses the three terms explained in the root README; cleaned originals and translated files received some of five edit types: paths replaced by data_path, user or cloud-folder names removed, long dashes replaced, unpublished expected values moved to the local reference file, unpublished numbers removed from texts. The last column names the part of the manuscript or supplement whose numbers depend on the script, as far as it could be traced (`support module` = used by other scripts; `not traceable` = no reported item found). Folder names are descriptive; the scripts themselves keep their original file names because they import each other by name.


### `./`

Evaluation script, report writer, rule functions, scaffold, adapter, code-list builder and tests.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `adapter_cache.py` | original, unchanged | Adapter from the feature cache of the measurement software to the measurement record of the rule functions (voltage and interval columns only). | Methods (voltage and interval inputs) |
| `k1_auswertung.py` | cleaned original | Evaluation script of the frozen version 3.2: builds the analysis tables from the machine measurements and diagnosis codes, applies the rule functions, fits the pre-specified risk-ratio, Cox and logistic models with person-cluster bootstrap and Bonferroni limits, masks small cells and writes the aggregate results; checks the hashes of its bound helper files at start. | Methods; Table 1; Supplement S2 and S3a to S3f |
| `k1_bericht.py` | cleaned original | Writes the markdown report of all verdicts, estimates, intervals and secondary analyses from the aggregate results; computes nothing and is not part of the frozen script. | support module for Supplement S3a to S3e (report layout) |
| `k1_codelisten.py` | original, unchanged | Builds the explicit diagnosis code lists from prefix rules against the ICD dictionary table (codes and titles only). | Methods and Supplement S2 (outcome and exposure code lists) |
| `k1_geruest.py` | original, unchanged | Framework functions of the evaluation: seed from the hash of the bound specification, input-hash manifest check, mode guard for the locked test partition, small-cell masking, verdict classification. | Methods; Supplement S1 and S2 (binding, small-cell rule, verdict rules) |
| `regeln.py` | original, unchanged | The classical ECG rules as pure functions on one measurement record, following the wording of the primary sources, with three states (true, false, not evaluable). | Methods; Table 1 (the ten rules); Supplement S2 |
| `test_geruest.py` | original, unchanged | Tests of the scaffold functions on synthetic data. | support module for Supplement S1 (tests of the frozen script) |
| `test_k1_auswertung.py` | original, unchanged | Tests of the evaluation script on synthetic data, including an end-to-end synthetic world, the test guard, the small-cell mask, the Cox fit against lifelines and recovery of planned effects. | support module for Supplement S1 (tests of the frozen script) |
| `test_regeln.py` | original, unchanged | Unit tests of the rule functions on synthetic measurement records. | support module for Supplement S1 (tests of the rule functions) |

### `codelists/`

Explicit ICD-9 and ICD-10 code lists of the freeze (list, version, code, title). A code list is not patient data.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `codelisten_k1.csv` | original, unchanged | Explicit ICD-9 and ICD-10 code lists of the freeze (list identifier, version, code, title). | Methods and Supplement S2 (code lists) |

### `figures/`

Figure S1 of the supplement.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `abbildung1_pub13_v1.py` | cleaned original | Draws Figure S1: unadjusted and adjusted risk ratios for QTc above 500 ms in both partitions, from the aggregate results. | Supplement Figure S1 (S3g) |

## Checksums and cleaning

SHA-256 of every file as it was when it ran (original) and as published. Unchanged originals have one hash. For cleaned originals and translated files the cleaning column lists the edit types with the number of places; the edits were verified against the original by a token-level comparison, which allows only these edits.

| Script | Status | SHA-256 of the original | SHA-256 of the published file | Cleaning applied |
|---|---|---|---|---|
| `adapter_cache.py` | original, unchanged | `f4f651467e742169b844bdc65456bd365c1a99557dbd430bbae6c6a9e32c075d` | same file | - |
| `k1_auswertung.py` | cleaned original | `1d40d782c65ceeffd4be9c4380e4eef822cbc98aaa710a4d7bf340699bf773e8` | `67aaaf6acad88d53246320bf95f31b747d76c2042df9489d29ea160abed76943` | local paths replaced by data_path (11); user or cloud-folder names removed (8) |
| `k1_bericht.py` | cleaned original | `55e6954aeb9f29244be8153353a639d0bb642c24da9af1246b7fe2cca51f208b` | `14bdcd1c16692a6fc32afc4e77d769c12f191c986bc7746f6807bc088bda8470` | local paths replaced by data_path (1) |
| `k1_codelisten.py` | original, unchanged | `3245f2931fbddac091ccab8e217b60bfc9f258d993e377f9eb01e142708d7aba` | same file | - |
| `k1_geruest.py` | original, unchanged | `7736a0a7eece05f80f8f3c2e6239b63d4f5a1a244607c706d6828d321d44bd62` | same file | - |
| `regeln.py` | original, unchanged | `1e04fc99a5381512ade60654974b7a252548beee804b83225ef0ff50c9a58e11` | same file | - |
| `test_geruest.py` | original, unchanged | `8003e909b4623e809d827054b13733516c698ffc8bb2193879e23e2f60ab835f` | same file | - |
| `test_k1_auswertung.py` | original, unchanged | `c44ee1724f3ef30ecbc0aa9cb52f7c987983fa4844df5191ae68a9f574670cf2` | same file | - |
| `test_regeln.py` | original, unchanged | `fcbe4fa7a79cf30d7cfeca7251ff03a6453b7b006e87c6682667f8423637f92f` | same file | - |
| `codelists/codelisten_k1.csv` | original, unchanged | `14cfc4fbc71d37607cee05cd51c96ee99e0b819ca887d84017b42a416573d516` | same file | - |
| `figures/abbildung1_pub13_v1.py` | cleaned original | `86b3ec1bc75f7de7929285a100c1d784aa764c88832e3d586b52168121647d66` | `b8c476b6d9f569ec4e2e68304b3a595ed41d7578dd2a2d48b8d07e00a4bef5f1` | local paths replaced by data_path (1); user or cloud-folder names removed (4); house-style module replaced by a plain matplotlib style |

## Notes on the frozen version

* The evaluation script `k1_auswertung.py` is the version frozen on 15 October 2026 (SHA-256 of the original `1d40d782c65ceeffd4be9c4380e4eef822cbc98aaa710a4d7bf340699bf773e8`). The published file is the cleaned copy; its hash differs from the original only because local paths were replaced (see the checksum table).
* At start the script checks the SHA-256 of `regeln.py`, `k1_geruest.py` and `adapter_cache.py` against the bound values. These three files are published byte-identical, so the check passes. For that reason `regeln.py` (one drive path in a docstring) and `k1_geruest.py` (one drive path in a message text) were not cleaned; the paths are the standard result folders, without personal reference.
* The check of the hash of the bound specification needs the specification file, which is not part of this repository (`--spec`).
* The tests use synthetic data only and run with `python -I -m pytest -q test_k1_auswertung.py -p no:cacheprovider` in this folder.

## What is not in this folder

* The two experimental helper scripts of the working tree that the frozen version does not use, the intermediate archive copies of earlier script versions (3, 3.1) and the packaging of the manuscripts.
* All data, result tables, intermediate files and the local reference-value file.
* The unpublished expected values of 0 scripts: they were moved to a local file that is not published (see the root README, section Running the scripts).
* Modules shared with other works are in `../shared_support/`.

## Reproducing the numbers

The numbers are in the manuscript tables and the supplement. To reproduce an analysis, obtain the data, produce the LoopLens output tables with the snapshot, fill `config.toml` and adapt the scripts to your layout. Random seeds are set in the scripts and marked as such.

