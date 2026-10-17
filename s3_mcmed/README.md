# s3_mcmed

Working title of the manuscript: How much does the monitor T wave change per mmol/L of potassium? A within-patient analysis in emergency department patients with potassium below 6 mmol/L

Scripts behind a pre-specified within-patient analysis of serum potassium and the lead II T wave in the emergency department (MC-MED 1.0.1): the locked analysis script with its fixed measurement modules, the driver of the main run, the reference readings of the author, the post hoc analyses and the figure. The pre-specified analysis was locked on 16 October 2026 (20:07:45 local time), before any analysis of the relation between potassium and an ECG measure was run; a feasibility check on the training split preceded the lock and did not look at such a relation (Supplement S1b). Scripts of analyses that were added after the results were known are marked "post hoc" in the tables below and in their own headers.

## Data and inputs

MC-MED version 1.0.1 (PhysioNet; credentialed access under the PhysioNet Credentialed Health Data License 1.5.0, data use agreement required). The data set is not part of this repository and no signal, identifier or patient-level value is redistributed. The measurement algorithm is NeuroKit2 based and separate code; the frozen LoopLens snapshot of this repository was not used in this study. Set the locations in `config.toml` (see the root `README.md`); the keys used by this folder start with `s3.` in `config.example.toml`. The folder `data` of MC-MED must have `SHA256SUMS.txt` in its parent folder, because the locked script checks the input files against it. `s3.n3_dir` is the folder `n3/` of this directory, `s3.code_dir` is this directory.

## Running the scripts

Python 3.12.4 with numpy 2.5.3, pandas 2.3.3, scipy 1.18.1, statsmodels 0.15.0, duckdb 1.5.6, neurokit2 0.2.13, wfdb 4.3.1 and psutil 7.2.2 (the versions of the binding record, see `specification/BINDING_NOTE.md`). `aw197_messung.py` also imports `vitaldb` (1.7.2 in the environment in which the files were prepared) and the figure needs matplotlib; `requirements.txt` in the root lists both. Random numbers are numpy `default_rng` (PCG64) with the seeds of the configuration.

The scripts are run as files from this folder, with Python's isolated mode as in the headers, for example:

```
python -I s3_auswertung_v2_2.py selbsttest        # self-test on artificial data (40 checks)
python -I s3_auswertung_v2_2.py sperren           # write a lock file of your own, see below
python -I s3_auswertung_v2_2.py gesamtlauf --arbeiter 3
python -I s3_gesamtlauf_treiber.py pruefen        # readiness check of the driver
python -I s3_annotation_bauen.py selbsttest
python -I s3_posthoc_h2.py selbsttest
```

The mode names are German as in the original scripts. The main script needs the data of MC-MED and the folder of the private work files (`s3.work_dir`); the post hoc scripts also need the private database written by the main run. None of this can run without the credentialed data.

## Binding and the published copies

The locked script checks at every start the SHA-256 of the specification, of itself, of the three fixed modules (`s3_signal.py`, `aw197_messung.py`, `aw203_messung.py`), of its configuration and of the time and unit contract against a lock file, and stops on any difference. The lock file of the original run and the SHA-256 of every original file are in the table of checksums below and in `specification/BINDING_NOTE.md`.

Only the two modules in `n3/` are published byte for byte. The locked script, `s3_signal.py`, the specification and the lock file contain local paths and are published as cleaned copies, so their hashes differ from the recorded ones. Run unchanged, the published copies therefore stop at the lock check. To run the published files end to end, write a lock file of your own with the mode `sperren` from the published specification and files. That is a new binding of the published copies, not the original binding; it makes no claim that the numbers equal those of the original run. The post hoc scripts `s3_posthoc_h2.py` and `s3_posthoc_basis.py` also contain an assertion with the SHA-256 of the original locked script (`04ac15da...`); replace the constant by the hash of your copy.

## Check of the published copies on artificial data

The self-tests of the published files were run on artificial data in an environment with the packages above, with a configuration that points all keys to an empty folder (no real data). Results: `s3_auswertung_v2_2.py` 40 of 40 checks; `s3_auswertung_v2.py` 34 of 34; `s3_annotation_bauen.py` 10 of 10; `s3_annotation_einfluss.py` 12 of 12 (this includes the in-memory patch of the page text made by `s3_annotation_bauen.py`); `s3_annotation_einfluss_auswerten.py` 7 of 7; `s3_posthoc_h2_oeffentlich.py` all checks; `s3_posthoc_basis_pruefung.py` 6 of 6; `s3_gesamtlauf_treiber.py` 8 of 9 (the check of the memory reservation tool fails because that tool is not published). `s3_posthoc_h2.py` passed 16 checks and `s3_posthoc_basis.py` 9 of 10; each fails only the assertion that pins the SHA-256 of the original locked script, and `s3_posthoc_h2.py` then stops because no lock file exists. The figure script ran on a synthetic input that holds only the published values and wrote the figure files, which also tested the plain style that replaces the house-style module. The self-tests of `s3_posthoc_zusatz.py`, `s3_laden.py`, `s3_zaehlung.py`, `s3_aggregat.py`, `s3_signal.py` and `s3_annotation_auswertung.py` need data or have no self-test mode.

## Scripts

21 code files and 3 documents: 2 original and unchanged, 19 cleaned originals, 0 translated. The status column uses the terms of the root README. The last column names the part of the manuscript or supplement whose numbers depend on the script, as far as it could be traced (`support module` = used by other scripts; no reported item = no reported number depends on it). The scripts keep their original file names because they import each other by name; the comments, messages and German mode names are those of the original run.

### `./`

Locked analysis script with the driver of the main run, the feasibility scripts of the training split, the scripts of the reference readings, the post hoc scripts and the helper files of the run.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `s3_auswertung_v2_2.py` | cleaned original | Locked analysis script (specification v2.2, pipeline label aw255-v3). Checks the lock file and the input hashes, loads the three splits into a private database, finds and measures the ECG blocks with the fixed modules, builds the pair tables and runs the pre-specified models (within-patient slope with patient-cluster bootstrap, sensitivity analyses, negative controls, repeatability, the secondary questions) and writes the masked aggregate output. Contains a self-test on artificial data with 40 checks. | Methods; Results; Table 1; Supplement S1a and S2 to S8 and S10 (every locked estimate) |
| `s3_signal.py` | cleaned original | Fixed module of the locked run: reads lead II segments, applies the quality rules of a 10 s block and, in its own modes, ran the feasibility measurements on the training split (block quality, calibration of the noise threshold, measurability). | Supplement S1b and S2 (ECG segments and windows; noise threshold of the configuration) |
| `s3_gesamtlauf_treiber.py` | cleaned original | Driver of the main run (not bound): binding through the script, readiness check, start through signal files, memory reservation, state file, and the optional preparation and evaluation of an annotation set. Contains a self-test on artificial data with 9 checks. | Supplement S1a (binding and run); no reported number |
| `s3_wacht.py` | cleaned original | Watcher that restarts the driver after a crash, at most 5 times in 60 minutes, and never without the start signal. | support module for the driver; no reported item |
| `start_wmi_s3.ps1` | cleaned original | Starts a detached process through WMI for the watcher and the driver. | support module for the driver; no reported item |
| `s3_laden.py` | cleaned original | Pre-lock feasibility check on the training split: loads laboratory results, orders and visits of the training split into a private database. The test split is refused. | Supplement S1b |
| `s3_zaehlung.py` | cleaned original | Pre-lock feasibility check: counts laboratory availability, pairs, ECG coverage in time, treatments and disturbing factors on the training split, without any ECG measure. Masked output. | Results (planned spread of the potassium change); Supplement S1b |
| `s3_aggregat.py` | cleaned original | Pre-lock feasibility check: aggregates block quality, measurability and repeatability of the measurement chain together with the pair and extreme-value tables of the training split. Masked output. | Results (planned spread of the potassium change); Supplement S1b |
| `s3_auswertung_v2.py` | cleaned original | Earlier version of the analysis script (specification v2), superseded by v2.2 before the lock. It is imported by s3_annotation_bauen.py for the block search and the polarity rule used in the draw of the reference strips. No reported estimate comes from it. Contains a self-test with 34 checks. | Supplement S9 (draw of the strips) |
| `s3_annotation_bauen.py` | cleaned original | Reference reading of the author (specified in section 10 of the specification): determines the eligible blocks in the training and validation splits, draws 100 views (90 strips, 10 of them shown twice, stratified by QRS direction, fixed seed), writes the key and one self-contained HTML page with the signals. The script contains no data. Self-test with 10 checks. | Supplement S9 (Material) |
| `s3_annotation_auswertung.py` | cleaned original | Descriptive comparison of the reader's T amplitude with the measurement chain per strip: bias, limits of agreement, ICC(1,1), Spearman correlation, repeat differences of the strips shown twice, by QRS direction. No gate. The chain is called through the checked entry of the locked script. | Results (Reference reading); Supplement S9 |
| `s3_annotation_einfluss.py` | cleaned original | Post hoc. Draws the strips of the second blind reading: both ends of the influential primary pairs (rule: absolute DFBETAS of the slope above 2/sqrt(n), as in the locked script) and the same number of filler pairs, and builds the page. Self-test with 12 checks. | Results (Dependence on influential pairs); Supplement S5 (Reader check of influential pairs) |
| `s3_annotation_einfluss_auswerten.py` | cleaned original | Post hoc. Compares the reader with the measurement chain at the influential and the filler ends and recomputes the slope with the reader's values in the variants of the supplement. Self-test with 7 checks. | Results (Dependence on influential pairs); Supplement S5 (Tables S13 and S14) |
| `s3_posthoc_h2.py` | cleaned original | Post hoc. Robustness of the steepness slope: sensitivity analyses, series of bootstrap seeds, repeatability of steepness, controls, comparison of the scales of steepness and amplitude, and the robust repeatability limit of the amplitude. Imports the locked script unchanged. | Results (Steepness); Supplement S4 (robust repeatability limit) and S7 |
| `s3_posthoc_h2_oeffentlich.py` | cleaned original | Post hoc. Builds the public aggregate version of the output of s3_posthoc_h2.py with the small-cell rule (counts of 1 to 9 as <10, differences and complements rounded down to tens) and the table fragments. | Supplement S7 (tables) |
| `s3_posthoc_basis.py` | cleaned original | Post hoc. Describes the primary pairs and the other valid pairs (groups A to D) per patient with the small-cell rule. Imports the locked script unchanged and reads the private database read only. | Table 2; Supplement S11 |
| `s3_posthoc_basis_pruefung.py` | cleaned original | Post hoc. Independent check that no count of 1 to 9 can be derived from the published tables of S11 (shares, differences, complements, text). Reads no data. | support module for Supplement S11 (small-cell check); no reported number |
| `s3_posthoc_zusatz.py` | cleaned original | Post hoc. Adds triage level, disposition, length of stay and emergency department diagnoses from the visit file for the groups of s3_posthoc_basis.py. | Table 2; Supplement S11 |

### `n3/`

The two fixed measurement modules of the locked run, published byte for byte.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `aw197_messung.py` | original, unchanged | Fixed module: measurement library (R peak detection, beat selection, window checks, noise and rhythm rules, median beat). Imports vitaldb when it is loaded. | Methods (ECG measurement); Supplement S2 (Measurement algorithm) |
| `aw203_messung.py` | original, unchanged | Fixed module: measurement chain version 2: R-aligned median beat, T amplitude and steepness of the descending limb, baseline from the PR segment. | Methods (ECG measurement); Supplement S2 (Measurement algorithm) |

### `figures/`

Figure 1 of the manuscript.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `ABBILDUNG_S3_v5.py` | cleaned original | Draws Figure 1 (expected and observed change in T amplitude per change in potassium) from the masked aggregate output and checks every drawn number against the text of the manuscript. | Figure 1 |

### `specification/`

The specification, the lock file and a short English summary of the binding record.

| File | Status | What it is | Manuscript part |
|---|---|---|---|
| `SPEZIFIKATION_S3_MCMED_v2_2.md` | cleaned document | The pre-specified analysis, version 2.2, in German as bound: hypotheses, pairs, treatment categories, windows, models, categories of the result, controls, precision, interpretation table, the optional reference reading, review table. | Methods; Supplement S1a |
| `spec_lock_s3_v2_2.json` | cleaned document | Lock file written at the binding: hashes, the configuration `cfg` with all parameters and seeds, the contract and the hashes of the input files listed in `SHA256SUMS.txt` of MC-MED. | Supplement S1a |
| `BINDING_NOTE.md` | new text | English summary of the internal binding record: what was bound and when, the hashes, the software, the inputs, what had been seen before, the helper files, the order of the run. | Supplement S1a |

## Checksums and cleaning

SHA-256 of every file as it was when it ran or was bound (original) and as published. Unchanged originals have one hash. For cleaned files the cleaning column lists the edits with the number of places. The edits of the Python files were checked by a token-level comparison with the original that allows only path replacements and text changes in comments and strings; it reports no other difference than the ones in the list below. The PowerShell file was compared by eye.

| File | Status | SHA-256 of the original | SHA-256 of the published file | Cleaning applied |
|---|---|---|---|---|
| `s3_auswertung_v2_2.py` | cleaned original | `04ac15dae0f8a6a0a9b2d154e43b4ff3d240acd410d10707498fa53d48c9dcea` | `db2b56666e33e17203ca54d9ea5188aabeb0a93e5e2a32b0bf82b90cf385cbe0` | local paths replaced by data_path (4); path, user and cloud-folder text in comments and texts replaced by placeholders (1) |
| `s3_signal.py` | cleaned original | `6450c7e5d0a0205f4f2333b93c992354bb2df41f84f2be811ca1165fe5784832` | `db4ae93e946e0c788844efac273a6b1e71bdfd46a947dc57af701d66a2695fc7` | local paths replaced by data_path (3); path, user and cloud-folder text in comments and texts replaced by placeholders (3) |
| `s3_gesamtlauf_treiber.py` | cleaned original | `e1c4baf9e623d4cb5e82196a981c233fb10b5a48fb3939f3390d090395cccec5` | `c07a4f7b2b31255f0b904e9bff651361efda2b307aceb1d2ccca9aa7f9a89f89` | local paths replaced by data_path (4); path, user and cloud-folder text in comments and texts replaced by placeholders (1) |
| `s3_wacht.py` | cleaned original | `6104e2a8841990ad3845c9c9638a75c951aa686df4f4885450e7a4f964514c06` | `2949e11e7e9cead303a8d43c0691b71d892d8862446c55b2104ab1cb8f0d17d9` | local paths replaced by data_path (2); path, user and cloud-folder text in comments and texts replaced by placeholders (1) |
| `start_wmi_s3.ps1` | cleaned original | `51a3556ffae82ab61286cd5f2f9286e6748d3d986f87411d849ad5affb624d7d` | `9905ed2891f71e1ca8416943c0e3c74d839bf9b236a76e4c8a9fe7553be7bfa1` | default working directory replaced by the current directory |
| `s3_laden.py` | cleaned original | `17f5799b3bff7a7a83bc41bc2331cde42f1ab9ad6602c6631c24e46f819c9320` | `ae9a457b847f24c8bbe0aaa7405a0420bf365094de10385dd8ce634f6737f87e` | local paths replaced by data_path (2); path, user and cloud-folder text in comments and texts replaced by placeholders (1) |
| `s3_zaehlung.py` | cleaned original | `1f15bd5394d4c1376295e005d6338947f2b9f5603310cedfcc517682e8f13af5` | `f62606bb41a8a81d44be9b8640e5665d6ab42cb13043692dfbb9ef33d67aabed` | local paths replaced by data_path (2); path, user and cloud-folder text in comments and texts replaced by placeholders (2) |
| `s3_aggregat.py` | cleaned original | `f12292e6587d73238d33df6e155d00e4fc04cfdfb57abad2c25aaf375cdc9c57` | `129a83a1322f532e39bcbd79237fe16b0d10b976db740931ba9cd4f95e70f24f` | local paths replaced by data_path (1); path, user and cloud-folder text in comments and texts replaced by placeholders (1) |
| `s3_auswertung_v2.py` | cleaned original | `65ddeaf004bd33d983f03037077980555cc10ddda9d2214610d51c13a8fbeb2c` | `74dc2d8ca354d2374da07fd68cbafeb7b22f636bd7adfd68393d621e7e5bee4e` | local paths replaced by data_path (4); path, user and cloud-folder text in comments and texts replaced by placeholders (1) |
| `s3_annotation_bauen.py` | cleaned original | `117ed5ff6f75d1715df1ff88bc67be63748c3cd16cdfb98237c2f0b32ddce5cb` | `95a79da4352b6246b02de1793ebddcbba6ca40b9215427a4b76d207217f99448` | local paths replaced by data_path (2); path, user and cloud-folder text in comments and texts replaced by placeholders (4); folder hint in two page texts shortened to s3\annotation\ (the influence script patches this text) |
| `s3_annotation_auswertung.py` | cleaned original | `fe40dc6db45c48cd683edb3c0bec3896409c250bcb92db1b26567872d4ee1db0` | `c8a4155575f1b9108d713e9f9f697bffed3ef2bc8282026d4c362a496e038465` | local paths replaced by data_path (4); path, user and cloud-folder text in comments and texts replaced by placeholders (4) |
| `s3_annotation_einfluss.py` | cleaned original | `fd5a4104f18fb35c534b824cc24ad435f401b0273097bd051067bdd9b0c4145a` | `bfe7127ac938ea2ccdeb825f34363e1d3c332010b8e7d07a909fb5ad79e6da78` | local paths replaced by data_path (6); path, user and cloud-folder text in comments and texts replaced by placeholders (2) |
| `s3_annotation_einfluss_auswerten.py` | cleaned original | `1663d9083c40f1c53694c4b8b8e270cb9e8231a3dfaf7a315a9277ff1c631b30` | `a13a26feedceb0a7d1027820ea2f6394d6f6b75149d17a0f607e573f3da86368` | local paths replaced by data_path (1); path, user and cloud-folder text in comments and texts replaced by placeholders (2) |
| `s3_posthoc_h2.py` | cleaned original | `efc6b4e1e6fbb0f55ffaa01a3dfad3426102027385bd73cd06a68cd6b8c60f55` | `feb844cf3a08940de2212f525edcfcfaaca9c3d0bd8b25874f6b6912762b8814` | local paths replaced by data_path (2); path, user and cloud-folder text in comments and texts replaced by placeholders (3) |
| `s3_posthoc_h2_oeffentlich.py` | cleaned original | `4df3c309c95d960750a7e7bf08e5132f6ab94f9b4406022c42f5b59b6c000811` | `4d6d051ec0aac0018988bf31632fa09199aea9ce9206b31978d35af610787c61` | path, user and cloud-folder text in comments and texts replaced by placeholders (1); unpublished bootstrap P value in a table text removed; estimate digits in a formatting test replaced by artificial numbers |
| `s3_posthoc_basis.py` | cleaned original | `cdc4a31afd2ed10281b9a631002ba946fc9f28bbedfbdede316382dda0161fb7` | `f85d6b971409470fa3b2803dbee0ae7b5b6df42e6056ec38591c6e8a814c077b` | local paths replaced by data_path (2); path, user and cloud-folder text in comments and texts replaced by placeholders (2) |
| `s3_posthoc_basis_pruefung.py` | cleaned original | `058bd107a85e0e7a3945c3b8b2de2966660afd8055824c6302bcc97e6f5a9532` | `1f90aab166813d2395fcb4ba8779d7d9a5ccdc9109f5b2e7aba5d095fb66a09a` | local paths replaced by data_path (2) |
| `s3_posthoc_zusatz.py` | cleaned original | `a23acaad299b8e06e1f7837bb497aaf83efa03c27bb1007b7098ecf11ca74999` | `3cf404f6a99562bf2fa227a721952f671572c786caac444436f06a2b98a42265` | local paths replaced by data_path (2); path, user and cloud-folder text in comments and texts replaced by placeholders (1) |
| `n3/aw197_messung.py` | original, unchanged | `03e2b2d28aa04a231c6d3280376255e329fdf4d8fa28465f0b4891ccb35e7f67` | same file | - |
| `n3/aw203_messung.py` | original, unchanged | `37e3b06a9dd84ba4ab2f16d98c940ad04d42325a8348d12ed21bf04e0614f193` | same file | - |
| `figures/ABBILDUNG_S3_v5.py` | cleaned original | `26e5f3458650c9fa6e2d5451e41b3b3af02a97cc61c5994a9e044d9e830a5606` | `5581d9f772b4d385655be06436d2298965aabd5f26a50fcb6003513b7e90fe60` | local paths replaced by data_path (3); path, user and cloud-folder text in comments and texts replaced by placeholders (2); author name replaced (1); private house-style module replaced by a plain matplotlib style; manuscript text and output folder read from configured paths |
| `specification/SPEZIFIKATION_S3_MCMED_v2_2.md` | cleaned document | `8e09350dab33bc5ce81e2a11127899c2ba8afa84de5a5b4f6833a3577e42381e` | `741f35f274de955f2062379022388c17dc2a81d7b6a352dbc05a966f6997bb89` | internal identifiers and finding numbers removed; names of review tools and roles replaced by neutral words; private file names and folders removed; descriptive counts of the pre-lock feasibility check that are not in the manuscript left out (46 replacement rules); no change to a hypothesis, a model, a threshold or a rule |
| `specification/spec_lock_s3_v2_2.json` | cleaned document | `bed63aff161d627afe62c256705927b48c14b134bd8f7b02f7a7e1d766ddd886` | `059c7fb63f44d6a1091bca8744fd3c2644da8b59b7396a75227f8f2ae467b562` | one local path to `SHA256SUMS.txt` replaced by a placeholder (1) |
| `specification/BINDING_NOTE.md` | new text | `7f7a011713894de7ffed8aec05477dc6b43e32abf33c5a88fc986abdae73e322` (internal record, not published) | `54939b17e8ab782064908ef8da9196f2e76e72ea607bc1accc506b0ef3ca940f` | written for this repository; contains no path, identifier or internal note |

Edits beyond the standard types, all in files that are not hash-bound by the lock:

* `s3_annotation_bauen.py`: two texts of the HTML page that named the private output folder now name `s3\annotation\` only. `s3_annotation_einfluss.py` patches exactly this text in memory, so the hint was shortened instead of replaced by a placeholder.
* `s3_posthoc_h2_oeffentlich.py`: a bootstrap P value that is not reported in the manuscript was removed from a table text, and the digits of an estimate in a formatting test were replaced by artificial numbers.
* `start_wmi_s3.ps1`: the default working directory is now the current directory.
* `figures/ABBILDUNG_S3_v5.py`: the private house-style module of the original script is replaced by a plain matplotlib style defined in the file (look of the figure may differ, content does not), and the output folder and the local copy of the manuscript text come from the configuration (`s3.figure_out_dir`, `s3.manuscript_md`) instead of the private working tree.

## What is not in this folder

* Data of any kind, the private database and every result file, the key of the reference readings and the exported reader files.
* The versions v1 and v2.1 of the analysis script (superseded before the lock and not used by any other published script) and the internal review reports.
* The tool for the memory reservation (`s3.ram_reservation_py`), used by the driver and by two post hoc scripts, and the private register of data accesses (`s3.test_access_register_md`), which some post hoc scripts check. Both are specific to the machine of the author.
* The tables and text fragments written into the supplement from the outputs of the post hoc scripts, and the packaging of the manuscript.
* Later development of the measurement software; see the root README.

## Notes on the files

* The docstring of `s3_auswertung_v2_2.py` says that the script has not yet run on real data. This sentence is from the time of the binding and is left as it was.
* Names with the prefix AW and a number in docstrings, and the labels aw197, aw203 and aw255-v3, are internal run numbers and version labels of the research project; the root README explains them. In the specification the prefixed names are removed.
* Line numbers in section 14 of the specification refer to the original script.
* The record of the binding lists a different SHA-256 for `s3_wacht.py` (`539a6b7519078577b08a2c21f6f69a22de9e967fe88a60d755f0a551b7cfcf3d`) than the file whose hash is given above. The file was edited after the binding, before the main run: its module docstring contained a Windows path that Python read as an escape sequence, so the watcher stopped at start; the docstring was made a raw string. It restarts the driver and does not enter the analysis.
* `s3_posthoc_basis.py` and `s3_posthoc_basis_pruefung.py` were edited on 17 October 2026 after the run recorded for Table 2 and Supplement S11. Before release, the published (final) version of `s3_posthoc_basis.py` was rerun on the same data, and the tables of `s3_posthoc_zusatz.py` were rebuilt; the public outputs were identical to those of the recorded runs apart from the script hash and the time stamp.
* The selection record of the first reference reading (not published) lists the SHA-256 `f16f31d7d2f75bc286a659c0b6c37f3cee019bc356d68a87b306a20cb057c753` for `s3_annotation_bauen.py`; the file above has a different hash. The record notes that the wording of the rule for the repeated strips was clarified after the draw, with the draw, the key and the selection hash unchanged.
