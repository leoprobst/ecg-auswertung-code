# Binding note for the pre-specified analysis (specification v2.2)

This is a short English summary of the internal binding record of the analysis. It contains no data, no identifiers and no local paths. The internal record (SHA-256 of the original `7f7a011713894de7ffed8aec05477dc6b43e32abf33c5a88fc986abdae73e322`) is not published.

## What was bound, and when

The specification (version 2.2), the analysis script, the three fixed measurement modules, the configuration of the script and the time and unit contract were bound together on 16 October 2026 at 20:07:45 local time (18:07:45 UTC). The binding wrote a lock file, `spec_lock_s3_v2_2.json`, by calling the mode `sperren` of the script through the driver. The file was held outside the project folder and has no independent time stamp. No analysis was run at the binding, the test split was neither opened nor hashed, and the check of the lock file after writing it passed.

| Object | SHA-256 |
|---|---|
| Specification `SPEZIFIKATION_S3_MCMED_v2_2.md` | `8e09350dab33bc5ce81e2a11127899c2ba8afa84de5a5b4f6833a3577e42381e` |
| Script `s3_auswertung_v2_2.py` (self-test 40 of 40 checks) | `04ac15dae0f8a6a0a9b2d154e43b4ff3d240acd410d10707498fa53d48c9dcea` |
| Module `s3_signal.py` | `6450c7e5d0a0205f4f2333b93c992354bb2df41f84f2be811ca1165fe5784832` |
| Module `aw197_messung.py` | `03e2b2d28aa04a231c6d3280376255e329fdf4d8fa28465f0b4891ccb35e7f67` |
| Module `aw203_messung.py` | `37e3b06a9dd84ba4ab2f16d98c940ad04d42325a8348d12ed21bf04e0614f193` |
| Configuration `cfg` of the script (all measurement parameters, seeds, thresholds) | `12a93e65e801a48094bec1c454f4ea1323fd34983e249599afabf0c9b826f823` |
| Time and unit contract | `c282902b12e36b833bf16dc7b3ae58b9f98936b1468bbc4b7b43bc6c0f522aa0` |
| Lock file `spec_lock_s3_v2_2.json` | `bed63aff161d627afe62c256705927b48c14b134bd8f7b02f7a7e1d766ddd886` |
| Pipeline label | `aw255-v3` |

The published copies of the specification, the script, `s3_signal.py` and the lock file differ from these originals by the edits listed in the README of this folder, so their hashes differ. The two modules `aw197_messung.py` and `aw203_messung.py` are published byte for byte.

## Software of the bound run

Python 3.12.4, numpy 2.5.3, pandas 2.3.3, scipy 1.18.1, statsmodels 0.15.0, duckdb 1.5.6, neurokit2 0.2.13, wfdb 4.3.1, psutil 7.2.2. Random numbers come from numpy `default_rng` (PCG64) with the seeds fixed in the bound configuration. The environment in which the files were prepared also holds vitaldb 1.7.2 and matplotlib 3.11.2; `aw197_messung.py` imports vitaldb when it is loaded, but no VitalDB data enter this analysis. The versions of these two packages are not recorded in the lock file.

## Inputs

The binding refers to the file `SHA256SUMS.txt` of MC-MED 1.0.1 (SHA-256 `ebcb7c677a74c3ea9f8371d45329490b16e6dc62fc4275b8d0aa2fd32f274070`) and carries the hashes listed there for `labs.csv`, `orders.csv`, `visits.csv`, `numerics.csv` and the three split files. Before any visit is loaded, the script checks all seven files against `SHA256SUMS.txt` and stops if a file is missing or differs. Hashes of the waveform files were not part of the binding; the headers are read during the run.

## What had been seen before the binding

A feasibility check on the training split counted pairs, laboratory availability, ECG coverage and signal quality, and looked at the repeatability of the measurement and at the marginal distributions of the outcomes. It did not look at any relation between potassium (or any other exposure) and an ECG measure, did not read the validation or test split, and ran no negative control. The expectations for the hypotheses came from two earlier analyses of the author in other databases and were therefore not blind. Supplement S1b of the manuscript describes this in the same terms.

## Helpers that are not bound

| File | SHA-256 | Task |
|---|---|---|
| `s3_gesamtlauf_treiber.py` | `e1c4baf9e623d4cb5e82196a981c233fb10b5a48fb3939f3390d090395cccec5` | Driver: binding, readiness checks, start with signal files, memory reservation, state; optional preparation and evaluation of the reference annotation (self-test 9 of 9 checks) |
| `s3_wacht.py` | `539a6b7519078577b08a2c21f6f69a22de9e967fe88a60d755f0a551b7cfcf3d` at the binding | Watcher that restarts the driver after a crash, never without the start signal |
| `start_wmi_s3.ps1` | `51a3556ffae82ab61286cd5f2f9286e6748d3d986f87411d849ad5affb624d7d` | Detached start through WMI |

The driver calls only bound functions and sets no configuration value. `s3_wacht.py` was edited after the binding; the SHA-256 of the file before cleaning is `6104e2a8841990ad3845c9c9638a75c951aa686df4f4885450e7a4f964514c06`. It restarts the driver and does not enter the analysis.

## Order of the main run

1. The reference annotation of the author (Supplement S9) was done in the morning of 17 October 2026, before the main run, without knowledge of potassium values or pipeline values.
2. The run starts only if the lock file is valid, a start signal file exists, the annotation result file exists or a waiver file is set, and no stop file exists.
3. The script then checks the lock file, checks the inputs against `SHA256SUMS.txt`, loads all splits into a private database, checks the time contract, measures the samples (resumable; measurements carry the hash of the lock file, other measurements are moved to archive tables), builds the tables, runs the analysis and writes the masked output without identifiers (small-cell rule for counts of 1 to 9).
4. If the script refuses (lock file, input hash, time contract, module hash), the driver sets a stop file and does not restart.

The driver is configured for 3 worker processes. Nothing in the bound analysis was changed after the binding; analyses added after the results were known are post hoc and marked in the README.

## Known points

Section 14 of the specification lists two known points. The first is a gap in the check of `chain(cfg)`: a change of the parameters of `aw203_messung.py` alone is not detected when the configuration is unchanged; the regular run has no trigger for it, and the module bytes are hashed at run time against the lock file. The second is that `annotation_vorbereiten` of the script is very slow for the annotation; the reference annotation was therefore prepared with the separate script `s3_annotation_bauen.py`.
