# a_measurement_audit

Working title of the manuscript: Repeat variation and definition dependence of ECG/VCG markers: an exploratory audit of one implementation (LoopLens 10.1)

Scripts behind the measurement audit of the LoopLens marker catalog: repeat variation in pairs of routine-care ECGs, dependence of marker values on the reconstruction matrix and on boundary shifts, boundary errors against annotated recordings (LUDB), the replication on the held-out test partition, and the external repeat pairs in PTB-XL.

## Data and inputs

MIMIC-IV-ECG 1.0 and MIMIC-IV 3.1 (clinical context), PTB-XL 1.0.3, LUDB 1.0.1, the PTB Diagnostic ECG Database, Autonomic Aging, ECGRDVQ and ECGCIPA. The marker values themselves are LoopLens output tables produced with the snapshot in `../looplens_snapshot_2026-10-07/`. None of these data is part of this repository. Set the locations in `config.toml` (see the root `README.md`); the keys used by this folder start with `a.` in `config.example.toml`.

## Scripts

64 files: 10 original and unchanged, 22 cleaned originals, 32 translated. The status column uses the three terms explained in the root README; cleaned originals and translated files received some of five edit types: paths replaced by data_path, user or cloud-folder names removed, long dashes replaced, unpublished expected values moved to the local reference file, unpublished numbers removed from texts. The last column names the part of the manuscript or supplement whose numbers depend on the script, as far as it could be traced (`support module` = used by other scripts; `not traceable` = no reported item found). Folder names are descriptive; the scripts themselves keep their original file names because they import each other by name.


### `circular_marker_check/`

Check of the circular markers.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `a_r5k.py` | cleaned original | Computes repeat variation of periodic markers as the shortest angular distance with the period of each marker (95th percentile of the distance, circular-mean bias, random-pairing reference, coverage of the 1.96 x SD rule) and the catalog medians without the periodic markers, with person bootstrap. | Supplement S2i (Tables S2i-1 and S2i-2; Supplementary Data S2 repeat files); Section 2.3 and Table 2 (catalog without circular markers) |

### `delineator_input_check/`

Delineator-input check and the rerun of the stored boundary errors on LUDB.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `__init__.py` | original, unchanged | Marks the delineator-input check scripts as one importable package. | support module for the delineator-input check (Section 3.3; Supplement S2h) |
| `auswerten.py` | translated | Aggregates the per-ECG, per-record and per-beat results of the delineator-input check into paired shifts, limits of agreement, bootstrap intervals and the prespecified practical-dependence classes. | Section 2.4 and Section 3.3 (delineator-input paragraph); Supplement S2h (Table S2h-1); Supplement S0a (Section 3.3) |
| `kern.py` | translated | Runs the production fusion delineation on one record with the VCG input swapped (Kors, inverse Dower, none or measured Frank leads) without altering the release source, and returns fiducials and native markers. | support module for the delineator-input check (Section 3.3; Supplement S2h) |
| `ludb_repro.py` | translated | Checks that the Kors variant of the delineator-input check reproduces the stored LUDB beat-boundary errors of the earlier expert-boundary analysis. | Supplement S2h (reproduction statement for the LUDB rows); Supplement S10a (row Delineator-input check) |
| `ludb_run.py` | translated | Delineates every LUDB record with Kors, inverse Dower and no VCG input and matches the results to the expert boundary clusters. | Supplement S2h (LUDB arm, errors against expert boundaries); Section 3.3 (similar behavior in LUDB and PTB) |
| `mimic_run.py` | translated | Delineates a hash-ranked sample of development MIMIC-IV-ECG recordings with Kors, inverse Dower and no VCG input, with one checkpoint per ECG. | Section 3.3 (delineator-input paragraph); Supplement S2h (MIMIC arm, Table S2h-1) |
| `ptb_run.py` | translated | Delineates the first record of each PTB person that has Frank leads with Kors, inverse Dower, no VCG and the measured Frank VCG as input. | Supplement S2h (PTB arm and measured Frank leads as delineator input); Section 3.3; Supplement S0a (Section 3.3) |
| `tabellen.py` | translated | Formats the aggregate CSV results of the delineator-input check as Markdown tables for the report. | support module for Supplement S2h (Table S2h-1) |

### `error_variance_stages/`

Stages of the Deming-type correction cross-check.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `a_r1_stufe0_rahmen.py` | cleaned original | Rebuilds the analysis frame of the archived V6 R-peak time versus posterior wall thickness test from the bound original code and checks the algebra of the standardized Deming slope (SD(Y)/SD(X)). | Supplement S12a-1 (Correction of the T2 (i) estimate) |
| `a_r1_stufe1_fehlervarianz.py` | cleaned original | Estimates the within-person error variances of V6 R-peak time (consecutive ECGs up to 24 h apart) and posterior wall thickness (consecutive echocardiography reports up to 7 days apart) that define the error ratio of the Deming regression. | Supplement S12a-1 (Correction of the T2 (i) estimate) |
| `a_r1_stufe2_steigungen.py` | cleaned original | Computes the slopes in original units (ordinary least squares, Pearson r, reliability-corrected least squares, Deming with the measured error ratio and sensitivities) with person bootstrap. | Supplement S12a-1 (table of estimators) |

### `figures/`

Figure script of the manuscript.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `abbildungen_a_final_v27.py` | translated | Builds the final figures from the public aggregate tables: key-measure repeatability, ICC distribution, catalog matrix classes, matrix context, LUDB error distributions (copied unchanged from an earlier set) and the change in AUC of the repair variants. | Figure 1; Figure 2; Supplement Figure S1 (S2i); Figure S2 (S2g); Figure S3 (S3b); Figure S4 (S4) |

### `matrix_variants/`

Matrix-variant analysis (Kors against inverse Dower, boundary-shift experiment) and its report.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `aw150_analysis.py` | translated | Reproduces the catalog matrix classes from the stored paired values and computes the plus/minus 4 ms boundary-shift benchmark, cross-fitted affine calibration, fold concordance, matrix-to-repeat ratios, sampling-rate sensitivity and the common-sample area-angle agreement. | Table 4; Section 3.3 (area-angle agreement and boundary-shift paragraphs); Key findings; Supplement S2g (Tables S2g-1 to S2g-3, S2g-4, Table S2g-2 sampling rate); Figure S2; Supplementary Data S2 (matrix file) |
| `aw150_report.py` | cleaned original | Aggregates the controlled sensitivity results of the matrix comparison (boundary shifts of the delineation, expanded boundary panel, 500 Hz versus anti-aliased 250 Hz, fold cross-table, repeat-relative matrix and boundary deviation) and writes the marker-level matrix file and a context figure. | Supplement S2g (Figure S2, Tables S2g-1 to S2g-3, sampling-rate sensitivity); Supplementary Data S2 marker matrix file; Section 2.4 and 3.3 |
| `lauf.py` | cleaned original | Provides the checkpoint, hash and resource-budget helpers and a capped smoke-test runner for variant recalculations of the catalog markers. | support module for the matrix-comparison runs (Table 4; Supplement S2g) |

### `measurement_audit/`

Early measurement-audit helpers: loading, statistics, subgroup and boundary analyses.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `common.py` | translated | Shared helpers of the measurement-audit scripts: memory guard, hashing, input binding with the specification hash, bootstrap percentile interval and DuckDB connection. | support module for the measurement-audit scripts (Table 2; Supplement S2a-2, S3, S3a) |
| `effekte.py` | translated | Compares the mean and median marker difference between endpoint-positive and endpoint-negative persons with the repeat-variation coefficient of the same markers in several pair populations. | not traceable to a reported item |
| `frank.py` | cleaned original | Computes the angle between QRS and T area or peak vectors from the Kors and inverse-Dower reconstructions and from the measured Frank leads, with reconstruction-minus-Frank differences by header group. | Section 3.5 (Frank comparison); Supplement S2a (full angle table) |
| `frank_atlas.py` | translated | Draws image atlases of the largest deviations between reconstructed and measured Frank angles, with the delineation boundaries and the loops in three planes. | not traceable to a reported item |
| `inventar.py` | cleaned original | Reads the schemas and row counts of the index, label, replicate-pair and feature-cache parquet files used by the measurement audit and checks free memory before any run. | support module for the measurement-definition audit data preparation (Supplement S3a) |
| `p_ergebnis.py` | translated | Merges the technical image decisions of the P-boundary audit with the audit sample and computes category shares, population-weighted confirmed-error bounds, boundary-specific bounds and single-lead P durations. | Supplement S3a-3 (historical label reconciliation); Supplement S10a (Historical and later boundary-screen reconciliation) |
| `ptb_drehsinn.py` | translated | Classifies the QRS and T loop rotation of PTB healthy controls (Kors reconstruction) in three planes at several minor-area thresholds and by majority rule, with bootstrap intervals. | not traceable to a reported item |
| `referenz_auswerten.py` | translated | Computes frontal, transverse and right-sagittal QRS and T loop classes (clockwise, counterclockwise, minor-area class) of the report-selected reference at several minor-area thresholds with person-bootstrap intervals, overall and for ages 20-59. | Supplement S2a-2 (Table S2a-2); Section 3.5 (sentence on the threshold dependence of the frontal minor-area class); Supplement S0a (Section 3.5) |
| `signale.py` | original, unchanged | Recomputes for each reference record the representative-beat P boundaries (recomputed P duration), the origin-corrected loop rotation measures and, for the audit sample, single-lead P boundaries. | Section 2.2 and Section 3.4 (P availability and recomputed P duration); Supplement S3a (cached versus recomputed P duration); Supplement S2a-2 (loop inputs) |
| `statistik.py` | original, unchanged | Library of the moment-based ICC(A,1), bias, difference SD and repeat-variation estimators with person-level bootstrap, and of the pair-subgroup frame (rhythm, QRS class, pacing, block, report-defined stability proxy). | support module for the repeat-variation analyses (Table 2; Supplement S3, S3c) |
| `transformation.py` | translated | Compares Kors and inverse-Dower values of the twin markers on the historical 60 training partitions, overall and by rhythm, QRS class, pacing, block, cart and year, and audits the result against the stored twin aggregate. | not traceable to a reported item |
| `untergruppen_klein.py` | translated | Writes descriptive repeat-variation and Kors/Dower rows for subgroups with fewer than 30 persons, without ICC, intervals or decisions. | not traceable to a reported item |
| `vorbereiten.py` | cleaned original | Selects the development reference of one earliest report-selected normal-sinus ECG per person, projects the cached marker columns and assigns the P-duration strata for the visual audit. | Supplement S3a (Table T4c, S3a-1; 1,500-person reference and P-audit strata); Section 3.4 |
| `wiederholbarkeit.py` | original, unchanged | Original block-wise computation of ICC(A,1), bias, repeat variation and bootstrap intervals for all catalog markers in all pair populations and strata, with the catalog summary. | Table 2 and Figures 1 and 2 (historical generator; see Supplement S10a); Supplement S3 |

### `recomputation_wave1/`

First recomputation wave of the revision: independent recalculation of the correction, route difference between production and source measurements, precision of the QTcF band limit.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `a_r1_gegenrechnung.py` | cleaned original | Recomputes the Deming-regression correction independently from the person table and the stored paired differences with newly written code and compares all quantities with the first computation. | Supplement S12a-1 (Correction of the T2 (i) estimate) |
| `a_r1_mc_check.py` | original, unchanged | Estimates the Monte Carlo spread of the bootstrap limits of the R-peak-time on wall-thickness regression (correlation, ordinary and reliability-corrected least squares) by repeating the bootstrap with five further seeds. | Supplement S12a-1 (correction of the T2 (i) estimate) |
| `a_r2_routendifferenz.py` | cleaned original | Computes the participant-level difference between the production-route and source-annotation drug contrasts at 2 h with bootstrap and paired-t intervals and multiplicity adjustment. | Table 6 (difference D); Section 3.6; Supplement S6b-4 (Table S6b-5) |
| `a_r3_praezision.py` | cleaned original | Repeats the PTB-XL band classification as ten independent runs of 2,000 bootstrap draws and derives the Monte Carlo error of the pooled limits and the lognormal parametrization of the reference repeatability coefficient. | Supplement S14c (Monte Carlo error at the QTcF band edge), Tables S14-2 and S14-3; Table 3 (QTcF class at the band edge) |

### `recomputation_wave2/`

Second recomputation wave: selection coupling of the stability proxy, periodic and categorical markers, replication and composition, PTB-XL on identical pairs.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `a_r4.py` | original, unchanged | Recomputes catalog repeatability for the stability proxy without the QRS-class condition, with equal person weights and with all pairs of the same persons, and documents the quality gate of the pair set. | Supplement S3g (Table S3g-1); Section 3.2 (proxy sensitivity) |
| `a_r5.py` | cleaned original | Computes repeatability and matrix classes of periodic markers with wrapped differences, kappa and prevalence-adjusted agreement of the binary flag markers, and a range screen of degree-valued markers. | Supplement S2i (Flags paragraph, binary markers); the wrapped-difference part was withdrawn and replaced by the shortest-distance analysis of S2i |
| `a_r5c.py` | cleaned original | Recomposes the catalog median ICC values with wrapped-difference ICC values in place of the raw values for the non-constant periodic markers. | not traceable to a reported item |
| `a_r6.py` | cleaned original | Decomposes the difference between test and development median ICC into between-person and error variance from stored aggregates and compares the case mix of test and development pairs, including standardization of the development pairs to the test mix. | Section 3.7 (test-partition replication); Supplement S13e (Table S13e-1) |
| `a_r7.py` | original, unchanged | Compares LoopLens and BRAVEHEART repeat variation on identical marker-wise pair sets of PTB-XL, with a paired ratio of repeatability coefficients and two pair-set variants. | Supplement S14d (Table S14-4) |
| `h244.py` | cleaned original | Shared helpers for the post hoc recomputations: memory guard, script hash stamp check, paths and output writing. | support module for the post hoc recomputations behind Supplement S3g, S2i, S13e and S14d |
| `postproc244.py` | translated | Redacts public aggregate files after the small-cell check (drops or rounds columns from which counts below ten could be back-calculated) and keeps the unredacted copies in a private location. | support module for Supplement S2i (flag and range-screen aggregates) and S13e (Table S13e-1) |

### `repeatability/`

The repeatability implementation and its aggregations (repeat variation, class shares, matrix classes).

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `aw149_auxiliary.py` | translated | Computes the 95th percentile of the absolute Frank-versus-reconstruction angle deviation in healthy controls with bootstrap intervals, reconciles the historical P-boundary label counts, and reconstructs the repeat-pair flow and the age and sex of persons with and without repeats. | Section 3.5 (P95 values of the healthy controls); Supplement S2a (updated P95 intervals); Supplement S3a-3 and S10a (label reconciliation); Supplement S5a (Tables S5a-1 and S5a-2) |
| `aw149_clusters.py` | translated | Computes all pairwise exact Spearman correlations of the catalog markers in the fixed training sample and derives redundancy components with one representative each. | Section 2.2 (components of the catalog); Supplement S3d (Table S3d-2, empirical redundancy) |
| `aw149_common.py` | translated | Isolated resource guard, hashing and input-binding helpers of the independent re-aggregation scripts. | support module for the independent re-aggregation of the repeat pairs (Supplement S3c-S3f, S5a) |
| `aw149_repeat.py` | translated | Independently re-aggregates the stored replicate pairs (ICC(A,1), bias, repeat variation, bootstrap intervals per marker) and adds variation strata, heart-rate change, family distributions, sex and age strata, bounded-angle limits and the equal-time rule calibration. | Table 1 (separately written aggregation); Table 2 (reproduction check); Section 3.2 (equal-time and report-normal area-angle repeat variation); Supplement S3c, S3d, S3e, S3f, S10a; Supplementary Data S2 (reliability file) |
| `class_proportions.py` | original, unchanged | Recomputes the class shares of the matrix classification using the estimable denominator of each bootstrap draw. | Supplement S2f (Table T3c: shares of draw-estimable markers and bootstrap denominators) |
| `common.py` | translated | Shared paths, seeds, memory guard, hashing and bootstrap helpers of the repeatability and matrix-class scripts. | support module for the repeatability and matrix-class scripts (Table 2, Table 4; Supplement S2f, S3) |
| `klinisch_p.py` | cleaned original | Computes the share of ECGs whose spatial QRS-T angle crosses fixed thresholds when inverse Dower replaces Kors in a historical clinical training subsample, and the P-duration availability and distribution of the 1,500-person reference by cache availability. | Supplement S2f (T3d fixed angle-threshold stress-tests, historical training rows); Supplement S3a (Table T4c and P-duration table) |
| `kontraste.py` | translated | Computes contrasts between pair populations (stability proxy minus all pairs, with persons held fixed) on the markers estimable in both populations within each common bootstrap draw. | Supplement S3 (T2b contrast table); Section 3.2 and Supplement S3g (proxy contrasts, reproduced by the post hoc sensitivity) |
| `sign_availability.py` | translated | Counts candidate, finite and unavailable projections for the sign sensitivity of the inverse-Dower Z/V4 coefficient. | Supplement S2f (Z/ sign sensitivity table) |
| `transformation.py` | original, unchanged | Classifies every catalog marker into the technical matrix classes per fold with person-bootstrap class counts, pooled-SD normalization, a fixed grid of rule variants and fold sensitivity. | Table 4 (historical class counts); Figure S1; Supplement S2f (Table T3c, rule grid, ten highest minimum-fold correlations) |
| `vorzeichen.py` | cleaned original | Reprocesses PTB records with the inverse-Dower Z/V4 coefficient sign changed and with a boundary fusion without VCG input, and computes segment-integral angle differences, net-area turn classes and minor-area classes. | Supplement S2e-2 and S2f (Z/ sign sensitivity; optional no-Kors-boundary-fusion comparison); Section 3.5 |
| `wiederholbarkeit.py` | original, unchanged | Computes ICC(A,1), bias, repeat variation and person-bootstrap intervals for all catalog markers from the replicate-pair blocks for all pairs, the stability proxy and sensitivity populations. | Table 2; Figure 1; Figure 2; Table 3 (estimators reused for the PTB-XL pairs); Supplement S3, S14 |

### `supplementary_checks/`

Checks added in revision (device measurements, failure reasons).

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `a_nr5_geraetemessungen.py` | cleaned original | Computes repeat variation (ICC, RC, bias; angular distance for axes) of the device-reported measurements on the same development pairs and its ratio to the LoopLens values on identical pairs. | Supplement S3h (Tables S3h-1 and S3h-2); Section 3.2 (machine-reported measurements); Key findings |
| `a_s2g4_ausfallgruende.py` | cleaned original | Aggregates from the public marker file why 31 catalog outputs are not estimable in the full comparison and 31 outputs of the historical set in the boundary-shift experiment. | Supplement S2g-4 |

### `test_partition_replication/`

Replication on the test partition under the fixed protocol.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `aw159_inputs.py` | cleaned original | Binds the inputs of the fixed test-partition replication without computing replication statistics: source and function hashes, development extraction equivalence on 200 ECGs, pair and P selections, privacy audit and manifest. | Supplement S13a (freeze, exclusions and source binding) and S13b (selection); Section 3.7 |
| `aw163_extract.py` | translated | Binds the frozen inputs and extracts marker values and release statuses for the selected test-partition pairs and matrix halves, with a logged access ledger and without any replication statistic. | Section 2.1 and Section 3.7; Supplement S13a (freeze, exclusions, source binding); Supplement S13b |
| `aw163_extract_v2.py` | translated | Continues the test-partition extraction after a technical stop by canonicalizing a stored option list and adds the fixed-coefficient sign sensitivity; parameters and selections stay unchanged. | Supplement S13d (options-comparison stop paragraph); Supplement S13a |
| `aw163_finalize.py` | translated | Validates the extraction checkpoints and writes the column-wise test input files for the repeat and matrix analyses without evaluating any criterion. | support module for Section 3.7 and Supplement S13a-S13b |
| `aw163_preflight.py` | cleaned original | Checks the hashes of the fixed specifications and sources and writes the development-only benchmarks, marker universes (two technical exclusions) and a status file before any test value is opened. | Supplement S13a (freeze, exclusions and source binding); S13 (criteria and benchmarks); Section 3.7 |
| `aw163_run.py` | translated | Single-run replication runner: freezes the statistics, then computes the repeat, matrix and P-availability criteria with shared person bootstraps and writes immutable checkpoints and summary files. | Abstract; Table 1; Section 3.7; Supplement S13b (criterion decisions); Supplement S13c (original-denominator sensitivity) |
| `aw163_statistics.py` | translated | Pure functions of the replication statistics: percentile intervals, shared person-resample weights, weighted rank and quintile metrics, class combination, agreement and kappa, and self-tests. | support module for Section 3.7 and Supplement S13b |
| `aw163_verify.py` | translated | Final check of the replication deliverables: output hashes, completeness of the marker universes and absence of real identifiers in the project files. | support module for Supplement S13 (delivery verification of the replication outputs) |
| `torlogik.py` | translated | Pure functions that encode the pre-specified criteria of the test-partition replication (interval inside band, repeat gate, matrix gate, P-availability gate, duration gate). | Section 3.7; Supplement S13 (criterion table); Supplement S13b |

## Checksums and cleaning

SHA-256 of every file as it was when it ran (original) and as published. Unchanged originals have one hash. For cleaned originals and translated files the cleaning column lists the edit types with the number of places; the edits were verified against the original by a token-level comparison, which allows only these edits.

| Script | Status | SHA-256 of the original | SHA-256 of the published file | Cleaning applied |
|---|---|---|---|---|
| `circular_marker_check/a_r5k.py` | cleaned original | `005bc65b3c41cecb106708efd51a2bcc128a479a26ee2cc82d6ad8d984cbaa3e` | `a18ca816322ad172c7eaf7cf03091bfc9e10253430d24556107f05b61b8f7b77` | local paths replaced by data_path (1); user or cloud-folder names removed (1) |
| `delineator_input_check/__init__.py` | original, unchanged | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | same file | - |
| `delineator_input_check/auswerten.py` | translated | `9136341927d5e0b5a33a546b036a0df150f5e160e1f7aab675d747f267924780` | `ccf191a9a14244ffa997bb7ff51863998a7844d22414a292635c055cf05fb398` | comments, docstrings and messages translated; local paths replaced by data_path (2) |
| `delineator_input_check/kern.py` | translated | `c942440c2d3db7f066d19ddfbadb2641e2b0babc37758f7db3b5c54af673b7f4` | `a974c3f467397e263c11f5044c45860ba288213b223350dd78e4bca85c220a0f` | comments, docstrings and messages translated; local paths replaced by data_path (2) |
| `delineator_input_check/ludb_repro.py` | translated | `74184595c02b73f6ecfb77b9435ff645b16acd3970d38530a632f9d23babdea0` | `eeffefd302eb9aaf8536b3fffc3c61603b5ffad18de7bdfe1a8e9c6a3a32dd0b` | comments, docstrings and messages translated; local paths replaced by data_path (1) |
| `delineator_input_check/ludb_run.py` | translated | `4c7ad556b2fc0c01701f0241149bd03359cd2626fa7fb0abb48eea94900f66db` | `66cafa16427c79146300ade4c6aecbef6316b6245fb0af727df1b2ac7258924a` | comments, docstrings and messages translated; local paths replaced by data_path (1) |
| `delineator_input_check/mimic_run.py` | translated | `5219e2166f02b1992dc7bf23b0b6bc15de6086e9ab7f5924e7f79fac0091a325` | `e5e104e09eef4993d18b96c3d19907536d20063ca2e6e3dac38125a003f8eaf4` | comments, docstrings and messages translated |
| `delineator_input_check/ptb_run.py` | translated | `e631d38113a073a9d6788e0f4322c4f2fbf48cefc5f2c0f432bfc1721c700d42` | `4d15227977925517786252b39a9d3339de0df50c2b082f4027c7dd445db2c134` | comments, docstrings and messages translated; local paths replaced by data_path (1) |
| `delineator_input_check/tabellen.py` | translated | `a7c4104ef49f2c965d2aa1bf47abfc6c811cf35aa6e32b7280199d7aee2ccc97` | `2ed2455f1f4f7025182bd1fbc4a1e3c09d0ffab6cc8ef1eaf09128c8a7776796` | comments, docstrings and messages translated; local paths replaced by data_path (1) |
| `error_variance_stages/a_r1_stufe0_rahmen.py` | cleaned original | `7bb41bd2206e02a74c0e96f795f947b667de33fb33310ccd0f25c15be63085be` | `c87dda6592b214a3d9d013459a52c113fb9be506602179cb6b883669494aaf7d` | local paths replaced by data_path (2); user or cloud-folder names removed (1) |
| `error_variance_stages/a_r1_stufe1_fehlervarianz.py` | cleaned original | `ef04b17cefb5938506b23cfaf0e46a6a996c3553aa6a7179648d923da8792c02` | `c8165bf693d2d2fadf3b187bab9a061d036237600a0ee7bd487bb4e3e9a0be29` | local paths replaced by data_path (2) |
| `error_variance_stages/a_r1_stufe2_steigungen.py` | cleaned original | `cbe1b532e1144e1a64030d85d4ec6bab1bf06d44a752bc2f6f9f79a48ebc6734` | `e722816a4556dc8bf6d7f7e9af4df9003aa7b2805c3a466b5f5714d597b49a22` | local paths replaced by data_path (1) |
| `figures/abbildungen_a_final_v27.py` | translated | `50af45d55c08a7d7f78bc681f5c07ad638e6d5de5a84586a50f2f50fe2dd940e` | `e3284e6ac3fbab430cc9aec282fc608eb172be8cf9adbab611bf2b88746c0d9d` | comments, docstrings and messages translated; local paths replaced by data_path (1); user or cloud-folder names removed (4); house-style module replaced by a plain matplotlib style |
| `matrix_variants/aw150_analysis.py` | translated | `ee5fa067726f95ac52bf59e984f46aae1c0ec1b10e1f18055518fada26659ac8` | `bf227827d79a7419e4f914a3298bbdc886d534ab5689b34a2793733eafbeff5f` | comments, docstrings and messages translated; local paths replaced by data_path (3) |
| `matrix_variants/aw150_report.py` | cleaned original | `366f4ca90946c0b47101eeb590578a5a8dfcb48136b42aa7e9ed6ca3e4466126` | `6dbbec1077598a4ff394a81b9d0b8227a4fe3aacd3b709ea2dc2921117aab1e2` | long dashes replaced (12); unpublished numbers removed from texts |
| `matrix_variants/lauf.py` | cleaned original | `c7918bc4d4e20f505e8781911c77b814659bacb6c50f233ee38e774fbb2e9631` | `0be7de870effdf201c670e4440c258ad2e411b22058300dc979ef87a70eed10b` | local paths replaced by data_path (1) |
| `measurement_audit/common.py` | translated | `e8a9cbef930ee1c9bd5e9bf3c39a3b77667fba16cafde48c6696a6f34c020772` | `02cf053356f88f8bb8c9bdaadab2186d189e355bd6c70e4b9560ea7324dc281c` | comments, docstrings and messages translated; local paths replaced by data_path (2) |
| `measurement_audit/effekte.py` | translated | `be118eeacd87dc8bb970cd4f811904615cfa1e91e33f61e7a975487df410142a` | `c0a5e1dfdf63b7b63029d39baa79c2e1bb44165cced7208005f5b554f34138a1` | comments, docstrings and messages translated; local paths replaced by data_path (1) |
| `measurement_audit/frank.py` | cleaned original | `a0749260d665825a4c9b2da8f8ec8e243524d6b10ed7a75fbe6c7e62089bffb8` | `d1961365640f98e79cad2244f60c66ffb8bc5e8605b38e7d14024234bd10ccfb` | local paths replaced by data_path (2) |
| `measurement_audit/frank_atlas.py` | translated | `4228c9eb66b313b600244130532ae11733ad4a219ed7c5665340019a2a04e06f` | `0abce136a33700896d7eed57b0e2ea00236b51744281a43b29912da2f384fffc` | comments, docstrings and messages translated; local paths replaced by data_path (1) |
| `measurement_audit/inventar.py` | cleaned original | `27d3bafeeca6bc979bc2b26fe7c5daf0a64c8b93e491a3edbcd3be0467fded9f` | `e65d332c80cc470ce7a774c75193c7ab33e59360d282ff7678d6df971c2261c6` | local paths replaced by data_path (4) |
| `measurement_audit/p_ergebnis.py` | translated | `86b1b42df42db53219287459048bc7d94a60e3c2080d1c3c0cdd20188143baba` | `dd22642e4e0e32d048f66efaa45313cef6b80bb70d929620e6a2de69359dfdef` | comments, docstrings and messages translated |
| `measurement_audit/ptb_drehsinn.py` | translated | `e090634e1d40f9fe7e22efc5a120a7de42ddbf1055cab435fdc286b2f8c6046b` | `598f56da3f0abe0d4a4f9551bc42d87ca34576f8e5addf0e40f4db14d1d281a7` | comments, docstrings and messages translated; local paths replaced by data_path (1) |
| `measurement_audit/referenz_auswerten.py` | translated | `31690bc7f719568fceb64d6978385fcbb7e7c45b889f7203cc188d92130f9db4` | `1cd653c017af93b38dff0296053bdf7916ffb6f8982caa4171d15e4254aa73e8` | comments, docstrings and messages translated |
| `measurement_audit/signale.py` | original, unchanged | `7f880263c8ee4f795dc32a67fed3a3539fb4abebb656ba8c6e97fff7f4eafc29` | same file | - |
| `measurement_audit/statistik.py` | original, unchanged | `30d9f703975da5bdffb19fbe185319c80d318a19d7419ab18cdf72486967f574` | same file | - |
| `measurement_audit/transformation.py` | translated | `0bb166c6e2bc22fb0517d265f1b6829992d8d7e7f1f938ceee44770db3f80b29` | `7b87ad1f5577062f0ad5254147b2c77a57e433a6a607525f49636ce48ebd3e76` | comments, docstrings and messages translated; local paths replaced by data_path (1) |
| `measurement_audit/untergruppen_klein.py` | translated | `2b657b12900955a69473a0e512eeac24c84310de00acdd697402bd2a2691af59` | `708ec798f6f063bd5fc3748a7a3616a4d8f6fd124ea96ba38f40881cba6b65c1` | comments, docstrings and messages translated |
| `measurement_audit/vorbereiten.py` | cleaned original | `a0b21bfa5cb123e2dee928a7d6fdec4da3f5d8db25712c3b489414bf7335f31c` | `8865746a51e7a47d7c06e267902c7d286205c96067dd7798e2a9bdb7b3d2f7c8` | local paths replaced by data_path (2) |
| `measurement_audit/wiederholbarkeit.py` | original, unchanged | `f3351b070b869013dcbcb268e017efb4df71e0c841672b15d4d13668585be39e` | same file | - |
| `recomputation_wave1/a_r1_gegenrechnung.py` | cleaned original | `ebfb4ab5a338abb4738c4dfef48bc78d0f32673687141a371368a87f9edd9646` | `4a9f28ddce805f110f0584ba0fb291b28cc743d62e448cd73c96b93e0f582736` | local paths replaced by data_path (2) |
| `recomputation_wave1/a_r1_mc_check.py` | original, unchanged | `7e9273271aa70a833d8eff913d3faed42e716c634e7d656f2b6e6d5fc958f559` | same file | - |
| `recomputation_wave1/a_r2_routendifferenz.py` | cleaned original | `fb29ee254b64353186fc563a429f759c812d0c914cd8ed32e5e7f00c66b3fccb` | `2881c19ca1bf0f40874a061c8099957940c48f084ed4440608b5e4719b0b91d2` | local paths replaced by data_path (1) |
| `recomputation_wave1/a_r3_praezision.py` | cleaned original | `58a1ddb2cbf0322e2d1e730566f85dae8e1b93c11a352326d585162e5b731cba` | `67b81e613963992f7c0de713c47a5950088881c3f3c247c92ae97cff52896046` | local paths replaced by data_path (3); user or cloud-folder names removed (2) |
| `recomputation_wave2/a_r4.py` | original, unchanged | `7692118054c7b39dc579a3a6d7084f2e906b17a8e68480103a9e80bde3d73252` | same file | - |
| `recomputation_wave2/a_r5.py` | cleaned original | `4f1ce589752fc4ce6be4fd9ad45e3387f0e149153a25467e3a1ec916866443e9` | `665f1cca5a85bb4753815cb25d14893e6736fad487ba0ce397db9074380b5fca` | local paths replaced by data_path (2) |
| `recomputation_wave2/a_r5c.py` | cleaned original | `325cb2dd38dc39a9a4ba58e092f825f5bd9c97ae54a51f326d6752ef7dfac6d3` | `a50e53a0885653b91ea88480d7050233bbf89e51499d90dd0f5dc4a3a9815945` | local paths replaced by data_path (1) |
| `recomputation_wave2/a_r6.py` | cleaned original | `85b679bf1deadd1b8f65fad2506c26359da555b1f28cb7c3914480ff973d91ce` | `fff4969a76c3a14c21b1698b44fea32918c4dc4846307994f3c337e259923918` | local paths replaced by data_path (2) |
| `recomputation_wave2/a_r7.py` | original, unchanged | `cd2e77c5b4873b83a5c7e21209f8e6f950355b3b00e4a35ebb3d0bfe38a7973f` | same file | - |
| `recomputation_wave2/h244.py` | cleaned original | `f1dac48c19cfd2023356244edfafbab6c647bce95cda19ec265eae4e37a98808` | `1e87bce99e1ac4372da08c6757146a14cbe03e3a4119947285130e8d9774a8ed` | local paths replaced by data_path (4); user or cloud-folder names removed (3) |
| `recomputation_wave2/postproc244.py` | translated | `b0e72f4932dd033061f7185866b04db4d1cd75c8ad902ba973d9fd53a7d3d94e` | `4fe14739382edd4e46cbfd2bc2114d6e7c34a83427ccf4684426baa6b4806ac6` | comments, docstrings and messages translated; local paths replaced by data_path (2); user or cloud-folder names removed (2) |
| `repeatability/aw149_auxiliary.py` | translated | `f91e2702d8c728468212fb00dec61ca36cc22b2f6db1246818431aa3c802d21d` | `7dacf871cda2b58d2a0cdb127cef6d7a9a6a070ce2014ff96ab0a70425a3cb4e` | comments, docstrings and messages translated; local paths replaced by data_path (2); long dashes replaced (1); unpublished values moved to the local reference file (2) |
| `repeatability/aw149_clusters.py` | translated | `499288b6f0d5ebc117105790a4147ca7812fcf6dea3ddf50bf1c9dce8fad8d06` | `2ed0c51b84624f52bcc811d815d7b462517d5237289e52c199dc977682cbefa3` | comments, docstrings and messages translated |
| `repeatability/aw149_common.py` | translated | `7f7e0de083cbbef43580af5ad70b5b77a3392b00088b69772a20d2e64ef57123` | `8f6fead814a21d46a253c21cdc0b8af49e917dd583b7d19f13f16ba2378ba216` | comments, docstrings and messages translated; local paths replaced by data_path (3) |
| `repeatability/aw149_repeat.py` | translated | `6d14e3fb524976104c3ca287fea6ec6c95e18fb78c1f8f543e690d7af7c2a4e4` | `ff5df104fff65d15b6837c7839f6a927b2d24e5d70f6132ff65b9c02642c5070` | comments, docstrings and messages translated |
| `repeatability/class_proportions.py` | original, unchanged | `7cda9f5a1f096102fa97d65efe95ade3e359404d98a42a87825f6460cde7fc50` | same file | - |
| `repeatability/common.py` | translated | `d962665c69a5ac3f731cf5302b55b9cbce60418fb1ecbdf11eaeadc2ff3b0006` | `c86c27f4bba9379046da8a628655e07be5694ccace93ff045b4bfc4f8bdd77ea` | comments, docstrings and messages translated; local paths replaced by data_path (4) |
| `repeatability/klinisch_p.py` | cleaned original | `f20198c55be1fec5b55fc9f366b3e6f7c2a925ca40e1f03cc9fdf2313f14f583` | `276064938a7d6f613605b7b088f80faf79cc27325c5aa493c84e275e21fb541e` | local paths replaced by data_path (1) |
| `repeatability/kontraste.py` | translated | `ab028a19de71354659b69905e0ebf932330331ad271bbf34ff8275b7386aae5b` | `e397da380419d63ee5738ceb5f707528ea170b9a9b16fdc1e83198769e5e28e8` | comments, docstrings and messages translated |
| `repeatability/sign_availability.py` | translated | `43988a479f0b28f1a78043a51b76d8cdf0bf2cc6ad14eb82ad1da694495c47f1` | `b6836f1ecb83c3e3b2e20d3d0b8aff74b340aee72742604b7692328935b7ecdc` | comments, docstrings and messages translated |
| `repeatability/transformation.py` | original, unchanged | `1a38d367121ba011e5973843f2da16012dfe27f764a0e4c7cceac69c0e2ec8fa` | same file | - |
| `repeatability/vorzeichen.py` | cleaned original | `0f3d5ea5a0485c751e44bd775aa90800a884e7490517cbbd821f4fe59a0e321a` | `b6cae1878239ed84d6ddad5630ae9176c22feee133748ede502f8cc329a095ba` | local paths replaced by data_path (5) |
| `repeatability/wiederholbarkeit.py` | original, unchanged | `77485bcdfdcb926e908a48b59f0b00f5ae308275ed6733260c138392e9c7bf49` | same file | - |
| `supplementary_checks/a_nr5_geraetemessungen.py` | cleaned original | `78c466d84f2cb29ff3a77399e393b910adcb826ea49f574868904027c8505825` | `8380987c50f90a74cba984d0f8d60a76495a0edd5f9ef60ef2e746e0b3ed2795` | local paths replaced by data_path (1); user or cloud-folder names removed (1) |
| `supplementary_checks/a_s2g4_ausfallgruende.py` | cleaned original | `827c1cb93877e877d1cdba55872e9fe84f0aee76e06c7b89e7891a44c7fea3c1` | `675e4c30a658e7bde815e21a1fe1698576d77fc48ee57390ea75f6e794487a76` | local paths replaced by data_path (1); user or cloud-folder names removed (2) |
| `test_partition_replication/aw159_inputs.py` | cleaned original | `5aaee52ce10781361a1ddf8cdb9e841bf39a4160d0e381732c08034a80f08119` | `087c3c4e08b1a5815c99603650c4aa9293f340f262d9227354d95bb3a1144b30` | local paths replaced by data_path (9); user or cloud-folder names removed (1); long dashes replaced (4); unpublished numbers removed from texts |
| `test_partition_replication/aw163_extract.py` | translated | `992929500b9b5942dc01682c60b92a97f1b8df7f29fa8174aa64a51f16a6c98f` | `7681780a8e79e2e5533d81b601ee8bb46ba8ccd6d0ad9233c7bec5a7409dd94c` | comments, docstrings and messages translated; local paths replaced by data_path (2) |
| `test_partition_replication/aw163_extract_v2.py` | translated | `3f8acc795b4ebb4b98c2f6808dbc93e0cfade7a21751f98b1cc689b6057162bf` | `bfc4bd9963e9075eac4818b008a8cdc68a726d537dce162a7bb05b5c4d6b60ef` | comments, docstrings and messages translated |
| `test_partition_replication/aw163_finalize.py` | translated | `a1d56377e7d0fbe92887aa3e5c0f697ac34741bb19d75a586a74692cb5e0848a` | `f0d2a6df7257d4c10a80b6f5f807ba9badb9d66434e2a4052635fcac3d06067a` | comments, docstrings and messages translated |
| `test_partition_replication/aw163_preflight.py` | cleaned original | `e6a3659131acbe4ad98674e922ec42c2ac90d22763a2c32f0e6c7eee6cbf8f9e` | `3a1df478ede73c8c29232068438f684d43431c1c44b0b676bdb9887e238ebb82` | local paths replaced by data_path (5) |
| `test_partition_replication/aw163_run.py` | translated | `087b9b6fd7dd1a14f2c1cd90536338f27916a5e8f4d9d4f7546376ec9ff4b76d` | `8caabfd3a4b6b5e55c234395d43085e0ab95c1bcbe90436aa77f3b0326bb3a0d` | comments, docstrings and messages translated; local paths replaced by data_path (3); unpublished values moved to the local reference file (1) |
| `test_partition_replication/aw163_statistics.py` | translated | `e9967960b7a04778d332c9b2036acdf2a0a47627e1829f1800009baecf4e580f` | `d4c5679b3a1593870236cec3911e04f3a683b10d82adaf1f41bde9938b24db89` | comments, docstrings and messages translated |
| `test_partition_replication/aw163_verify.py` | translated | `583f0b8df5e168923e65e62f4c4200ef5fbc2dd780553f5829803e5eb678130d` | `c1c58e123f7f565913d34e47902a7ccc0a3caf61bec4ec1df110b81b0451526a` | comments, docstrings and messages translated |
| `test_partition_replication/torlogik.py` | translated | `d395b5c15234dcb68d33c759937e2edf00d74f3a64542298fc23fda3ac6cd365` | `0bf253d39d9d112b2e54fb671abe9018f412801484e06c63a7914a6260e1c926` | comments, docstrings and messages translated |

## What is not in this folder

* Self-tests and unit tests of the analysis code, download helpers and packaging of the manuscripts (not part of the reported analyses).
* All data, result tables, intermediate files and the local reference-value file.
* The unpublished expected values of 2 scripts: they were moved to a local file that is not published (see the root README, section Running the scripts).
* Modules shared with other works are in `../shared_support/`.

## Reproducing the numbers

The numbers are in the manuscript tables and the supplement. To reproduce an analysis, obtain the data, produce the LoopLens output tables with the snapshot, fill `config.toml` and adapt the scripts to your layout. Random seeds are set in the scripts and marked as such.

