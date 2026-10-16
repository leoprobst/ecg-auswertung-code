# s5_time_alignment

Working title of the manuscript: Heart-rate-based clock alignment in intensive-care ECGs, serial-ECG matching and administrative follow-up in MIMIC-IV

Scripts behind the alignment of bedside heart-rate records and ECG-derived rates, the simulation of known offsets, the serial-ECG matching and the follow-up counts.

## Data and inputs

MIMIC-IV-ECG 1.0 and MIMIC-IV 3.1 (icu and hosp modules); MC-MED 1.0.1 for the waveform-based validation. Heart rates of the ECGs are LoopLens output. None of these data is part of this repository. Set the locations in `config.toml` (see the root `README.md`); the keys used by this folder start with `s5.` in `config.example.toml`.

## Scripts

44 files: 5 original and unchanged, 36 cleaned originals, 3 translated. The status column uses the three terms explained in the root README; cleaned originals and translated files received some of five edit types: paths replaced by data_path, user or cloud-folder names removed, long dashes replaced, unpublished expected values moved to the local reference file, unpublished numbers removed from texts. The last column names the part of the manuscript or supplement whose numbers depend on the script, as far as it could be traced (`support module` = used by other scripts; `not traceable` = no reported item found). Folder names are descriptive; the scripts themselves keep their original file names because they import each other by name.


### `earlier_analyses/`

Earlier alignment analyses whose numbers carried over into the supplement.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `aw178_audit.py` | translated | Checks the sealed inputs by hash, stops the historical comparison at its pre-specified gates, and reproduces the development-only census of the stored AF landmark selector together with the age and sex description of the selected landmarks. | Table S4.1; Supplement S3.2 (reproduction of the sealed landmark selection) |
| `aw178_deliver.py` | cleaned original | Assembles the aggregate outputs of the first rerun of the atrial-fibrillation observation count (stage counts, age and sex of the selected index discharges, claim table) and updates an earlier manuscript draft with them. | Supplement S4 (Table S24, selected index discharges); Supplement S3.4 (Table S21, integer-hour columns); the other outputs of this script are superseded by the exact-timestamp recomputation (Table 2, Table S20) |
| `aw181_compute.py` | translated | Computes in four phases the development-only supplementary results: injected-shift recovery, within-person permutation diagnostic, tie and near-optimal-zone variants and strata of the offset estimate; the serial-pair matching bounds under varied calipers; the integer-hour AF censuses reproduced against the sealed census; and the admission-boundary plausibility check. | Table S1.4; Table S1.5; Table S1.6; Table S1.7; Supplement S1.5 (chart-minute counts); Supplement S1.6 (admission-boundary check); Table 1; Table S2.4 (Supplement S2.4); Table S3.2 (integer-hour columns); Section 3.1 (tie-rule shares); Section 3.3 |
| `aw181_deliver.py` | cleaned original | Writes the complete aggregate population-curve file and the supplement and manuscript tables of the second rerun (injected-shift recovery, permutation diagnostic, tie rules, aggregate strata, matching caliper sensitivities, pre-correction AF census) from stored aggregate results. | Supplementary Data S1 and Figure 1A (population curve CSV); Supplement S1.6 (Tables S5, S6), S1.7 (Tables S7, S8), S1.8 (Tables S9, S10); Supplement S2.5 (Table S19); Section 3.3 (caliper sensitivity sentences); Supplement S3.4 (Table S21, integer-hour columns) |

### `ed_waveform_validation/`

Waveform-based validation in the emergency-department dataset.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `s5_mcmed_auswertung_v17.py` | cleaned original | Locked analysis script of the external emergency-department check: extracts waveform heart rate from lead II in 5-min units, selects the beat-detector candidate, estimates offsets for the technical control (monitor reference) and the primary relation (charted reference), and runs the null controls, shift recovery and sensitivities. | Section 2.5; Section 3.5; Supplement S7.3 (Table S29), S7.4 (Table S30), S7.5 (Tables S31, S32) |
| `s5_mcmed_bericht.py` | cleaned original | Builds the results report of the external check from the masked aggregate outputs of the locked analysis without new calculations. | Section 3.5; Supplement S7.4 (Table S30), S7.5 (Tables S31, S32) |
| `s5_mcmed_diagnose_aw251.py` | cleaned original | Post hoc diagnosis on the training split of why only part of the eligible visits are identifiable: reasons for exclusion, relaxed connectedness rules, location of the minimum, null controls and the reference-density curve. | Section 3.5 (post hoc diagnosis, last sentences); Supplement S7.6 (Table S33) |
| `s5_mcmed_nenner_zeit_v16.py` | cleaned original | Counts upper bounds of the usable visits for the primary relation from time metadata only (longest segment, number of complete units, anchor condition, admissible rotations) without heart-rate values or offsets. | support module for Supplement S7.3 (design rules for the primary relation, denominator status thresholds); no reported number depends on its output |
| `s5_mcmed_pipeline.py` | cleaned original | Driver that runs the steps of the locked specification in order: binding of the specification, extraction and technical check for each detector candidate, freezing, pilot on the validation split, training-split extraction, confirmation and report. | support module for Supplement S7.4 (Table S30) and S7.5 (Tables S31, S32), the steps of the locked analysis |
| `s5_mcmed_wacht.py` | cleaned original | Watchdog that restarts the pipeline driver if it disappears before completion or stop, with a limit on restarts. | support module for the driver of the external check (Supplement S7.4 to S7.5); no reported item |
| `s5_mcmed_zaehlung.py` | cleaned original | Helper and counting module for the external check: reads the visit and numerics tables and the waveform headers (lead II), defines the 5-min units and time conversions, verifies input hashes and counts visits and units under masking of small cells. | support module for the locked analysis script of the external check (Supplement S7.2 data, S7.3 design) |
| `start_wmi.ps1` | cleaned original |  |  |

### `figures/`

Figure script of the manuscript.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `fig1_s5_zeitversatz.py` | translated | Draws the three-panel figure of the population heart-rate error curve, the people-flow funnel and the identifiable-offset shares from the stored aggregate offset results, with consistency checks. | Figure 1 (panels A to C) |

### `landmark_selection/`

Landmark and risk-set selection of the documented-AF analysis, named in the number evidence as the source of the selection order. The two scripts import the helper modules a155_research_low_cost and onedrive_names from the same folder (original import path audit_tools.<module>).

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `a155_research_af_landmark_revision002.py` | cleaned original | Selects the first eligible discharge per person and the latest qualifying sinus ECG before it, without marker, outcome or follow-up information, and reruns the association for that landmark design. | Supplement S3.1 (selection of the landmarks) |
| `a155_research_documented_af.py` | cleaned original | Defines the report rule for sinus rhythm and the ordered gates of the risk set (usable metadata, reported sinus, no prior or index AF, discharge within 7 days, no known AF through discharge, consistent event coverage, positive observed time), then fits the post-discharge documented-AF association. | Supplement S3.1 (selection order, report rule and AF source of the landmark selection) |
| `a155_research_low_cost.py` | cleaned original | Helper library of the post-discharge analyses: atomic JSON writing, file hashing, resumable checkpoints, table outputs and the shared run paths and limits that the documented-AF scripts import. | support module for Supplement S3.1 (imported by the landmark-selection scripts) |
| `onedrive_names.py` | cleaned original | Checks and rewrites file-name components so that copies are safe for cloud-synced folders (reserved names, unsafe characters, length) without touching scientific identifiers. | support module for Supplement S3.1 (imported by a155_research_low_cost.py) |

### `wave0_checks/`

Percentage recomputation of the first correction wave.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `s5_r1_abgleich.py` | cleaned original | Recomputes each percentage stated in the earlier manuscript and supplement from numerator and denominator in the aggregate files and reports any rounding or value discrepancy. | Section 3.1 (checked percentages, one corrected value); Supplement S1.3, S3.3 percentages (verification only) |

### `wave1_addendum/`

Observation time after discharge, circular-shift null, audit of hour-boundary differences.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `g243_audit_agg.py` | cleaned original | Converts the raw counts of the hour-boundary audit into a publishable table with the small-cell rule and coarsening applied. | support module for the hour-boundary audit behind Supplement S3.4 (statement on other analysis code) and Section 2.4 |
| `g243_audit_zaehlung.py` | cleaned original | Counts how many assignments and values change when differences stored as crossed hour or day boundaries are computed exactly in seconds, for the AF selector and the other time-based endpoints. | Section 2.4 (differences now computed in seconds); Supplement S3.4 (closing statement on other analysis code, Table S21 context); Supplement S3.2 (reproduction of the index selection) |
| `g243_guard.py` | cleaned original | Resource guard that stops the process when free memory or the process memory limit is breached. | support module for the wave 1 addendum scripts (exact AF census, circular-shift null, audit); no reported item |
| `g243_kern.py` | original, unchanged | Library of the exact-second observation-span arithmetic for the AF landmarks, with both the integer-hour (archive) and the exact variant, used for the end-of-observation classification. | support module for Table 2 and Supplement S3.2 to S3.4 (Table S3.1, Table S3.2, Table S3.3) |
| `g243_null_agg.py` | cleaned original | Aggregates the circular-shift null distributions (three variants) and compares them with the within-person permutation null, including the mixture reading. | Section 3.2; Supplement S1.10 (Table S12 and mixture reading) |
| `g243_null_lauf.py` | cleaned original | Runs one variant of the circular-shift null (shift of each stay's chart series) with the unchanged identifiability procedure on the people with at least three complete ECGs. | Section 3.2; Supplement S1.10 (Table S12) |
| `g243_null_lib.py` | original, unchanged | Library with the nearest-reference selection on the offset grid and the circular shift of a chart series used by the circular-shift null. | support module for the circular-shift null (Section 3.2; Supplement S1.10, Table S12) |
| `g243_r4_lauf.py` | cleaned original | Recomputes the AF 365-day observation census from the sealed index discharges with exact timestamps (at-risk and complete-case stages, end reasons, ties, span in days) as an independent counterpart to the first computation. | Table 2; Section 3.4; Supplement S3.2 (Flow S6, Table S20); S3.3; S3.4 (Table S21 exact columns) |
| `g243_r4_sqlcheck.py` | cleaned original | Second computation route in plain SQL for the at-risk persons and the end reasons of the exact-timestamp AF census, used as a cross-check of the main computation. | verification of Table 2 and Supplement Table S20 (Section 3.4; Supplement S3.2) |

### `wave1_alignment/`

Kernels, runs and self-tests of the heart-rate clock alignment and of the known-offset simulation.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `agg_publish.py` | cleaned original | Aggregates the private raw outputs of the wave 1 reanalyses into publishable aggregate tables (simulation, null and mixture, predictive value, loss waterfall, conditional yield, group comparisons, AF span, reference variants) with the small-cell rule. | support module for Supplement S1.3, S1.4, S1.9, S1.10, S1.12 and S3.4 to S3.5 (aggregate tables of the first reanalysis wave) |
| `aw242_data.py` | cleaned original | Loads the working database of the alignment audit read-only and computes the per-person alignment quantities over all people. | support module for the wave 1 alignment reanalyses (Supplement S1.3, S1.9, S1.12) |
| `aw242_guard.py` | cleaned original | Resource guard that stops the process when free memory or the process memory limit is breached. | support module for the wave 1 alignment scripts; no reported item |
| `aw242_kern.py` | original, unchanged | Library of the unchanged identifiability procedure (offset grid, nearest-reference matrix, near-optimal zone and criteria, Wilson interval, standardized differences) in vectorized form. | support module for Section 3.1, Section 3.2 and Supplement S1.2a to S1.2b, S1.7 to S1.10 (Tables S1.2, S1.2b, S1.3, S1.10 to S1.15) |
| `e0_extract_raw_hr.py` | cleaned original | Extracts the raw charted heart-rate rows (item 220045) with storage time from the compressed ICU chart-event file for the development people. | Section 2.2 (reference description, storage lag); Supplement S1.12 (reference origin, Table S15 storage-lag variants) |
| `e1_delay_stats.py` | cleaned original | Checks the raw extraction against the standardized reference rows and computes the distribution of the storage lag (storage time minus chart time) and of minute and second of the chart times. | Section 2.2 (storage lag of the reference); Supplement S1.12 (reference origin); Supplement S1.9 (empirical delay scenario) |
| `r0_baseline.py` | cleaned original | Recomputes the baseline per-person alignment results over all people and checks parity with the archived results. | support module for the wave 1 reanalyses (baseline for Supplement S1.3, S1.9, S1.12); parity check for Figure 1 and Table S1 counts |
| `r1_gegenrechnung.py` | cleaned original | Independent recomputation of the alignment counts from raw sources with separately written code, compared person by person with the baseline results. | Section 3.1; Figure 1B and 1C; Supplement Table S1 (independent verification of reported counts) |
| `r2_run.py` | cleaned original | Runs the simulation of known clock offsets on the observed chart sampling, the within-person permutation null (200 replications), the mixture estimate and the predictive value of the flag for large offsets. | Section 3.2; Supplement S1.9 (Table S11); S1.10 (Tables S12, S13) |
| `r2_sim_lib.py` | original, unchanged | Library for the simulation of known true offsets: robust variogram, Ornstein-Uhlenbeck fit and bridge sampling of the latent heart rate, generation of pseudo-ECG rates with delay and noise, and scenario runner. | support module for Supplement S1.7 (Table S1.10), S1.8 (Tables S1.11, S1.12) and Section 3.2 |
| `r3_loss.py` | cleaned original | Computes the exclusive loss decomposition in fixed order, the conditional identifiability yield with Wilson intervals and the standardized differences between included and excluded people. | Figure 2; Section 3.1; Supplement S1.3 (Tables S2, S3); S1.4 (Table S4) |
| `r4_af.py` | cleaned original | Computes the AF administrative-span quantities: causes of zero-time losses, shift between zero and positive span caused by integer-hour rounding, census without rounding, contact sources, tie rule and contact-chain sensitivity. | Supplement S3.1 (contact sources); S3.4 (Tables S21, S22); S3.5 (Table S23); Section 3.4 (integer-hour statement) |
| `r5_ref.py` | cleaned original | Runs the reference sensitivity under the unchanged full-coverage rule (sinus rhythm without pacing text, storage-lag limits, off-hour times, narrower matching windows, combinations) with permutation null rates. | Section 3.2 (last paragraph); Supplement S1.12 (Table S15) |
| `r5b_pairs.py` | cleaned original | Restricts the reference at pair level while keeping the analysis cohort of the baseline and computes the population curve for each restriction. | Supplement S1.12 (Table S16) |

### `wave2_individual_support/`

Individual-support analysis and its small-cell post-processing.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `s5_r6_kleinzellen.py` | original, unchanged | Applies the small-cell rule to the aggregate individual-support table (cell of 1 to 9 becomes <10, complement not shown, merged row for 3 to 5 ECGs). | Table S1.13; Supplement S1.9 (counts by number of complete ECGs); Section 3.2 |
| `s5_r6_lauf.py` | cleaned original | Computes leave-one-ECG-out stability and the within-person permutation p value for each person with at least three complete ECGs and calibrates the expected counts without signal on permuted pseudo data sets. | Abstract and Key findings (strictest-group share); Section 3.2 (individual algorithmic support); Supplement S1.11 (Table S14) |
| `s5_r6_lib.py` | cleaned original | Library for the individual support analysis: leave-one-out offset, exact and sampled permutation p values and the pseudo data set calibration. | support module for Supplement S1.11 (Table S14) and Section 3.2 (individual algorithmic support) |

## Checksums and cleaning

SHA-256 of every file as it was when it ran (original) and as published. Unchanged originals have one hash. For cleaned originals and translated files the cleaning column lists the edit types with the number of places; the edits were verified against the original by a token-level comparison, which allows only these edits.

| Script | Status | SHA-256 of the original | SHA-256 of the published file | Cleaning applied |
|---|---|---|---|---|
| `earlier_analyses/aw178_audit.py` | translated | `66535770f20034d3cbfde53b6eb738a8f8c14b8912bd0840cb10ad23119fb3fc` | `462d80e636308f22c08164cfb29aa5ec91ebe6d287cb4c9b8910d5a763dc66d3` | comments, docstrings and messages translated; local paths replaced by data_path (4); user or cloud-folder names removed (2) |
| `earlier_analyses/aw178_deliver.py` | cleaned original | `74ee1113196e05cb4f121b708b0a3771b690791fd55cd45813fcddb195848aaf` | `8cf76340b9fb22dcb9eaa2576a4558a5cef869f630a7ba066725717b7660d36f` | local paths replaced by data_path (3); user or cloud-folder names removed (2); long dashes replaced (5) |
| `earlier_analyses/aw181_compute.py` | translated | `e56b31d4a89cd78b04db3fdfc8a2191d72c28145da529b9ab729284685fcb553` | `833f3eeb4a5330dc998d4681eb52ba8061977888fe5946085fe17abbbd86e7ac` | comments, docstrings and messages translated; local paths replaced by data_path (13) |
| `earlier_analyses/aw181_deliver.py` | cleaned original | `c5eeb29e7bb508c09bbf27e9d35db0c9bce60ff5ae8e9f39ec3e8b8badf1daf7` | `4eecef14f23dc10dc25ef0d35449f856486abfaa10c7364cfe6f7e77491e6830` | local paths replaced by data_path (5); user or cloud-folder names removed (2); long dashes replaced (51); unpublished values moved to the local reference file (21); unpublished numbers removed from texts |
| `ed_waveform_validation/s5_mcmed_auswertung_v17.py` | cleaned original | `5b2532374fa9388f9592cfca673b989437e60b047bcb9e807893fd874ab0e00e` | `6d7c53d66e8b4aebc0eecbca77af844827bf01e45394d8b4c8ef35bcf14c8650` | local paths replaced by data_path (5); user or cloud-folder names removed (11) |
| `ed_waveform_validation/s5_mcmed_bericht.py` | cleaned original | `4571040e1c2859d03ae6b229daa5be22ea92f2b5afb3651482695855122b0147` | `ad966b5bae80b68941e3bc0c03cc4f415d04a52b186f2ec0438bbe56c730510f` | local paths replaced by data_path (2); user or cloud-folder names removed (2) |
| `ed_waveform_validation/s5_mcmed_diagnose_aw251.py` | cleaned original | `6269ac0e712b1310a7ab56177918ac93322ca6a831bbdb3f7944836b1ce22a03` | `26074c057f748f3ce38aded67a35051a4bd2d9956a1af8138cd6a98ce4c5a379` | local paths replaced by data_path (2); user or cloud-folder names removed (3) |
| `ed_waveform_validation/s5_mcmed_nenner_zeit_v16.py` | cleaned original | `21c453594e0ad86cbb959b1bd14ba7f41894d770928bc13a5e9b04427d96a0fa` | `133c7032334abf16a5b0cf9628bc3689eaed06b5b9a288cbdccffd8036a17407` | user or cloud-folder names removed (1) |
| `ed_waveform_validation/s5_mcmed_pipeline.py` | cleaned original | `dcc1a15238f5cd370ef11eec9d4b721e1823ffe51a30ec873c82f80bfecf07a6` | `cda2092047865a190dc6fbb3203cf497d06a108b2018534a75c529a09cdc28f9` | local paths replaced by data_path (4); user or cloud-folder names removed (6) |
| `ed_waveform_validation/s5_mcmed_wacht.py` | cleaned original | `4389573abd43e8628ddf75f0e7e3c9639ad3926613fb275bd831e18f619ca18e` | `8fc0e6693ed6140df4968fb56fe8b2405a904b70589e1a208e1f6700e4fa701f` | local paths replaced by data_path (2); user or cloud-folder names removed (2) |
| `ed_waveform_validation/s5_mcmed_zaehlung.py` | cleaned original | `6e393e54afe0d03afc75b269f9ddc1bd4bdb2271e9e99c2b0b6d379ad1ecc357` | `71b8bd0770168c7c618f9369f604dee22b957942f947b25467639ed97e382a43` | local paths replaced by data_path (2); user or cloud-folder names removed (2) |
| `ed_waveform_validation/start_wmi.ps1` | cleaned original | `d560db5e62f479628d34a048330409be8e5b20eec457c25d3676aef900ae5c2d` | `9905ed2891f71e1ca8416943c0e3c74d839bf9b236a76e4c8a9fe7553be7bfa1` | user or cloud-folder names removed (2) |
| `figures/fig1_s5_zeitversatz.py` | translated | `2d64d50a2312d4d74d467b429192870a5d0cf8ea00bd79a327729d3736b6bb5f` | `dc70c0e4938d6d38e95e631d52f3d6c38d500acae69abcd76b172f71a56495dd` | comments, docstrings and messages translated; local paths replaced by data_path (1); user or cloud-folder names removed (5); house-style module replaced by a plain matplotlib style |
| `landmark_selection/a155_research_af_landmark_revision002.py` | cleaned original | `d9d26db23ab4ec85953b51a96d7d5dc91307f30345f13842c9c19081a757ad67` | `c1916f8f9cc1d66f19c594f72fbe39b6e51fba48daee16a6acd9582da077441f` | local paths replaced by data_path (1); user or cloud-folder names removed (1) |
| `landmark_selection/a155_research_documented_af.py` | cleaned original | `dbd833afbbfe564799cecbff657df95fe5880602244232ac4b4580a6b7719be3` | `1425bf2d11825aa89f5a9593fc963ce34685496ed37a321e3496b8ee0c11d562` | local paths replaced by data_path (1); user or cloud-folder names removed (1) |
| `landmark_selection/a155_research_low_cost.py` | cleaned original | `8d7a522a9bcb6876a0037a8d0e25a51cd03a9432cc8344043e6177b2b25f950a` | `d46cf72996b49bc12780a84c4af1a955eadd0ef18b4002d41ebe6ed5134a6117` | local paths replaced by data_path (2); user or cloud-folder names removed (2) |
| `landmark_selection/onedrive_names.py` | cleaned original | `3193d36b8cc2ee0abb1f2ad8dcbed9266912e106fb70d9afd8dc798753c3abfc` | `6985dfba6d092c28d3afdffa869f5dba6e775620876f1c8db6099ae259dfcd2e` | user or cloud-folder names removed (2) |
| `wave0_checks/s5_r1_abgleich.py` | cleaned original | `fad55b2b3fa1d3dc99c39889f7f0e90dc5dd26f615bb28332a4aa49898d1df63` | `7ae12118ec8d70ec0d539daf9b3c9e10c7cf803d19255a7b05b4237da6a00ef9` | local paths replaced by data_path (1); user or cloud-folder names removed (2); unpublished values moved to the local reference file (2); unpublished numbers removed from texts |
| `wave1_addendum/g243_audit_agg.py` | cleaned original | `31ef1bd2dcd64b0506ad6d41a78f2c5afb8f68d5126ef8e7ebcb643f2a528e2e` | `f3a0981034266921965bed96dc6e314dd60748e31129589d554347f85258003d` | local paths replaced by data_path (1) |
| `wave1_addendum/g243_audit_zaehlung.py` | cleaned original | `6b0f4f20f3850e8c0c1b16e4b1635a0c87f9adf09b9ae597f415a2950fac688a` | `0e9d21bb84c9386da62aa7bb200575577a63597fa9653feb14c3b45476629bb9` | local paths replaced by data_path (3) |
| `wave1_addendum/g243_guard.py` | cleaned original | `7437532b05c05f6fce2dd1b66322049df8b025674003dd1daefb24bb6c3a4b51` | `5dc689fda14a111d8186dc88b3ced20f20a9759758ee10d744acbb3d064affe9` | local paths replaced by data_path (1) |
| `wave1_addendum/g243_kern.py` | original, unchanged | `5fed91467efe202905f5fb890c3762f1a79d113d13203f7875037c87cf4ab319` | same file | - |
| `wave1_addendum/g243_null_agg.py` | cleaned original | `ad51c1dff2281e959d664dab973628fd210d1692f970aefd2b5dfacf36ac12b1` | `cba604d032c4594fc2ce5caf5219470978c9b5df5b7ef86f993cd962d5ede72a` | local paths replaced by data_path (2) |
| `wave1_addendum/g243_null_lauf.py` | cleaned original | `01cf6490f0bb2d3a1d9768714ea99aef1cfbb1bfa44b9ac21f72a8efc88e3709` | `33a8d69dc7fadaa3d851f5e8e0f8ad9ebdcd947725383a619ad4abfc801aad39` | local paths replaced by data_path (1) |
| `wave1_addendum/g243_null_lib.py` | original, unchanged | `107d51eb73e06d84bb3f2165b431991c4a60bf34cd4ee9ef62da01d6b70ea7a9` | same file | - |
| `wave1_addendum/g243_r4_lauf.py` | cleaned original | `4671edc6039b68a12aea359bcc97597a2624483eb35d141397d7c138785d93ca` | `c9995bd78991d9aef57bb7d9284f868163bfc829bb9c4172fbffcb101908f95f` | local paths replaced by data_path (4); unpublished numbers removed from texts |
| `wave1_addendum/g243_r4_sqlcheck.py` | cleaned original | `06870ffa940d4dbab01d163c008be51418273c16ce3e793191bfabe97bdf01f6` | `f7dc00fee7980c737d065074f0d7a4ea901171ce317861941ac8fe604201bf7e` | local paths replaced by data_path (2) |
| `wave1_alignment/agg_publish.py` | cleaned original | `51ed1369af30b88a73a3619a096296b4a030aacc0c44977805c8827017b6346b` | `3290967333677589026c13ae81e89b872a0bb3b1be720afe017be9854a6fa871` | local paths replaced by data_path (1); unpublished values moved to the local reference file (3); unpublished numbers removed from texts |
| `wave1_alignment/aw242_data.py` | cleaned original | `cdaea7ffbc128705dddefacd699ee9ee553edf62d889232b2341925799c7d967` | `e3a007e188fea96e9f972c4220458f340269d739691d241c842405086452dba8` | local paths replaced by data_path (1) |
| `wave1_alignment/aw242_guard.py` | cleaned original | `d3e03d1c4158e603ca36568648a166eb0d4d8ffc2e658072921d778f30e2666d` | `2db47066ac6ade36e94df4a178dbaba11b53da431b79db118d9a6e2050696fab` | local paths replaced by data_path (1) |
| `wave1_alignment/aw242_kern.py` | original, unchanged | `8ebb2b92629fd8d61b3967502333839f363789a5ed75cb50d160bc2aa7f24c83` | same file | - |
| `wave1_alignment/e0_extract_raw_hr.py` | cleaned original | `a912fed77cad32ece200d83975813455e04457ddb032da1a3d8fdbac28dffee1` | `90fffe16a6184d95328ac2efc18901798bd1de4703dc12d9aba6a629378e9a67` | local paths replaced by data_path (3) |
| `wave1_alignment/e1_delay_stats.py` | cleaned original | `03c15916b5976353d60f1ee6a89dca387ac5bdc1915e4413714bb97c759e82bd` | `16444fb00296da90e816816486f2f332f8f5bef1a689a36a983dc8fd0aa7a207` | local paths replaced by data_path (1) |
| `wave1_alignment/r0_baseline.py` | cleaned original | `734897e8f8eab1c6de446cfb7953c5d5da366ccdc72aac31e7758bb7304a41b5` | `ebfe2632ecbe4b7f99282314845f065e94819c6943599b07c64e9fb43270ab90` | local paths replaced by data_path (1) |
| `wave1_alignment/r1_gegenrechnung.py` | cleaned original | `f9544e4e818ba3ed3bd856e40c5d7b7df762ba8cdeeda20bebce14c3a47b1893` | `704aa9df4bf45c5ababac90c6a2464e1686810439b89c8962d8a08f95597674d` | local paths replaced by data_path (1) |
| `wave1_alignment/r2_run.py` | cleaned original | `753eb2b5a883fdf4b2bf09ccc0d2790ed25526ce88ef6e0de75150c02bc59257` | `a9069db8f4bc5174f92a0ec66433e9c2088b18bd1af8a8154f41b1d2231b47a9` | header or path setup only |
| `wave1_alignment/r2_sim_lib.py` | original, unchanged | `8ccb28a9c08e78d0014dc28188db26ceedf22ad6c108e384ddb122891e0b6ce2` | same file | - |
| `wave1_alignment/r3_loss.py` | cleaned original | `a677ae681c980079007cf5ac5c2fd505a9998083200dd9bc6475d7ef4474725d` | `edafd615a648ad433b56e0c8d9ba86a0e369d54f05c623ba94b0e5d783a23674` | local paths replaced by data_path (2) |
| `wave1_alignment/r4_af.py` | cleaned original | `3d84960030490474d903e43d54dd01db30a4a92a02d83978be4512ac8445a050` | `186662ce96755e2f5a4a04731b001e9dffa24bbbedd453aa873648cf94aa0c7e` | local paths replaced by data_path (4) |
| `wave1_alignment/r5_ref.py` | cleaned original | `cc69398c345b2f45fa56b8407099b03fb27dd26362d247f6b2e728e5c181fa5e` | `d766d225a97363e9d305b9e9dcb2141f299475659fe41e5e04226282e446a9e2` | local paths replaced by data_path (4) |
| `wave1_alignment/r5b_pairs.py` | cleaned original | `df143bf8947b31b6c820b28952f3c7edd44a8103edd7dc4a8317d1b97f0c831e` | `4dce12efe17614599f7aeae9c16c7aed9083308b6129c7531e4cc9341ef3bf20` | local paths replaced by data_path (1) |
| `wave2_individual_support/s5_r6_kleinzellen.py` | original, unchanged | `78aaff48981e0e7fb45cad51c51a84f54eb29b04334d760374baab7d3be69c7a` | same file | - |
| `wave2_individual_support/s5_r6_lauf.py` | cleaned original | `0a2c4324f607c642c36e95b3388aa8d1652a15fe7358afc08525de95940ef421` | `9d7d3fdacf6ae9afc09d14fae83c50f37e58386c75d62611160e9121d233067b` | local paths replaced by data_path (2) |
| `wave2_individual_support/s5_r6_lib.py` | cleaned original | `e8c7fc9e08f62b55539c3e307f4d076de528c0ed9b5a186aef7119fe724cf9ef` | `16e2fac25a6e43a87e61b42532af6a3b2711935c8eefc941a8a671802a940357` | header or path setup only |

## What is not in this folder

* Self-tests and unit tests of the analysis code, download helpers and packaging of the manuscripts (not part of the reported analyses).
* All data, result tables, intermediate files and the local reference-value file.
* The unpublished expected values of 3 scripts: they were moved to a local file that is not published (see the root README, section Running the scripts).
* Modules shared with other works are in `../shared_support/`.

## Reproducing the numbers

The numbers are in the manuscript tables and the supplement. To reproduce an analysis, obtain the data, produce the LoopLens output tables with the snapshot, fill `config.toml` and adapt the scripts to your layout. Random seeds are set in the scripts and marked as such.

