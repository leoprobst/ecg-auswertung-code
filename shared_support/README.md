# shared_support

Helper modules that several works import. Folder names are descriptive; the scripts keep their original file names and import paths (`werkzeuge.<original folder>.<module>`), which the compatibility package `werkzeuge/` maps to these folders.

10 files.


### `replication_support/`

Statistics and run helpers of the earlier replication tooling (logistic fit, resource guard, runners, checks).

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `__init__.py` | original, unchanged | Package marker of the development recomputation helpers. | support module for the shared regression and AUC library used throughout (Table 2; Supplement S3, S9-S27) |
| `aw054.py` | cleaned original | Early development analysis of the basic-model coefficients, stronger-basis AUC increments and shared echo/NT-proBNP reference, and source of the person-cluster sandwich variance function. | support module for the person-cluster robust (CR1) Wald intervals of Table 2 and Supplement S3 (its own development outputs are not reported) |
| `b3.py` | cleaned original | Early development recomputation of the basic model (B3) and its AUC variants with fold intervals. | not traceable to a reported item |
| `core.py` | cleaned original | Library of the logistic fit (ridge Newton), weighted AUC, natural cubic age basis, development guard and interval helpers used by the recomputation scripts. | support module for all fitted models of the study (Table 2; Supplement S3, S9-S27) |
| `resources.py` | original, unchanged | Monitors memory availability and the peak resource use of the own process and waits when memory is short. | support module for the test-partition replication runs (Supplement S13) |
| `runner.py` | cleaned original | Pilot runner of the first development recomputation with a zero-effect gate and two pilot endpoint pairs. | not traceable to a reported item |

### `tests/`

Tests of the variant code with synthetic data.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `test_varianten.py` | cleaned original | Unit tests with synthetic signals and identifiers for the curve-variant laboratory, including the test-curve generator reused by the variant evaluation. | support module for the rotation-removal comparison (Supplement S10, Table S40) |

### `transform_katalog/`

Catalog continuation and the sign-variant functions of the matrix-variant analysis.

| Script | Status | What it does | Manuscript part |
|---|---|---|---|
| `fortsetzung.py` | cleaned original | Runs the resumable, pilot-gated comparison that computes every catalog marker with the Kors and the inverse-Dower reconstruction for the 5,648 sample ECGs and writes the value blocks. | Table 4; Figure S1; Supplement S2c and S2f (matrix class tables); Supplementary Data S2 |
| `inventar.py` | cleaned original | Builds the inventory of the catalog markers (family, unit, implementation and schema checks) over the bound 5,648-ECG sample and verifies the sample and waveform mapping hashes. | support module for the matrix comparison (Table 4; Supplementary Data S2 marker identifiers) |
| `metriken.py` | original, unchanged | Library of the descriptive agreement metrics of the matrix comparison (bias, limits of agreement, Spearman correlation, relative limits-of-agreement width, quintile switches, technical class). | support module for the matrix classification (Section 2.4, Table 4, Supplement S2f) |

## Checksums and cleaning

| Script | Status | SHA-256 of the original | SHA-256 of the published file | Cleaning applied |
|---|---|---|---|---|
| `replication_support/__init__.py` | original, unchanged | `54725da8ea1bca131e28d24ef4d761e9e5a907fb0c31175e067dc705b93537f3` | same file | - |
| `replication_support/aw054.py` | cleaned original | `ccf9a056a11733db6acfc6414e88066e4e68a018ae5f7e96ea8d99e9b9e88652` | `78d2ee2e12c80f5d73013bdabc5c79ea5d36da6b070e6c0aa02e32746aa5743d` | local paths replaced by data_path (1) |
| `replication_support/b3.py` | cleaned original | `4e62e948ee8b0ceeaa0d6096c71a5193db322ecc08b2c760f5591e1ee6ffc736` | `ed9c87bed52244f7c861eeb6e5e5883af7d06d2994fff74906d6926108511958` | local paths replaced by data_path (3); long dashes replaced (10) |
| `replication_support/core.py` | cleaned original | `ccd42ffec3c12fa586609ada5f7450334a5f2802592004251d8ea03a33df5828` | `b514d72eeb35b643abdc9300217b05fb63ebdb76bd3463746fa20c9028b4ebba` | long dashes replaced (1) |
| `replication_support/resources.py` | original, unchanged | `04fac55f207ca852728156f0ccd5d18e611aa749cc1b08e26fcbdc653a19215a` | same file | - |
| `replication_support/runner.py` | cleaned original | `f52cc568c305507ea87f1d146424999f2593d3753a7198f110694a5414708dec` | `b1fb9c95321e5e25eb3f494b53da622e86056422f35d2282c059c91d66584cf6` | local paths replaced by data_path (3) |
| `tests/test_varianten.py` | cleaned original | `4ca8348445f4b743a8a6aa6416ecbb9eb28b8c3206e32f4d9a6297f36fa3c9f3` | `222a956f936e1f9fea7bbcbf7f56487124aac524380bfbde71bfc6782bd0aa22` | local paths replaced by data_path (4) |
| `transform_katalog/fortsetzung.py` | cleaned original | `113b7306caff58fb54f19ea84c91c822391cd28fde305b1bbeab914c25b9e3ce` | `f3cb7c969c041e012ea05b398a413008db7c243a98422d2633a555bb7aa47d2b` | local paths replaced by data_path (1) |
| `transform_katalog/inventar.py` | cleaned original | `754c788296e58600afca6795619d7ac87bb2a2c964dadf29e5d98f19d611b90e` | `70024c66a5ac0de16e7b4c4ce1cf0a211fbdc72e1cd09b513cfd22627d57988b` | local paths replaced by data_path (7) |
| `transform_katalog/metriken.py` | original, unchanged | `3e10302b1748c04d24d6357a80859d26edfb6ef934b35a83c21db65646e4fafc` | same file | - |
