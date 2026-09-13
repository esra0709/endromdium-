# Hematological Versus Clinical Predictors of Tumor Grade in Endometrial Cancer

Analysis code for:

> Akaydın Gültürk, E.; Genç, Ş.Ö. Hematological Versus Clinical Predictors of Tumor Grade in
> Endometrial Cancer: A Comparative Machine Learning Study with SHAP-Based Explainability.
> *Diagnostics* **2026**, *16*, xxxx. https://doi.org/10.3390/xxxxx

This repository contains the analysis pipeline that generates every table and figure reported
in the article and in its Supplementary Material.

---

## Data availability

**The patient-level data are not included in this repository and cannot be distributed here.**

The underlying records are pseudonymized rather than anonymized. A fully anonymized extract,
from which the linkage key has been destroyed, can be made available upon reasonable request
and subject to approval by the Non-Interventional Clinical Research Ethics Committee of Sivas
Cumhuriyet University (decision no. 2024/09-50, dated 19 September 2024).

Place the data file at `data/endometrium.sav`, or point the scripts at it:

```bash
export EC_DATA=/path/to/your/file.sav
```

`.gitignore` excludes everything in `data/` except its README, so a data file placed there
cannot be committed by accident.

---

## How to run

The notebook must run first: it fits the primary models and writes `outputs/oof_primary.npz`,
which three of the four scripts read.

```bash
pip install -r requirements.txt

# 1. primary analysis — Tables 3, 4, S2-S7 and the out-of-fold probabilities
jupyter nbconvert --execute --to notebook --inplace analysis/endometrium_analysis.ipynb

# 2. descriptive tables and the inflammatory index family
python analysis/02_tables_1_2.py

# 3. figures
python analysis/03_figure1_flow.py        # Figure 1
python analysis/03_figures_2_4_5.py       # Figures 2, 4, 5
python analysis/03_figure3_shap.py        # Figure 3

# 4. exploratory nine-classifier comparison
python analysis/04_figure6_algorithms.py  # Figure 6 and Table S5b
```

Everything lands in `outputs/`.

Step 4 re-fits nine algorithms and takes roughly half an hour. Its script has a `CALISTIR`
switch: left at `False` it redraws the figure from the saved `tableS5b_nine_algorithms.csv`
in seconds. Set it to `True` to refit from the data.

### What produces what

| Output | Produced by |
|---|---|
| Table 1a, Table 1b | `02_tables_1_2.py` |
| Table 2 (nine inflammatory indices, BH-FDR) | `02_tables_1_2.py` |
| Table 3 (primary model, three cohorts) | notebook |
| Table 4 (incremental value) | notebook |
| Tables S2, S2b, S2c (missingness) | notebook |
| Table S3 (common cohort) | notebook |
| Table S4 (fold-wise collinearity) | notebook |
| Table S5 (subtype by grade) | notebook |
| Table S5b (nine classifiers) | `04_figure6_algorithms.py` |
| Table S6 (hyperparameters) | notebook |
| Table S7 (ER/PR assessment pattern) | notebook |
| Table S8 (non-superiority) | notebook |
| Table S9 (refit rule) | notebook |
| Table S10 (MICE) | notebook |
| Figure 1 (participant flow) | `03_figure1_flow.py` |
| Figures 2, 4, 5 (ROC, calibration, DCA) | `03_figures_2_4_5.py` |
| Figure 3 (SHAP) | `03_figure3_shap.py` |
| Figure 6 (nine classifiers) | `04_figure6_algorithms.py` |

Supplementary Table S1 appraises seven published studies and is not computed from data.

### Two places where the printed output and the article differ in the last digit

Both are rounding, not disagreement, and are recorded here so that neither is mistaken for a
failure to reproduce.

- **Table 2, ELR, q value.** The Benjamini–Hochberg adjusted value is 0.03665. The script prints
  `0.037`; the article prints `0.036`. The conclusion — the only index to survive correction — is
  the same either way.
- **Section 3.6, sixth sensitivity analysis.** The clinical set is reported there as AUC 0.690 with
  a Brier score of 0.224, against 0.687 and 0.225 in Table 3. These are not the same estimate:
  Section 3.6 quotes Supplementary Table S9, which re-runs the model under both hyperparameter
  refit rules in a single self-contained comparison, and its own "best inner-loop score" baseline is
  0.6898 / 0.2241. Table 3 is the main twenty-repeat analysis. The script reproduces both.

---

## What the code does

Prediction target: preoperative discrimination of high-grade (G3) from low-grade (G1–G2)
endometrial carcinoma.

**Four feature sets.** Estrogen and progesterone receptor status appear in none of them.
Receptor status is determined on the hysterectomy specimen and so is not available at the time
of prediction; separately, the source coding contains no negative category, and the value that
reads as "negative" denotes "not assessed". The variable is documented in section 12 of the
notebook and used in no model.

| Set | Contents | p |
|---|---|---|
| A | Reduced inflammatory panel, after collinearity reduction | 8 |
| A-full | Full inflammatory panel | 14 |
| B | Preoperative clinical variables (age, CA-125, preoperative albumin) | 3 |
| C | Combined, B + A | 11 |

**One note on dNLR.** Two dNLR values exist in this pipeline, and the difference is
deliberate rather than an inconsistency:

| Where | Value used | Why |
|---|---|---|
| Table 2, and any univariate statistic | recomputed as `neu / (WBC − neu)` from the **corrected** white cell count | this is the definition stated in Section 2.3 of the article, and the correction described in Section 2.2 precedes it |
| Set A-full, in the modelling notebook | the stored `dnlr` column of the source file | this is what was fitted for the published Table 3, and reproducing the published table requires it |

The stored column was derived before the white cell count was corrected. It differs from
the recomputed value in 14 of 225 records; in one record the stored value is 0.000, which
is impossible. Refitting Set A-full on the recomputed value moves the pooled out-of-fold
AUC from 0.520 to 0.521 in the full cohort and from 0.556 to 0.568 in the endometrioid
cohort — that is, the choice does not change any conclusion, but it does change the third
decimal of a published number, so the code keeps the published one. dNLR never enters Set
A: the collinearity reduction drops it as algebraically redundant with NLR (Section 2.4).

**Three cohorts**, every analysis repeated in each.

| Cohort | n | High-grade events |
|---|---|---|
| Full | 225 | 59 (26.2%) |
| Endometrioid only | 183 | 20 (10.9%) |
| Receptor-assessed | 159 | 28 (17.6%) |

**Prespecified primary model**

| Setting | Value |
|---|---|
| Model | Elastic-net logistic regression |
| Solver | `saga`, `max_iter=20000` |
| C grid | `np.logspace(-5, 1, 20)` |
| l1_ratio grid | `[0.0, 0.5, 1.0]` |
| Search | Exhaustive grid (`GridSearchCV`), scoring `roc_auc` |
| Inner resampling | Stratified 5-fold |
| Outer resampling | Stratified 5-fold |
| Outer repeats | `N_REPEATS = 20` |
| Bootstrap resamples | `N_BOOT = 5000` |
| Permutations | `N_PERM = 200` |
| Class imbalance | `class_weight='balanced'` |
| Imputation | Fold-wise median (no-op in complete-case sets) |
| Scaling | `StandardScaler`, fitted within fold |
| Decision threshold | Youden J, selected **within each training fold** |

Every preprocessing step sits inside the pipeline, so it is refitted within each fold; nothing
is fitted on data used for evaluation. A single algorithm is prespecified: the comparison among
nine classifiers is exploratory and reported as such, because selecting the best of nine on
pooled out-of-fold performance would carry selection optimism.

**Also computed**

- Permutation of the outcome labels with the full pipeline re-run (null AUC distribution)
- Benjamini–Hochberg FDR correction within the family of nine inflammatory indices
- One-sided non-superiority tests against a discrimination margin of 0.70, reported as
  exploratory because the margin was adopted post hoc
- Flexible calibration curves, calibration-in-the-large and calibration slope, before and after
  logistic recalibration
- Decision curve analysis on recalibrated out-of-fold probabilities, with bootstrap bands
- SHAP values for an exploratory gradient-boosted tree model, on the log-odds scale
  (`model_output="raw"`), not on the probability scale
- Sensitivity analyses: exclusion of records with an inconsistent differential count, the
  alternative grade dichotomisation, fold-wise repetition of the collinearity reduction,
  evaluation in the common complete-case population, multiple imputation by chained equations
  within the folds, and the hyperparameter refit rule

---

## Environment

Python 3.13.15 with:

```
numpy        2.1.3
scipy        1.16.3
scikit-learn 1.6.1
statsmodels  0.14.6
shap         0.52.0
```

Install with:

```bash
pip install -r requirements.txt
```

The random seed is fixed throughout (`SEED = 42`). Results reproduce on these package
versions; minor numerical differences can appear across versions of `scikit-learn` and `shap`.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── analysis/
│   ├── endometrium_analysis.ipynb   # primary analysis, run first
│   ├── 02_tables_1_2.py
│   ├── 03_figure1_flow.py
│   ├── 03_figures_2_4_5.py
│   ├── 03_figure3_shap.py
│   └── 04_figure6_algorithms.py
├── data/                            # NOT committed — see Data availability
└── outputs/                         # tables and figures produced by the code
```

The notebook is commented in Turkish; the scripts are commented in English.

---

## Citation

If you use this code, please cite the article above.

## Contact

Esra Akaydın Gültürk — Department of Biostatistics, Faculty of Medicine,
Sivas Cumhuriyet University, 58140 Sivas, Türkiye.
