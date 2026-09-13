# -*- coding: utf-8 -*-
"""
Tables 1a, 1b and 2 of the main text.

Table 1a  Preoperative characteristics eligible as candidate predictors, by grade.
Table 1b  Postoperative and pathological characteristics, descriptive only.
Table 2   The nine inflammatory indices: distribution by grade, effect size,
          discrimination, and Benjamini-Hochberg FDR correction within the family.

Run after the notebook (01). Writes to ../outputs/.

    python 02_tables_1_2.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

SEED = 42
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
os.makedirs(OUT, exist_ok=True)

# The data file is not distributed with this repository; see ../data/README.md
SAV = os.environ.get('EC_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', 'data', 'endometrium.sav'))


# ---------------------------------------------------------------------- data
def load():
    df, _ = pyreadstat.read_sav(SAV)
    df = df.dropna(subset=['GRADE']).copy()
    df['y'] = (df['GRADE'] == 3).astype(int)          # target: G3 = 1, G1-G2 = 0

    # White cell count is corrected against the differential BEFORE any
    # collinearity diagnostic, so that the diagnostics see the corrected values.
    dsum = df[['nötrofil', 'lenfosit', 'monosit', 'eozinofil']].sum(axis=1)
    bad = (df['wbc'] - dsum).abs() > 2
    df.loc[bad, 'wbc'] = dsum[bad]

    df = df.rename(columns={
        'nötrofil': 'neu', 'lenfosit': 'lym', 'monosit': 'mon', 'eozinofil': 'eos',
        'platelet': 'plt', 'wbc': 'WBC', 'Dage': 'age', 'ca125': 'CA125',
        'albumin': 'albumin_preop', 'albuminpo1': 'albumin_postop',
        'yatısgun': 'los', 'totallns': 'ln_total', 'pelviklnds': 'ln_pelvic',
        'paraalnd': 'ln_paraaortic', 'lenfndisek': 'lymphadenectomy',
        'nlr': 'NLR', 'plr': 'PLR', 'pıv': 'PIV', 'sıı': 'SII',
        'sırı': 'SIRI', 'mlr': 'MLR', 'dnlr': 'dNLR', 'elr': 'ELR'})
    df['LMR'] = df['lym'] / df['mon']

    # dNLR is RECOMPUTED from the corrected white cell count rather than taken
    # from the stored column, which was derived before the correction:
    #     dNLR = neutrophils / (WBC - neutrophils)
    # This is what Table 2 of the article reports. The stored column differs in
    # the third decimal and is what the feature sets use, so the two are kept
    # separate: dNLR_table2 here, dNLR in the modelling notebook.
    df['dNLR_raw'] = df['dNLR']
    df['dNLR'] = df['neu'] / (df['WBC'] - df['neu'])
    return df


# ------------------------------------------------------------------ helpers
def fmt_mean(x):
    """mean +- SD, on the non-missing values only"""
    x = pd.Series(x).dropna()
    return '%.2f ± %.2f' % (x.mean(), x.std(ddof=1))


def fmt_med(x):
    """median (Q1-Q3)"""
    x = pd.Series(x).dropna()
    q1, q3 = np.percentile(x, [25, 75])
    return '%.2f (%.2f–%.2f)' % (np.median(x), q1, q3)


def rank_biserial(a, b):
    """Rank-biserial correlation from the Mann-Whitney U statistic.

    r = 2U/(n1*n2) - 1, signed so that positive means higher values in group b
    (the high-grade group). Reported instead of a raw U because it is bounded
    and comparable across indices.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    u = stats.mannwhitneyu(b, a, alternative='two-sided').statistic
    return 2 * u / (len(a) * len(b)) - 1


def _midrank(a):
    """Mid-ranks, ties averaged. Helper for the DeLong variance."""
    s = np.argsort(a)
    a2 = a[s]
    n = len(a)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and a2[j + 1] == a2[i]:
            j += 1
        t[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n)
    out[s] = t
    return out


def auc_ci(y, x, oriented=True):
    """AUC with a DeLong 95% confidence interval, on the non-missing rows.

    The interval is DeLong's, not a bootstrap: this is the method stated in the
    footnote of Table 3 of the article, and a percentile bootstrap differs from
    it in the third decimal, which would make the table irreproducible.
    Implemented in the fast form of Sun and Xu (2014), for a single curve.

    oriented=True reports the AUC in whichever direction discriminates, i.e.
    max(AUC, 1 - AUC). Several of these indices are inversely associated with
    grade (LMR above all), and an unoriented AUC below 0.50 for those would
    describe the same discrimination as its mirror image. The published table
    reports the oriented value; the sign of the association is carried by the
    rank-biserial column instead.
    """
    m = ~pd.isna(x)
    y, x = np.asarray(y)[m], np.asarray(x, float)[m]
    if oriented and roc_auc_score(y, x) < 0.5:
        x = -x                                     # mirror, so AUC >= 0.50

    pos, neg = x[y == 1], x[y == 0]
    n1, n0 = len(pos), len(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    tx, ty = _midrank(pos), _midrank(neg)
    auc = (tz[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    # structural components: V10 over cases, V01 over controls
    v10 = (tz[:n1] - tx) / n0
    v01 = 1 - (tz[n1:] - ty) / n1
    se = np.sqrt(np.var(v10, ddof=1) / n1 + np.var(v01, ddof=1) / n0)
    return auc, auc - 1.96 * se, auc + 1.96 * se


def counts(mask_lo, mask_hi, n_lo, n_hi):
    return '%d (%.1f%%)' % (mask_lo.sum(), 100 * mask_lo.sum() / n_lo), \
           '%d (%.1f%%)' % (mask_hi.sum(), 100 * mask_hi.sum() / n_hi)


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def pfmt(p):
    return ('<0.001 ' + stars(p)) if p < 0.001 else ('%.3f %s' % (p, stars(p))).strip()


# ============================================================== Table 1a
def table1a(df):
    lo, hi = df[df.y == 0], df[df.y == 1]
    rows = []

    # Age: both groups approximately normal, Student t
    p = stats.ttest_ind(hi['age'].dropna(), lo['age'].dropna()).pvalue
    rows.append(['Age (years), n = %d/%d' % (lo['age'].notna().sum(), hi['age'].notna().sum()),
                 fmt_mean(lo['age']), fmt_mean(hi['age']), pfmt(p)])

    # CA-125 and albumin: skewed, Mann-Whitney U
    for col, lbl in (('CA125', 'CA-125 (U/mL)'), ('albumin_preop', 'Preoperative albumin (g/dL)')):
        a, b = lo[col].dropna(), hi[col].dropna()
        p = stats.mannwhitneyu(b, a, alternative='two-sided').pvalue
        rows.append(['%s, n = %d/%d' % (lbl, len(a), len(b)),
                     fmt_med(a), fmt_med(b), pfmt(p)])

    T = pd.DataFrame(rows, columns=['Characteristic',
                                    'Low-grade (G1–G2) n = %d' % len(lo),
                                    'High-grade (G3) n = %d' % len(hi), 'p'])
    T.to_csv(os.path.join(OUT, 'table1a_preoperative.csv'), index=False)
    return T


# ============================================================== Table 1b
def table1b(df):
    lo, hi = df[df.y == 0], df[df.y == 1]
    n_lo, n_hi = len(lo), len(hi)
    rows = []

    # Welch t where variances differ, Student t where they do not; see the
    # manuscript footnote for which test applies to which row.
    for col, lbl, kind in (
            ('albumin_postop', 'Postoperative day-1 albumin (g/dL)', 'welch'),
            ('los', 'Length of stay (days)', 'mw'),
            ('ln_total', 'Total lymph nodes harvested', 'welch'),
            ('ln_pelvic', 'Pelvic lymph nodes harvested', 'student'),
            ('ln_paraaortic', 'Para-aortic lymph nodes harvested', 'mw')):
        if col not in df.columns:
            continue
        a, b = lo[col].dropna(), hi[col].dropna()
        if kind == 'mw':
            p = stats.mannwhitneyu(b, a, alternative='two-sided').pvalue
            va, vb = fmt_med(a), fmt_med(b)
        else:
            p = stats.ttest_ind(b, a, equal_var=(kind == 'student')).pvalue
            va, vb = fmt_mean(a), fmt_mean(b)
        rows.append(['%s, n = %d/%d' % (lbl, len(a), len(b)), va, vb, pfmt(p)])

    # Lymphadenectomy: proportions, chi-square
    if 'lymphadenectomy' in df.columns:
        # Missing values are NOT counted as "not performed": the denominator is
        # the number of patients in whom the field was recorded.
        al, ah = lo['lymphadenectomy'].dropna(), hi['lymphadenectomy'].dropna()
        ml, mh = al == 1, ah == 1
        tab = np.array([[ml.sum(), (~ml).sum()], [mh.sum(), (~mh).sum()]])
        # Without the Yates continuity correction, matching the published table.
        p = stats.chi2_contingency(tab, correction=False)[1]
        cl, ch = counts(ml, mh, len(al), len(ah))
        rows.append(['Lymphadenectomy performed, n = %d/%d' % (len(al), len(ah)),
                     cl, ch, pfmt(p)])

    # Hormone receptor status: DESCRIPTIVE ONLY, and reported as
    # assessed / not assessed / no record rather than positive / negative,
    # because the source coding contains no negative category. No test is
    # reported: it would compare grade with whether the assay was requested.
    for src, name in (('östrojen', 'ER'), ('progesteron', 'PR')):
        if src not in df.columns:
            continue
        for lbl, sel in (('%s assessed and positive' % name, lambda s: s > 0),
                         ('%s not assessed' % name, lambda s: s == 0),
                         ('%s no record' % name, lambda s: s.isna())):
            ml, mh = sel(lo[src]), sel(hi[src])
            cl, ch = counts(ml, mh, n_lo, n_hi)
            rows.append([lbl, cl, ch, '—'])

    T = pd.DataFrame(rows, columns=['Characteristic', 'Low-grade (G1–G2)',
                                    'High-grade (G3)', 'p'])
    T.to_csv(os.path.join(OUT, 'table1b_postoperative.csv'), index=False)
    return T


# ============================================================== Table 2
IDX = ['NLR', 'dNLR', 'PLR', 'MLR', 'LMR', 'SII', 'SIRI', 'PIV', 'ELR']


def table2(df):
    """The nine inflammatory indices, with FDR correction WITHIN this family.

    The correction is applied here and nowhere else: the family is the nine
    indices tested against the same outcome.
    """
    lo, hi = df[df.y == 0], df[df.y == 1]
    rows, pvals = [], []

    for k in IDX:
        a, b = lo[k].dropna(), hi[k].dropna()
        p = stats.mannwhitneyu(b, a, alternative='two-sided').pvalue
        r = rank_biserial(a, b)
        m = df[k].notna()
        auc, cl, ch = auc_ci(df.loc[m, 'y'].values, df.loc[m, k].values)
        rows.append([k, fmt_med(a), fmt_med(b), '%+.3f' % r if r >= 0 else '−%.3f' % abs(r),
                     '%.3f (%.3f–%.3f)' % (auc, cl, ch), '%.3f' % p])
        pvals.append(p)

    # Benjamini-Hochberg across the nine indices
    q = multipletests(pvals, method='fdr_bh')[1]
    for i, qq in enumerate(q):
        rows[i].append('%.3f' % qq)

    T = pd.DataFrame(rows, columns=['Index', 'Low-grade\nmedian (IQR)', 'High-grade\nmedian (IQR)',
                                    'Effect size\n(rank-biserial r)', 'AUC (95% CI)', 'p', 'q (BH–FDR)'])
    T.to_csv(os.path.join(OUT, 'table2_inflammatory_indices.csv'), index=False)
    return T


# ---------------------------------------------------------------------- main
if __name__ == '__main__':
    df = load()
    print('analytic cohort n = %d, high-grade events = %d\n' % (len(df), int(df.y.sum())))

    for name, T in (('Table 1a', table1a(df)), ('Table 1b', table1b(df)), ('Table 2', table2(df))):
        print('===', name, '=' * 50)
        print(T.to_string(index=False))
        print()

    print('written to', os.path.normpath(OUT))
