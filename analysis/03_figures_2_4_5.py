# -*- coding: utf-8 -*-
"""
Figures 2, 4 and 5 — out-of-fold ROC curves, flexible calibration, and decision
curve analysis, all built from the out-of-fold probabilities the notebook saves
as oof_primary.npz.

Run after the notebook (01). Reads from and writes to ../outputs/.

    python 03_figures_2_4_5.py
"""
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
os.makedirs(OUT, exist_ok=True)

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm

Z = np.load(os.path.join(OUT, 'oof_primary.npz'), allow_pickle=True)
COH  = ['Full cohort', 'Endometrioid only', 'ER assessed']
COHL = {'Full cohort': 'Full cohort', 'Endometrioid only': 'Endometrioid tumors only',
        'ER assessed': 'Receptor-assessed subgroup'}
SETS = ['Set A', 'Set A-full', 'Set B', 'Set C']
SETL = {'Set A': 'Set A — inflammatory (reduced)', 'Set A-full': 'Set A-full — inflammatory (full)',
        'Set B': 'Set B — clinical', 'Set C': 'Set C — combined'}
SETS_SHORT = {'Set A': 'Set A', 'Set A-full': 'Set A-full', 'Set B': 'Set B', 'Set C': 'Set C'}
CLR  = {'Set A': '#1f77b4', 'Set A-full': '#ff7f0e', 'Set B': '#2ca02c', 'Set C': '#d62728'}

def get(c, s):
    return Z['%s|%s' % (c, s)], Z['y|%s|%s' % (c, s)].astype(int)

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def recal(p, y):
    """lojistik yeniden kalibrasyon: y ~ logit(p)"""
    X = sm.add_constant(logit(p))
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return m.predict(X)

# ------------------------------------------------------------------ Figure 4
fig, ax = plt.subplots(3, 4, figsize=(16.5, 11.6))
for i, c in enumerate(COH):
    for j, s in enumerate(SETS):
        a = ax[i, j]
        p, y = get(c, s)
        pr = recal(p, y)
        a.plot([0, 1], [0, 1], 'k--', lw=0.9, zorder=1)
        # ham olcekte kalibrasyon kesisimi ve egimi
        X = sm.add_constant(logit(p))
        m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                      offset=logit(p)).fit().params[0]
        slope = m.params[1]
        for pp, col, lab in ((p, '#c0392b', 'raw'), (pr, '#2471a3', 'recalibrated')):
            if np.ptp(pp) < 0.03:
                # egim ~0: yeniden kalibre olasiliklar prevalansa cokuyor
                a.plot([pp.min(), pp.max()], [y.mean(), y.mean()], color=col, lw=2.0,
                       label=lab, zorder=3)
                continue
            o = np.argsort(pp)
            sm_ = lowess(y[o], pp[o], frac=0.85, it=0, return_sorted=True)
            a.plot(sm_[:, 0], sm_[:, 1], color=col, lw=2.0, label=lab, zorder=3)
        a.plot(p, np.full_like(p, -0.035), '|', color='#c0392b', ms=4, alpha=.5, clip_on=False)
        a.set_xlim(0, 1); a.set_ylim(-0.05, 1)
        if i == 0:
            a.set_title(SETL[s], fontsize=15, pad=9)
        # "=" ve bosluk sart: "CITL-1.02" biciminde bitisik yazim eksi isaretini
        # etiketin parcasi gibi gosteriyor ve okunmuyor (uretim duzeltisi, yorum 195)
        a.text(0.97, 0.05, 'CITL = %.2f\nslope = %.2f' % (citl, slope), transform=a.transAxes,
               ha='right', va='bottom', fontsize=12, color='0.15',
               bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='0.80', lw=0.7, alpha=0.92))
        if j == 0: a.set_ylabel('%s\nObserved proportion' % COHL[c], fontsize=13.5)
        if i == 2: a.set_xlabel('Predicted probability', fontsize=13)
        a.tick_params(labelsize=11)
        if i == 0 and j == 0:
            a.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.92,
                     edgecolor='0.80')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'Figure4_calibration.png'), dpi=300, bbox_inches='tight')
print('Figure 4 yazildi')

# ------------------------------------------------------------------ Figure 5
def netben(y, p, t):
    pred = p >= t
    n = len(y)
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    return tp / n - (fp / n) * (t / (1 - t))

TH = np.linspace(0.05, 0.40, 71)
rng = np.random.default_rng(42)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for i, c in enumerate(COH):
    a = ax[i]
    _, y0 = get(c, 'Set B')
    prev = y0.mean()
    a.plot(TH, [prev - (1 - prev) * (t / (1 - t)) for t in TH], color='0.45', lw=1.6,
           label='Treat all')
    a.axhline(0, color='k', lw=1.2, label='Treat none')
    for s in SETS:
        p, y = get(c, s)
        pr = recal(p, y)
        nb = [netben(y, pr, t) for t in TH]
        lab = SETL[s] + (' (95% bootstrap band)' if s == 'Set B' else '')
        a.plot(TH, nb, color=CLR[s], lw=1.8, label=lab)
        if s == 'Set B':                                   # yalnizca klinik set icin bant
            B = np.empty((400, len(TH)))
            for b in range(400):
                idx = rng.integers(0, len(y), len(y))
                if y[idx].sum() < 3: idx = np.arange(len(y))
                B[b] = [netben(y[idx], pr[idx], t) for t in TH]
            a.fill_between(TH, np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0),
                           color=CLR[s], alpha=.16, lw=0)
    # Panel basligi yalnizca kohort adi. Onceki surum Set B'nin n/olay sayisini
    # basliga yaziyordu; oysa dort egri dort ayri tam-veri nufusundan geliyor
    # (or. tam kohortta Set A n = 224, Set B n = 181), bu yuzden tek bir n yanilticiydi.
    a.set_title(COHL[c], fontsize=13)
    a.set_xlabel('Threshold probability', fontsize=14)
    a.set_xlim(0.05, 0.40); a.tick_params(labelsize=12)
    a.set_xticks([0.10, 0.20, 0.30, 0.40])     # panel sinirlarinda etiket cakismasini onler
    if i == 0:
        a.set_ylabel('Net benefit', fontsize=14)
h5, l5 = ax[0].get_legend_handles_labels()
fig.legend(h5, l5, fontsize=12, frameon=False, ncol=3, loc='lower center',
           bbox_to_anchor=(0.5, -0.16))
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'Figure5_dca.png'), dpi=300, bbox_inches='tight')
print('Figure 5 yazildi')

# ------------------------------------------------------------------ Figure 2 (ROC)
from sklearn.metrics import roc_curve, roc_auc_score
fig, ax = plt.subplots(1, 3, figsize=(15, 5.0))
for i, c in enumerate(COH):
    a = ax[i]
    a.plot([0, 1], [0, 1], 'k--', lw=0.9)
    auc_txt = []
    for s in SETS:
        p, y = get(c, s)
        fpr, tpr, _ = roc_curve(y, p)
        a.plot(fpr, tpr, color=CLR[s], lw=1.8, label=SETL[s])
        # her set kendi tam-veri nufusunda degerlendirildigi icin n de kutuda verilir
        auc_txt.append('%s  %.3f  (n = %d)' % (SETS_SHORT[s], roc_auc_score(y, p), len(y)))
    # AUC degerleri egrilerin uzerine binmeyen, beyaz zeminli bir kutuda
    a.text(0.97, 0.04, 'AUC\n' + '\n'.join(auc_txt), transform=a.transAxes,
           ha='right', va='bottom', fontsize=11.5, linespacing=1.35,
           bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.75', lw=0.8, alpha=0.94))
    a.set_title(COHL[c], fontsize=13)
    a.set_xlabel('1 − specificity', fontsize=14)
    if i == 0: a.set_ylabel('Sensitivity', fontsize=14)
    a.tick_params(labelsize=12)
# dort set icin TEK ortak aciklama, panel basina tekrar yok
h2, l2 = ax[0].get_legend_handles_labels()
fig.legend(h2, l2, fontsize=12, frameon=False, ncol=4, loc='lower center',
           bbox_to_anchor=(0.5, -0.10))
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'Figure2_roc.png'), dpi=300, bbox_inches='tight')
print('Figure 2 yazildi')
