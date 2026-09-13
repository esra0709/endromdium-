# -*- coding: utf-8 -*-
"""
Figure 1 — STROBE participant flow diagram.

Run after the notebook (01). Reads from and writes to ../outputs/.

    python 03_figure1_flow.py
"""
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
os.makedirs(OUT, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(11.4, 8.8))
ax.set_xlim(0, 12); ax.set_ylim(0.6, 10.4); ax.axis('off')

BOX = dict(boxstyle='round,pad=0.42', linewidth=1.1, edgecolor='#333333', facecolor='white')
EXC = dict(boxstyle='round,pad=0.38', linewidth=1.0, edgecolor='#999999', facecolor='#F4F4F4')
CO  = dict(boxstyle='round,pad=0.38', linewidth=1.0, edgecolor='#1B6E3B', facecolor='#F0F7F2')


def box(x, y, txt, style=BOX, fs=9.4):
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs, bbox=style, linespacing=1.45)


def vline(x, y1, y2):
    """ok basi YALNIZCA bitis noktasinda; ara baglantilar duz cizgi"""
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle='-|>',
                 mutation_scale=13, linewidth=1.1, color='#333333'))


def plain(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color='#333333', lw=1.1, solid_capstyle='butt')


MX = 4.30                                   # ana eksen
# ---------------------------------------------------------------- 1. basamak
box(MX, 9.75, 'Patients with endometrial carcinoma\noperated during the study period\nn = 226')
plain(MX, 9.32, MX, 8.55)                   # dislama dali bu cizgiden ayrilir
plain(MX, 8.95, 7.65, 8.95)
ax.add_patch(FancyArrowPatch((7.65, 8.95), (7.97, 8.95), arrowstyle='-|>',
             mutation_scale=13, linewidth=1.1, color='#333333'))
box(9.55, 8.95, 'Excluded (n = 1)\nTumor grade not reported\nin the final pathology report', EXC, 8.6)
vline(MX, 8.55, 8.30)

# ---------------------------------------------------------------- 2. basamak
box(MX, 7.90, 'Analytic cohort\nn = 225   (G1 133, G2 33, G3 59)\nHigh-grade events 59 (26.2%)')
vline(MX, 7.45, 7.05)
box(MX, 6.78, 'Feature sets (complete-case within each set)', BOX, 9.2)

# ---------------------------------------------------------------- dallanma
LX, RX = 3.05, 8.05                         # iki oznitelik seti sutunu
plain(MX, 6.45, MX, 6.10)                   # asagi inen govde
plain(LX, 6.10, RX, 6.10)                   # yatay dagitim cubugu, sadece iki sutun arasi
vline(LX, 6.10, 5.74)                       # ok basi yalnizca kutulara girerken
vline(RX, 6.10, 5.74)

box(LX, 5.28, 'Inflammatory sets\n(Set A, Set A-full)\nn = 224, events 59', BOX, 9.2)
box(RX, 5.28, 'Preoperative clinical and\ncombined sets (Set B, Set C)\nn = 181, events 48', BOX, 9.2)

# dislamalar YANDA, kutulara giren okun uzerinden ayrilir
for xb, xe, txt in ((LX, 1.05, 'Excluded (n = 1)\nIncomplete differential\nleukocyte count'),
                    (RX, 10.95, 'Excluded (n = 44)\nMissing preoperative\nalbumin, CA-125 or both')):
    side = -1 if xe < xb else 1
    plain(xb, 5.92, xe - side * 1.30, 5.92)          # cizgi kutunun iç kenarinda biter
    ax.add_patch(FancyArrowPatch((xe - side * 1.30, 5.92), (xe - side * 1.04, 5.92),
                 arrowstyle='-|>', mutation_scale=12, linewidth=1.1, color='#333333'))
    box(xe, 5.92, txt, EXC, 8.0)

# ---------------------------------------------------------------- kohortlar
# kohort satiri: baglanti cizgisi yaziyi kesmesin diye once baslik, sonra dagitim cubugu
ax.text(6.0, 4.42, 'Cohorts in which every analysis was repeated',
        fontsize=9.8, style='italic', color='#333333', ha='center', va='center')

CX = (2.30, 6.00, 9.70)
plain(CX[0], 3.95, CX[2], 3.95)
for x in CX:
    vline(x, 3.95, 3.52)

box(CX[0], 2.80, 'Full cohort\nn = 225\nevents 59 (26.2%)\nendometrioid 81.3%', CO, 9.0)
box(CX[1], 2.80, 'Endometrioid only\nn = 183\nevents 20 (10.9%)\nendometrioid 100%', CO, 9.0)
box(CX[2], 2.80, 'Receptor assessed\nn = 159\nevents 28 (17.6%)\nendometrioid 87.4%', CO, 9.0)

ax.text(0.30, 1.85,
        'Analyses within each cohort: nested cross-validation of the prespecified elastic-net model in four\n'
        'feature sets; calibration and decision curve analysis on recalibrated out-of-fold probabilities;\n'
        'incremental value of the inflammatory panel over the preoperative clinical variables.',
        fontsize=8.8, color='#444444', ha='left', va='top', linespacing=1.6)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'Figure1_flow.png'), dpi=300, bbox_inches='tight')
print('Figure 1 (akis diyagrami) yazildi')
