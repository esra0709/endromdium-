# -*- coding: utf-8 -*-
"""
Figure 3 — SHAP beeswarm for the exploratory tree model, log-odds scale.

Run after the notebook (01). Reads from and writes to ../outputs/.

    python 03_figure3_shap.py
"""
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
os.makedirs(OUT, exist_ok=True)

# The data file is not distributed with this repository; see ../data/README.md
SAV = os.environ.get('EC_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', 'data', 'endometrium.sav'))

import warnings, numpy as np, pandas as pd, pyreadstat, shap
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

SEED = 42
df, _ = pyreadstat.read_sav(SAV)
df = df.dropna(subset=['GRADE']).copy()
df['y'] = (df['GRADE'] == 3).astype(int)
df = df.rename(columns={'Dage': 'age', 'ca125': 'CA125', 'albumin': 'albumin_preop'})

COLS = ['age', 'CA125', 'albumin_preop']
NICE = {'age': 'Age', 'CA125': 'CA-125', 'albumin_preop': 'Preoperative albumin'}

m = df[COLS].notna().all(axis=1).values          # Set B tam-vaka kohortu
Xb = df.loc[m, COLS]
yb = df.loc[m, 'y'].values
Ximp = SimpleImputer(strategy='median').fit_transform(Xb)

gb = GradientBoostingClassifier(random_state=SEED, max_depth=2, n_estimators=200,
                                learning_rate=0.05).fit(Ximp, yb)
sv = shap.TreeExplainer(gb).shap_values(Ximp)    # model_output='raw' -> log-odds

msh = pd.Series(np.abs(sv).mean(0), index=[NICE[c] for c in COLS]).sort_values(ascending=False)
print('mean |SHAP| (log-odds):'); print(msh.round(4).to_string())

plt.figure(figsize=(7.2, 3.6))
shap.summary_plot(sv, features=Xb.values, feature_names=[NICE[c] for c in COLS],
                  show=False, plot_size=None, color_bar_label='Feature value')
ax = plt.gca()
ax.set_xlabel('SHAP value (log-odds scale)', fontsize=12)
ax.tick_params(labelsize=11)
for t in ax.get_yticklabels():
    t.set_fontsize(12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_shap.png'), dpi=300, bbox_inches='tight')
print('Figure 3 yazildi')
