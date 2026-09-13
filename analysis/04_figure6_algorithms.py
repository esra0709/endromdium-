# -*- coding: utf-8 -*-
"""
Figure 6 and Supplementary Table S5b — exploratory comparison of nine classifiers
across the four feature sets.

Run after the notebook (01). Reads from and writes to ../outputs/.

    python 04_figure6_algorithms.py
"""
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
os.makedirs(OUT, exist_ok=True)

# The data file is not distributed with this repository; see ../data/README.md
SAV = os.environ.get('EC_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', 'data', 'endometrium.sav'))


import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                    # ekransız ortamda çizim (Colab'da bu satır gereksiz)
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

SEED = 42
CSV_YOL = os.path.join(OUT, 'tableS5b_nine_algorithms.csv')
PNG_YOL = os.path.join(OUT, 'Figure6_algorithms.png')

# ======================================================================
# BÖLÜM A — MODELLERİ ÇALIŞTIR  (CSV zaten varsa çalıştırmanıza gerek yok)
# ======================================================================
CALISTIR = False                          # True yaparsanız dokuz model yeniden eğitilir

if CALISTIR:
    import pyreadstat
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  GradientBoostingClassifier)
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    # ---------------------------------------------------------------- veri
    df, _ = pyreadstat.read_sav(SAV)
    df = df.dropna(subset=['GRADE']).copy()
    df['y'] = (df['GRADE'] == 3).astype(int)          # hedef: G3 = 1, G1-G2 = 0

    # WBC tutarsızlığı düzeltmesi — VIF hesabından ÖNCE yapılmalı
    dsum = df[['nötrofil', 'lenfosit', 'monosit', 'eozinofil']].sum(axis=1)
    bad = (df['wbc'] - dsum).abs() > 2                # diferansiyel toplamıyla uyuşmayan kayıtlar
    df.loc[bad, 'wbc'] = dsum[bad]

    df['LMR'] = df['lenfosit'] / df['monosit']        # lenfosit/monosit oranı
    df = df.rename(columns={
        'nötrofil': 'neu', 'lenfosit': 'lym', 'monosit': 'mon', 'eozinofil': 'eos',
        'platelet': 'plt', 'wbc': 'WBC', 'Dage': 'age', 'ca125': 'CA125',
        'albumin': 'albumin_preop', 'nlr': 'NLR', 'plr': 'PLR', 'pıv': 'PIV',
        'sıı': 'SII', 'sırı': 'SIRI', 'mlr': 'MLR', 'dnlr': 'dNLR', 'elr': 'ELR'})
    y = df['y'].values

    # ---------------------------------------------------------------- öznitelik setleri
    # ER/PR HİÇBİR SETTE YOK: histerektomi materyalinde belirleniyor, preoperatif değil
    FULL = ['eos', 'lym', 'mon', 'neu', 'WBC', 'plt',
            'NLR', 'PLR', 'PIV', 'SII', 'SIRI', 'MLR', 'dNLR', 'ELR']
    SET = {
        'Set A':      ['neu', 'lym', 'mon', 'plt', 'NLR', 'LMR', 'PIV', 'ELR'],  # kolinearite indirgenmiş
        'Set A-full': FULL,                                                       # tam enflamatuar panel
        'Set B':      ['age', 'CA125', 'albumin_preop'],                          # preoperatif klinik
    }
    SET['Set C'] = SET['Set A'] + SET['Set B']                                     # birleşik

    # ---------------------------------------------------------------- algoritmalar ve arama uzayları
    def algos():
        return {
            'LogReg_EN': (LogisticRegression(penalty='elasticnet', solver='saga', max_iter=20000,
                                             class_weight='balanced', random_state=SEED),
                          {'clf__C': np.logspace(-4, 1, 8), 'clf__l1_ratio': [0.0, 0.5, 1.0]}),
            'SVM_RBF': (SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=SEED),
                        {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 0.01, 0.1]}),
            'KNN': (KNeighborsClassifier(),
                    {'clf__n_neighbors': [5, 11, 21, 31]}),
            'RandomForest': (RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1),
                             {'clf__n_estimators': [400], 'clf__max_depth': [3, 5, None],
                              'clf__min_samples_leaf': [3, 8]}),
            'ExtraTrees': (ExtraTreesClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1),
                           {'clf__n_estimators': [400], 'clf__max_depth': [3, 5, None],
                            'clf__min_samples_leaf': [3, 8]}),
            'GradBoost': (GradientBoostingClassifier(random_state=SEED),
                          {'clf__n_estimators': [200], 'clf__learning_rate': [0.03, 0.1],
                           'clf__max_depth': [2, 3]}),
            'XGBoost': (XGBClassifier(eval_metric='logloss', random_state=SEED, n_jobs=-1,
                                      tree_method='hist', verbosity=0),
                        {'clf__n_estimators': [300], 'clf__learning_rate': [0.03, 0.1],
                         'clf__max_depth': [2, 3], 'clf__subsample': [0.8]}),
            'LightGBM': (LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, class_weight='balanced'),
                         {'clf__n_estimators': [300], 'clf__learning_rate': [0.03, 0.1],
                          'clf__num_leaves': [7, 15], 'clf__min_child_samples': [10]}),
            'CatBoost': (CatBoostClassifier(random_seed=SEED, verbose=0, allow_writing_files=False),
                         {'clf__iterations': [300], 'clf__learning_rate': [0.03, 0.1],
                          'clf__depth': [2, 4]}),
        }

    # ---------------------------------------------------------------- iç içe çapraz doğrulama
    def nested(cols, name, est, grid, n_rep=5):
        """Dış döngü performansı, iç döngü hiperparametreyi ölçer.
        ALGORİTMA SEÇİMİ YOK: her algoritma kendi başına değerlendirilir."""
        m = df[cols].notna().all(axis=1).values        # tam-vaka maskesi
        X = df.loc[m, cols].values
        yy = y[m]
        acc = np.zeros(len(yy))                        # tekrarlar boyunca biriken olasılıklar

        for rep in range(n_rep):
            out = StratifiedKFold(5, shuffle=True, random_state=SEED + rep)
            for tr, te in out.split(X, yy):
                pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                                 ('sc', StandardScaler()),
                                 ('clf', est)])
                g = GridSearchCV(pipe, grid, scoring='roc_auc',
                                 cv=StratifiedKFold(3, shuffle=True, random_state=SEED + rep),
                                 n_jobs=-1, refit=True)
                g.fit(X[tr], yy[tr])                   # iç döngü: yalnızca eğitim katmanında
                acc[te] += g.best_estimator_.predict_proba(X[te])[:, 1]

        p = acc / n_rep                                # dış-katman ortalama olasılıkları
        auc = roc_auc_score(yy, p)

        # bootstrap güven aralığı
        rng = np.random.default_rng(SEED)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(yy), len(yy))
            if len(np.unique(yy[i])) < 2:              # tek sınıflı örnek atlanır
                continue
            bs.append(roc_auc_score(yy[i], p[i]))
        lo, hi = np.percentile(bs, [2.5, 97.5])

        return dict(feature_set=name, n=len(yy), events=int(yy.sum()), p=len(cols),
                    auc=round(auc, 4), lo=round(lo, 4), hi=round(hi, 4))

    rows = []
    for aname, (est, grid) in algos().items():
        for sname, cols in SET.items():
            r = nested(cols, sname, est, grid)
            r['algorithm'] = aname
            rows.append(r)
            print('%-13s %-11s AUC %.3f (%.3f-%.3f)' % (aname, sname, r['auc'], r['lo'], r['hi']),
                  flush=True)

    T = pd.DataFrame(rows)[['algorithm', 'feature_set', 'p', 'n', 'events', 'auc', 'lo', 'hi']]
    T.to_csv(CSV_YOL, index=False)
    print('\nTablo S5b yazildi:', CSV_YOL)


# ======================================================================
# BÖLÜM B — ŞEKLİ ÇİZ  (CSV'den okur, model çalıştırmaz)
# ======================================================================
T = pd.read_csv(CSV_YOL)

# Kod adları yerine yayın adları — hakem bu noktayı eleştirdi
NICE = {'LogReg_EN': 'Elastic-net logistic regression',
        'SVM_RBF': 'SVM (RBF kernel)',
        'KNN': 'k-nearest neighbors',
        'RandomForest': 'Random forest',
        'ExtraTrees': 'Extremely randomized trees',
        'GradBoost': 'Gradient boosting',
        'XGBoost': 'XGBoost',
        'LightGBM': 'LightGBM',
        'CatBoost': 'CatBoost'}

SETS = ['Set A', 'Set A-full', 'Set B', 'Set C']
LBL = {'Set A': 'Set A — inflammatory (p = 8)',
       'Set A-full': 'Set A-full — inflammatory (p = 14)',
       'Set B': 'Set B — clinical (p = 3)',
       'Set C': 'Set C — combined (p = 11)'}
CLR = {'Set A': '#1f77b4', 'Set A-full': '#ff7f0e', 'Set B': '#2ca02c', 'Set C': '#d62728'}
OFF = {'Set A': .27, 'Set A-full': .09, 'Set B': -.09, 'Set C': -.27}   # satır içi dikey kaydırma

ORDER = list(dict.fromkeys(T.algorithm))[::-1]     # ilk algoritma en üstte görünsün

fig, ax = plt.subplots(figsize=(10.6, 8.4))        # uzun algoritma adları için geniş tutuldu
for i, a in enumerate(ORDER):
    for s in SETS:
        r = T[(T.algorithm == a) & (T.feature_set == s)].iloc[0]
        yy = i + OFF[s]
        ax.plot([r.lo, r.hi], [yy, yy], color=CLR[s], lw=1.7, solid_capstyle='butt')  # GA çubuğu
        ax.plot([r.auc], [yy], 'o', color=CLR[s], ms=4.5)                             # nokta tahmin

ax.axvline(0.5, color='k', ls='--', lw=1.1)        # şans düzeyi
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels([NICE[a] for a in ORDER], fontsize=11.5)
ax.set_xlabel('AUC-ROC (95% bootstrap confidence interval)', fontsize=12)
ax.set_title('Exploratory comparison: discrimination of nine classifiers across four feature sets',
             fontsize=12.5, pad=12)
ax.set_xlim(0.25, 0.95)
ax.tick_params(labelsize=10.5)
ax.grid(axis='x', color='0.9', lw=.8)
ax.set_axisbelow(True)

h = [plt.Line2D([], [], color=CLR[s], lw=2.4, label=LBL[s]) for s in SETS]
ax.legend(handles=h, fontsize=10, loc='lower right', frameon=True, framealpha=.95)

fig.tight_layout()
fig.savefig(PNG_YOL, dpi=300, bbox_inches='tight')
print('Sekil 6 yazildi:', PNG_YOL)
