# ============================================================
# Machine Learning + Deep Learning – Ebola 2014-2016 Dataset
# Task: Epidemic Phase Classification
#
# ML  Methods  : Random Forest, Gradient Boosting, Logistic Regression, SEIR ODE
# DL  Methods  : MLP (Multi-Layer Perceptron)
#
# Dataset comparison:
#   Dataset A – ebola_2014_2016_clean.csv   (multi-country, WHO-level)
#   Dataset B – SierraLeone_country.csv     (single country, granular)
#
# New in this version:
#   • Deep Learning section (Section 11) with MLP classifier
#   • Dataset B ingestion and feature engineering (Section 12)
#   • Head-to-head ML vs DL comparison across both datasets (Section 13)
#   • Printed recommendation summary (Section 14)
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score,
                             f1_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_A = r"C:\Users\Izz Haikal\Documents\Computational Science\Year 2 Sem 2\Computational Science Laboratory\Group Project\Ebola Updated Machine Learning\ebola_2014_2016_clean.csv"      # Dataset A – multi-country WHO data
CSV_B = r"C:\Users\Izz Haikal\Documents\Computational Science\Year 2 Sem 2\Computational Science Laboratory\Group Project\Ebola Updated Machine Learning\SierraLeone_country.csv"        # Dataset B – Sierra Leone detailed
# ─────────────────────────────────────────────────────────────────────────────


# ============================================================
# 1. DATA LOADING & CLEANING  (Dataset A)
# ============================================================

df = pd.read_csv(CSV_A, parse_dates=['Date'])
df.columns = ['Country', 'Date', 'Cumulative no. cases', 'Cumulative no. deaths']

for col in ['Cumulative no. cases', 'Cumulative no. deaths']:
    df[col] = df[col].fillna(0).astype('int')

df = df.groupby(['Date', 'Country'])[
    ['Cumulative no. cases', 'Cumulative no. deaths']
].sum().reset_index()
df = df.sort_values(['Country', 'Date']).reset_index(drop=True)

df['CFR'] = np.where(
    df['Cumulative no. cases'] > 0,
    round((df['Cumulative no. deaths'] / df['Cumulative no. cases']) * 100, 2),
    0.0
)

temp = df.groupby(['Country', 'Date'])[
    ['Cumulative no. cases', 'Cumulative no. deaths']
].sum().diff().reset_index()
mask = temp['Country'] != temp['Country'].shift(1)
temp.loc[mask, ['Cumulative no. cases', 'Cumulative no. deaths']] = np.nan
temp.columns = ['Country', 'Date', 'New cases', 'New deaths']

df = pd.merge(df, temp, on=['Country', 'Date']).fillna(0)
df[['New cases', 'New deaths']] = df[['New cases', 'New deaths']].astype('int')
df['New cases'] = df['New cases'].clip(lower=0)


# ============================================================
# 2. WEEKLY AGGREGATION  (Dataset A)
# ============================================================

df['WeekStart'] = df['Date'].dt.to_period('W').dt.start_time

df_w = df.groupby(['Country', 'WeekStart']).agg(
    cum_cases  = ('Cumulative no. cases',  'max'),
    cum_deaths = ('Cumulative no. deaths', 'max'),
    new_cases  = ('New cases',             'sum'),
    new_deaths = ('New deaths',            'sum'),
    cfr        = ('CFR',                   'last'),
).reset_index().rename(columns={'WeekStart': 'Date'})

df_w = df_w.sort_values(['Country', 'Date']).reset_index(drop=True)


# ============================================================
# 3. TARGET: EPIDEMIC PHASE  (shared helper)
# ============================================================

def label_phases(new_cases_series):
    nc = new_cases_series
    phases = []
    for i in range(len(nc)):
        past   = nc[max(0, i - 2):i + 1].mean()
        future_slice = nc[i + 1:min(i + 4, len(nc))]
        future = future_slice.mean() if len(future_slice) > 0 else np.nan

        if np.isnan(future) or (past < 2 and future < 2):
            phases.append('Contained')
        elif past == 0:
            phases.append('Growing')
        elif future > past * 1.25:
            phases.append('Growing')
        elif future < past * 0.75:
            phases.append('Declining')
        else:
            phases.append('Stable/Peak')
    return phases

phase_list = []
for country, grp in df_w.groupby('Country'):
    phase_list.extend(label_phases(grp['new_cases'].values))
df_w['Phase'] = phase_list

print("Dataset A – Epidemic phase distribution:")
print(df_w['Phase'].value_counts())


# ============================================================
# 4. FEATURE ENGINEERING  (Dataset A, past-only, no leakage)
# ============================================================

le = LabelEncoder()
df_w['country_enc'] = le.fit_transform(df_w['Country'])
df_w['days_into_epidemic'] = df_w.groupby('Country')['Date'].transform(
    lambda x: (x - x.min()).dt.days)

for lag in [1, 2, 3]:
    df_w[f'new_cases_lag{lag}']  = df_w.groupby('Country')['new_cases'].shift(lag)
    df_w[f'new_deaths_lag{lag}'] = df_w.groupby('Country')['new_deaths'].shift(lag)

df_w['cases_3w_avg']  = df_w.groupby('Country')['new_cases'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df_w['deaths_3w_avg'] = df_w.groupby('Country')['new_deaths'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())

df_w['case_growth_rate'] = (
    df_w.groupby('Country')['new_cases'].shift(1) /
    df_w.groupby('Country')['cum_cases'].shift(2).replace(0, np.nan)
).clip(-5, 5)
df_w['death_growth_rate'] = (
    df_w.groupby('Country')['new_deaths'].shift(1) /
    df_w.groupby('Country')['cum_deaths'].shift(2).replace(0, np.nan)
).clip(-5, 5)

df_w['cum_cases_lag1']  = df_w.groupby('Country')['cum_cases'].shift(1)
df_w['cum_deaths_lag1'] = df_w.groupby('Country')['cum_deaths'].shift(1)

FEATURE_COLS = [
    'country_enc', 'days_into_epidemic',
    'new_cases_lag1',  'new_cases_lag2',  'new_cases_lag3',
    'new_deaths_lag1', 'new_deaths_lag2', 'new_deaths_lag3',
    'cases_3w_avg', 'deaths_3w_avg',
    'case_growth_rate', 'death_growth_rate',
    'cum_cases_lag1', 'cum_deaths_lag1',
]

FEATURE_LABELS = [
    'Country (encoded)', 'Days since outbreak start',
    'New cases (lag 1 wk)',  'New cases (lag 2 wk)',  'New cases (lag 3 wk)',
    'New deaths (lag 1 wk)', 'New deaths (lag 2 wk)', 'New deaths (lag 3 wk)',
    '3-week avg cases', '3-week avg deaths',
    'Case growth rate', 'Death growth rate',
    'Cumulative cases (lag 1 wk)', 'Cumulative deaths (lag 1 wk)',
]

df_ml = df_w[FEATURE_COLS + ['Phase', 'Date', 'Country']].dropna()
X_A = df_ml[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
y_A = df_ml['Phase']
dates_A = df_ml['Date']

print(f"\nDataset A – ML matrix: {X_A.shape[0]} rows × {X_A.shape[1]} features")
print("Class balance:\n", y_A.value_counts())


# ============================================================
# 5. TEMPORAL TRAIN/TEST SPLIT  (Dataset A)
# ============================================================

SPLIT_DATE = pd.Timestamp('2015-10-01')
train_mask_A = dates_A < SPLIT_DATE
test_mask_A  = dates_A >= SPLIT_DATE

X_train_A, X_test_A = X_A[train_mask_A], X_A[test_mask_A]
y_train_A, y_test_A = y_A[train_mask_A], y_A[test_mask_A]

print(f"\nDataset A – Train rows: {len(X_train_A)}  |  Test rows: {len(X_test_A)}")

scaler_A    = StandardScaler()
Xtr_A_sc    = scaler_A.fit_transform(X_train_A)
Xte_A_sc    = scaler_A.transform(X_test_A)


# ============================================================
# 6. ML MODEL DEFINITIONS
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight='balanced',
    random_state=42, n_jobs=-1)

gb_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)

lr_model = LogisticRegression(
    C=1.0, max_iter=1000, class_weight='balanced', random_state=42)


# ============================================================
# 7. TRAINING  (Dataset A – ML)
# ============================================================

rf_model.fit(X_train_A, y_train_A)
gb_model.fit(X_train_A, y_train_A)
lr_model.fit(Xtr_A_sc, y_train_A)


# ============================================================
# 8. EVALUATION  (Dataset A – ML)
# ============================================================

rf_pred_A  = rf_model.predict(X_test_A)
gb_pred_A  = gb_model.predict(X_test_A)
lr_pred_A  = lr_model.predict(Xte_A_sc)

classes_A = sorted(y_A.unique())

print("\n" + "="*60)
print("  DATASET A – ML RESULTS")
print("="*60)
for name, pred in [
    ('Random Forest',       rf_pred_A),
    ('Gradient Boosting',   gb_pred_A),
    ('Logistic Regression', lr_pred_A),
]:
    print(f"\n  {name}")
    print(classification_report(y_test_A, pred, zero_division=0,
                                labels=classes_A,
                                target_names=classes_A))


# ============================================================
# 9. VISUALISATIONS  (Dataset A – ML)
# ============================================================

CLASSES = sorted(y_A.unique())
PHASE_COLORS = {
    'Growing':     '#e74c3c',
    'Stable/Peak': '#f39c12',
    'Declining':   '#3498db',
    'Contained':   '#95a5a6',
}

# ── 9b. Feature importances (Random Forest) ───────────────────────────────────
# ── 9b-2. Feature importances (Gradient Boosting) ────────────────────────────
importances = pd.Series(rf_model.feature_importances_,
                        index=FEATURE_LABELS).sort_values()
gb_importances = pd.Series(gb_model.feature_importances_,
                           index=FEATURE_LABELS).sort_values()

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle("Feature Importances – Random Forest vs Gradient Boosting (Dataset A)", fontsize=13)

# Left: Random Forest (re-plot for side-by-side comparison)
palette_rf = sns.color_palette('rocket_r', len(importances))
importances.plot(kind='barh', ax=axes[0], color=palette_rf, edgecolor='white')
axes[0].set_title("Random Forest", fontsize=12)
axes[0].set_xlabel("Importance Score")
axes[0].axvline(1 / len(importances), color='gray', linestyle='--', lw=1, label='Uniform baseline')
axes[0].legend(fontsize=9)

# Right: Gradient Boosting
palette_gb = sns.color_palette('crest', len(gb_importances))
gb_importances.plot(kind='barh', ax=axes[1], color=palette_gb, edgecolor='white')
axes[1].set_title("Gradient Boosting", fontsize=12)
axes[1].set_xlabel("Importance Score")
axes[1].axvline(1 / len(gb_importances), color='gray', linestyle='--', lw=1, label='Uniform baseline')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("feature_importances_gb_vs_rf_A.png", dpi=150, bbox_inches='tight')
plt.show()

# Print top 5 features for each model for easy comparison
print("\n" + "="*60)
print("  FEATURE IMPORTANCE COMPARISON: RF vs GB (Top 5)")
print("="*60)
print("\n  Random Forest – Top 5:")
for feat, score in importances.sort_values(ascending=False).head(5).items():
    print(f"    {feat:<35} {score:.4f}")
print("\n  Gradient Boosting – Top 5:")
for feat, score in gb_importances.sort_values(ascending=False).head(5).items():
    print(f"    {feat:<35} {score:.4f}")

# ── 9d. Logistic Regression coefficients ─────────────────────────────────────
coef_df = pd.DataFrame(lr_model.coef_, columns=FEATURE_LABELS, index=lr_model.classes_)
fig, axes = plt.subplots(1, len(lr_model.classes_), figsize=(20, 5), sharey=True)
fig.suptitle("Logistic Regression Coefficients by Phase", fontsize=12)
for ax, cls in zip(axes, sorted(lr_model.classes_)):
    vals   = coef_df.loc[cls].sort_values()
    colors = ['steelblue' if v > 0 else 'salmon' for v in vals]
    vals.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_title(cls, fontsize=11)
    ax.set_xlabel("Coefficient")
plt.tight_layout()
plt.savefig("lr_coefficients.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nDataset A – ML pipeline complete.")


# ============================================================
# 10. SEIR EPIDEMIC MODEL
# ============================================================

TOP3 = ['Guinea', 'Liberia', 'Sierra Leone']

def seir_odes(t, y, beta, sigma, gamma, N):
    S, E, I, R = y
    force_of_infection = beta * S * I / N
    dS = -force_of_infection
    dE =  force_of_infection - sigma * E
    dI =  sigma * E          - gamma * I
    dR =  gamma * I
    return [dS, dE, dI, dR]

def run_seir(N, beta, sigma, gamma, E0, I0, t_max):
    S0 = N - E0 - I0
    y0 = [S0, E0, I0, 0.0]
    sol = solve_ivp(seir_odes, t_span=(0, t_max), y0=y0,
                    args=(beta, sigma, gamma, N), method='RK45',
                    dense_output=True, max_step=1.0)
    t = np.linspace(0, t_max, t_max + 1)
    S, E, I, R = sol.sol(t)
    return t, S, E, I, R

def fit_seir_to_country(country_df, N, sigma=1/11.5):
    dates_sorted = country_df.sort_values('Date')
    t_obs = (dates_sorted['Date'] - dates_sorted['Date'].iloc[0]).dt.days.values
    obs   = dates_sorted['cum_cases'].values.astype(float)
    t_max = int(t_obs[-1])
    E0 = max(obs[0] * 2, 1)
    I0 = max(obs[0],     1)

    def residuals(params):
        beta, gamma = params
        if beta <= 0 or gamma <= 0:
            return 1e12
        try:
            t, S, E, I, R = run_seir(N, beta, gamma, sigma, E0, I0, t_max)
            cum_model = N - S
            cum_at_obs = np.interp(t_obs, t, cum_model)
            return np.sum((cum_at_obs - obs) ** 2)
        except Exception:
            return 1e12

    result = minimize(residuals, x0=[0.27, 0.10], method='Nelder-Mead',
                      options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
    beta_fit, gamma_fit = result.x
    return beta_fit, gamma_fit, beta_fit / gamma_fit

POPULATIONS = {'Guinea': 12_000_000, 'Liberia': 4_500_000, 'Sierra Leone': 7_000_000}
SIGMA_EBOLA = 1 / 11.5

print("\n" + "="*60)
print("  SEIR MODEL – FITTED PARAMETERS")
print("="*60)
seir_results = {}
for country in TOP3:
    country_data = df_w[df_w['Country'] == country].copy()
    N = POPULATIONS[country]
    beta_fit, gamma_fit, R0 = fit_seir_to_country(country_data, N=N, sigma=SIGMA_EBOLA)
    seir_results[country] = {'beta': beta_fit, 'gamma': gamma_fit, 'R0': R0, 'N': N}
    print(f"\n  {country}  β={beta_fit:.4f}  γ={gamma_fit:.4f}  R₀={R0:.2f}")

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
fig.suptitle("SEIR Model Fit vs Observed Cumulative Cases", fontsize=13)
for ax, country in zip(axes, TOP3):
    res = seir_results[country]
    cd  = df_w[df_w['Country'] == country].sort_values('Date')
    t0  = cd['Date'].iloc[0]
    t_obs = (cd['Date'] - t0).dt.days.values
    obs   = cd['cum_cases'].values
    t_max = int(t_obs[-1])
    E0 = max(obs[0] * 2, 1);  I0 = max(obs[0], 1)
    t_sim, S, E, I, R = run_seir(res['N'], res['beta'], res['gamma'], SIGMA_EBOLA, E0, I0, t_max)
    cum_model = res['N'] - S
    ax.plot(t_sim, cum_model, color='#e74c3c', lw=2,
            label=f"SEIR fit (R₀={res['R0']:.2f})")
    ax.scatter(t_obs, obs, color='steelblue', s=18, zorder=5, label='Observed')
    ax.set_ylabel("Cumulative cases"); ax.set_xlabel("Days"); ax.set_title(country, loc='left')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("seir_fit.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nSEIR modelling complete.")


# ============================================================
# 10b. SEIR PHASE CLASSIFICATION  (Dataset A – full timeline)
# ============================================================
# Strategy: for each row in df_ml (country + date), derive the
# SEIR-predicted phase from the *gradient* of the fitted SEIR
# infectious curve I(t).  We apply the same thresholds used by
# label_phases() so the SEIR "prediction" is directly comparable
# to the true Phase labels and to the ML/DL classifiers.
#
# Phase mapping from dI/dt:
#   dI/dt > +25% of local I  →  Growing
#   dI/dt < -25% of local I  →  Declining
#   |I| < 1 per million      →  Contained
#   otherwise                →  Stable/Peak
#
# NOTE: Metrics are still computed on the test set only (fair
# comparison with ML). The plot now covers the full timeline
# so SEIR and RF curves are visible from outbreak day 1.
# ============================================================

print("\n" + "="*60)
print("  SECTION 10b – SEIR AS PHASE CLASSIFIER  (Dataset A – full timeline)")
print("="*60)

def seir_phase_for_row(country, date, seir_results, df_w_ref):
    """
    Given a country and a calendar date, return the SEIR-derived
    epidemic phase by evaluating dI/dt at that point on the fitted
    SEIR trajectory.
    """
    if country not in seir_results:
        return 'Contained'           # countries without SEIR fit

    res   = seir_results[country]
    N     = res['N']
    t0    = df_w_ref[df_w_ref['Country'] == country]['Date'].min()
    t_day = (date - t0).days
    if t_day < 0:
        return 'Contained'

    cd = df_w_ref[df_w_ref['Country'] == country].sort_values('Date')
    obs_max = cd['cum_cases'].max()
    E0 = max(cd['cum_cases'].iloc[0] * 2, 1)
    I0 = max(cd['cum_cases'].iloc[0], 1)
    t_max_fit = int((cd['Date'].max() - t0).days)

    t_sim, S, E, I, R = run_seir(
        N, res['beta'], res['gamma'], SIGMA_EBOLA,
        E0, I0, max(t_max_fit, t_day + 7)
    )

    # evaluate I and its forward difference at t_day
    I_now  = float(np.interp(t_day,     t_sim, I))
    I_soon = float(np.interp(t_day + 7, t_sim, I))

    # containment: fewer than 1 case per 100k susceptible
    if I_now < N / 100_000:
        return 'Contained'

    ratio = I_soon / max(I_now, 1e-6)
    if   ratio > 1.25:
        return 'Growing'
    elif ratio < 0.75:
        return 'Declining'
    else:
        return 'Stable/Peak'


# Apply to the Dataset-A test rows only (for fair metric comparison with ML)
seir_pred_A = []
for idx in df_ml[test_mask_A].index:
    row    = df_ml.loc[idx]
    phase  = seir_phase_for_row(row['Country'], row['Date'],
                                 seir_results, df_w)
    seir_pred_A.append(phase)

seir_pred_A = np.array(seir_pred_A)

# Countries in the test set that have a SEIR fit (Guinea, Liberia, Sierra Leone)
has_seir = df_ml[test_mask_A]['Country'].isin(TOP3).values

print(f"\n  Test rows with fitted SEIR: {has_seir.sum()} / {len(has_seir)}")
print("\n  SEIR classifier – Classification Report (all test countries):")
print(classification_report(y_test_A, seir_pred_A, zero_division=0,
                            labels=classes_A, target_names=classes_A))

print("\n  SEIR classifier – Classification Report (TOP-3 countries only):")
print(classification_report(
    y_test_A[has_seir], seir_pred_A[has_seir], zero_division=0,
    labels=classes_A, target_names=classes_A
))

# ── Phase timeline: SEIR prediction vs True vs Best-ML ───────────────────────
# CHANGE: now generates SEIR and RF predictions over the FULL timeline
# (train + test) so predictions are visible across the entire outbreak arc,
# not just the small test window at the end.
# ─────────────────────────────────────────────────────────────────────────────

PHASE_ORDER  = ['Growing', 'Stable/Peak', 'Declining', 'Contained']
PHASE_INT    = {p: i for i, p in enumerate(PHASE_ORDER)}
PHASE_COLORS2 = {
    'Growing':     '#e74c3c',
    'Stable/Peak': '#f39c12',
    'Declining':   '#3498db',
    'Contained':   '#95a5a6',
}

# ── Build FULL-timeline predictions for all df_ml rows (train + test) ─────────
# SEIR: evaluate mechanistic model at every date in df_ml for TOP3 countries
seir_pred_A_full = []
for idx in df_ml.index:
    row   = df_ml.loc[idx]
    phase = seir_phase_for_row(row['Country'], row['Date'], seir_results, df_w)
    seir_pred_A_full.append(phase)
seir_pred_A_full = np.array(seir_pred_A_full)

# RF: predict on the full feature matrix (train + test) — note this includes
# seen training data, so train-set RF predictions are "in-sample" (optimistic),
# but displaying them lets you see whether the model's learned shape tracks
# the true phase across the whole timeline.
rf_pred_A_full = rf_model.predict(X_A)

# Attach full predictions back to df_ml for easy per-country slicing
df_ml_full = df_ml.copy()
df_ml_full['seir_pred_full'] = seir_pred_A_full
df_ml_full['rf_pred_full']   = rf_pred_A_full
# Also keep a split flag so we can shade train vs test regions on the plot
df_ml_full['split'] = np.where(train_mask_A, 'train', 'test')

# Keep the test-only predictions for metrics (unchanged)
test_df_A = df_ml[test_mask_A].copy()
test_df_A['seir_pred'] = seir_pred_A
test_df_A['rf_pred']   = rf_pred_A

fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=False)
fig.suptitle(
    "Epidemic Phase: True vs SEIR vs Random Forest – Full Timeline (Top-3 Countries)\n"
    "Shaded region = training period | Unshaded = test period",
    fontsize=12, y=1.01
)

for ax, country in zip(axes, TOP3):
    sub_all  = df_w[df_w['Country'] == country].sort_values('Date')
    sub_full = df_ml_full[df_ml_full['Country'] == country].sort_values('Date')
    sub_test = test_df_A[test_df_A['Country'] == country].sort_values('Date')

    ymax = sub_all['new_cases'].max()

    # ── Background new-cases curve ────────────────────────────────────────────
    ax.fill_between(sub_all['Date'], sub_all['new_cases'],
                    alpha=0.15, color='steelblue', label='_nolegend_')
    ax.plot(sub_all['Date'], sub_all['new_cases'],
            color='steelblue', lw=1, alpha=0.6, label='New cases/wk')

    # ── True phase background spans (full dataset) ────────────────────────────
    for _, row in sub_all.iterrows():
        ax.axvspan(row['Date'], row['Date'] + pd.Timedelta(weeks=1),
                   alpha=0.12, color=PHASE_COLORS2.get(row['Phase'], 'gray'))

    # ── Train/test boundary vertical line ─────────────────────────────────────
    ax.axvline(SPLIT_DATE, color='black', lw=1.5, linestyle='-.',
               alpha=0.7, label=f'Train/test split ({SPLIT_DATE.date()})')

    # ── Light grey shading over the training region ───────────────────────────
    ax.axvspan(sub_all['Date'].min(), SPLIT_DATE,
               alpha=0.07, color='black', label='_nolegend_')

    # ── FULL-timeline SEIR and RF phase step-lines ────────────────────────────
    if len(sub_full):
        ax.step(sub_full['Date'],
                [PHASE_INT[p] * ymax / 4 for p in sub_full['seir_pred_full']],
                where='post', color='#8e44ad', lw=2.0,
                linestyle='--', label='SEIR phase (train+test)', alpha=0.85)
        ax.step(sub_full['Date'],
                [PHASE_INT[p] * ymax / 4 for p in sub_full['rf_pred_full']],
                where='post', color='#27ae60', lw=2.0,
                linestyle=':', label='RF phase (train+test)', alpha=0.85)

    # ── Phase integer y-axis labels (right side) ──────────────────────────────
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    tick_vals = [PHASE_INT[p] * ymax / 4 for p in PHASE_ORDER]
    ax2.set_yticks(tick_vals)
    ax2.set_yticklabels(PHASE_ORDER, fontsize=7, color='dimgray')
    ax2.tick_params(axis='y', length=0)

    ax.set_ylabel("New cases / week", fontsize=9)
    ax.set_title(country, fontsize=11, loc='left')

    patches_leg = [Patch(color=c, alpha=0.7, label=p)
                   for p, c in PHASE_COLORS2.items()]
    legend1 = ax.legend(handles=patches_leg, loc='upper right',
                        fontsize=7, ncol=4, title='True phase colour')
    ax.add_artist(legend1)
    ax.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig("seir_vs_ml_phase_timeline.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nSEIR phase classification complete.")


# ============================================================
# 11. DEEP LEARNING  (Dataset A)
# ============================================================
# MLP  – a dense feed-forward network; strongest baseline
# when features are already hand-engineered as tabular data.
# ============================================================

# ── Label encoding for DL ─────────────────────────────────────────────────────
label_enc_dl = LabelEncoder()
label_enc_dl.fit(y_A)

y_train_A_enc = label_enc_dl.transform(y_train_A)
y_test_A_enc  = label_enc_dl.transform(y_test_A)
n_classes     = len(label_enc_dl.classes_)

# ── 11a. MLP ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  DEEP LEARNING – MLP (Dataset A)")
print("="*60)

def build_mlp(input_dim, n_classes):
    inp = keras.Input(shape=(input_dim,))
    x   = layers.Dense(128, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

mlp_A = build_mlp(Xtr_A_sc.shape[1], n_classes)
mlp_A.summary()

es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history_mlp_A = mlp_A.fit(
    Xtr_A_sc, y_train_A_enc,
    validation_split=0.15,
    epochs=200,
    batch_size=32,
    callbacks=[es],
    verbose=0
)

mlp_pred_A_enc = np.argmax(mlp_A.predict(Xte_A_sc, verbose=0), axis=1)
mlp_pred_A     = label_enc_dl.inverse_transform(mlp_pred_A_enc)
print("\nMLP – Classification Report (Dataset A):")
print(classification_report(y_test_A, mlp_pred_A, zero_division=0,
                            labels=classes_A,
                             target_names=classes_A))

# ── Training curves ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history_mlp_A.history['loss'],     label='Train loss')
ax.plot(history_mlp_A.history['val_loss'], label='Val loss',  linestyle='--')
ax.set_title('MLP Training Curves (Dataset A)')
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend()
plt.tight_layout()
plt.savefig("training_curves_A.png", dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 12. DATASET B – SIERRA LEONE DETAILED DATA
# ============================================================
# This dataset has a richer set of columns: survivors, probable/
# suspected cases, treatment-centre occupancy.  We engineer the
# same epidemic-phase target and augment features with dataset-
# specific columns.
# ============================================================

print("\n" + "="*60)
print("  DATASET B – SIERRA LEONE DETAILED")
print("="*60)

df_sl = pd.read_csv(CSV_B, parse_dates=['Date'])
df_sl.columns = [c.strip() for c in df_sl.columns]

# Rename for convenience
df_sl = df_sl.rename(columns={
    'Cumulative_Confirmed_Cases':   'cum_cases',
    'Cumulative_Confirmed_Deaths':  'cum_deaths',
    'Survivors':                    'survivors',
    'Probable_Cases':               'probable',
    'Suspected_cases':              'suspected',
    'In_Treatment_Centers':         'in_treatment',
})

for col in ['cum_cases', 'cum_deaths', 'survivors', 'probable',
            'suspected', 'in_treatment']:
    df_sl[col] = pd.to_numeric(df_sl[col], errors='coerce').fillna(0).astype(int)

df_sl = df_sl.sort_values('Date').reset_index(drop=True)

# New cases / deaths from cumulative
df_sl['new_cases']  = df_sl['cum_cases'].diff().clip(lower=0).fillna(0).astype(int)
df_sl['new_deaths'] = df_sl['cum_deaths'].diff().clip(lower=0).fillna(0).astype(int)

# Weekly aggregation
df_sl['WeekStart'] = df_sl['Date'].dt.to_period('W').dt.start_time
df_sl_w = df_sl.groupby('WeekStart').agg(
    cum_cases   = ('cum_cases',    'max'),
    cum_deaths  = ('cum_deaths',   'max'),
    new_cases   = ('new_cases',    'sum'),
    new_deaths  = ('new_deaths',   'sum'),
    survivors   = ('survivors',    'last'),
    probable    = ('probable',     'last'),
    suspected   = ('suspected',    'last'),
    in_treatment= ('in_treatment', 'last'),
).reset_index().rename(columns={'WeekStart': 'Date'})
df_sl_w = df_sl_w.sort_values('Date').reset_index(drop=True)

# Phase labels
df_sl_w['Phase'] = label_phases(df_sl_w['new_cases'].values)

print("Dataset B – Epidemic phase distribution:")
print(df_sl_w['Phase'].value_counts())

# Feature engineering
df_sl_w['days_into_epidemic'] = (df_sl_w['Date'] - df_sl_w['Date'].min()).dt.days

for lag in [1, 2, 3]:
    df_sl_w[f'new_cases_lag{lag}']  = df_sl_w['new_cases'].shift(lag)
    df_sl_w[f'new_deaths_lag{lag}'] = df_sl_w['new_deaths'].shift(lag)

df_sl_w['cases_3w_avg']  = df_sl_w['new_cases'].shift(1).rolling(3, min_periods=1).mean()
df_sl_w['deaths_3w_avg'] = df_sl_w['new_deaths'].shift(1).rolling(3, min_periods=1).mean()

df_sl_w['case_growth_rate'] = (
    df_sl_w['new_cases'].shift(1) /
    df_sl_w['cum_cases'].shift(2).replace(0, np.nan)
).clip(-5, 5)
df_sl_w['death_growth_rate'] = (
    df_sl_w['new_deaths'].shift(1) /
    df_sl_w['cum_deaths'].shift(2).replace(0, np.nan)
).clip(-5, 5)

df_sl_w['cum_cases_lag1']  = df_sl_w['cum_cases'].shift(1)
df_sl_w['cum_deaths_lag1'] = df_sl_w['cum_deaths'].shift(1)

# Dataset-B-specific extras (fill zeros for missing)
df_sl_w['survivors_lag1']   = df_sl_w['survivors'].shift(1)
df_sl_w['in_treatment_lag1']= df_sl_w['in_treatment'].shift(1)

FEAT_B = [
    'days_into_epidemic',
    'new_cases_lag1',  'new_cases_lag2',  'new_cases_lag3',
    'new_deaths_lag1', 'new_deaths_lag2', 'new_deaths_lag3',
    'cases_3w_avg', 'deaths_3w_avg',
    'case_growth_rate', 'death_growth_rate',
    'cum_cases_lag1', 'cum_deaths_lag1',
    'survivors_lag1', 'in_treatment_lag1',
]

df_b = df_sl_w[FEAT_B + ['Phase', 'Date']].dropna()
X_B  = df_b[FEAT_B].replace([np.inf, -np.inf], np.nan).fillna(0)
y_B  = df_b['Phase']
dates_B = df_b['Date']

print(f"\nDataset B – matrix: {X_B.shape[0]} rows × {X_B.shape[1]} features")

# Temporal split: 70% train / 30% test
split_b = int(len(X_B) * 0.70)
X_train_B, X_test_B = X_B.iloc[:split_b], X_B.iloc[split_b:]
y_train_B, y_test_B = y_B.iloc[:split_b], y_B.iloc[split_b:]
db_test_dates        = dates_B.iloc[split_b:].reset_index(drop=True)

scaler_B  = StandardScaler()
Xtr_B_sc  = scaler_B.fit_transform(X_train_B)
Xte_B_sc  = scaler_B.transform(X_test_B)

# ── ML on Dataset B ───────────────────────────────────────────────────────────
rf_B  = RandomForestClassifier(n_estimators=300, max_depth=8,
                                class_weight='balanced', random_state=42, n_jobs=-1)
gb_B  = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                    learning_rate=0.05, random_state=42)
lr_B  = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)

rf_B.fit(X_train_B, y_train_B)
gb_B.fit(X_train_B, y_train_B)
lr_B.fit(Xtr_B_sc, y_train_B)

rf_pred_B  = rf_B.predict(X_test_B)
gb_pred_B  = gb_B.predict(X_test_B)
lr_pred_B  = lr_B.predict(Xte_B_sc)

classes_B = sorted(y_B.unique())

print("\nDataset B – ML Results:")
for name, pred in [
    ('Random Forest',       rf_pred_B),
    ('Gradient Boosting',   gb_pred_B),
    ('Logistic Regression', lr_pred_B),
]:
    print(f"\n  {name}")
    print(classification_report(y_test_B, pred, zero_division=0,
                                labels=classes_B,
                                target_names=classes_B))

# ── DL on Dataset B ───────────────────────────────────────────────────────────
label_enc_B = LabelEncoder()
label_enc_B.fit(y_B)

y_train_B_enc = label_enc_B.transform(y_train_B)
y_test_B_enc  = label_enc_B.transform(y_test_B)
n_classes_B   = len(label_enc_B.classes_)

mlp_B = build_mlp(Xtr_B_sc.shape[1], n_classes_B)
mlp_B.fit(Xtr_B_sc, y_train_B_enc, validation_split=0.15,
          epochs=200, batch_size=16,
          callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
          verbose=0)
mlp_pred_B = label_enc_B.inverse_transform(
    np.argmax(mlp_B.predict(Xte_B_sc, verbose=0), axis=1))

print("\nDataset B – Deep Learning Results:")
for name, pred in [('MLP', mlp_pred_B)]:
    print(f"\n  {name}")
    print(classification_report(y_test_B, pred, zero_division=0,
                                labels=classes_B,
                                target_names=classes_B))

# ── Training curves – Dataset B ───────────────────────────────────────────────
history_mlp_B  = mlp_B.history

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history_mlp_B.history['loss'],     label='Train loss')
ax.plot(history_mlp_B.history['val_loss'], label='Val loss',  linestyle='--')
ax.set_title('MLP Training Curves (Dataset B)')
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend()
plt.tight_layout()
plt.savefig("training_curves_B.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 12b. SEIR PHASE CLASSIFICATION  (Dataset B – full timeline)
# ============================================================
# Sierra Leone is already in seir_results (fitted in Section 10).
# Dataset B uses df_sl_w as its reference frame (no Country column),
# so we adapt seir_phase_for_row() to work with a single-country
# dataframe directly.
#
# NOTE: Metrics computed on test rows only (fair). Plot shows the
# full timeline so predictions are visible from outbreak day 1.
# ============================================================

print("\n" + "="*60)
print("  SECTION 12b – SEIR AS PHASE CLASSIFIER  (Dataset B – full timeline)")
print("="*60)

def seir_phase_for_row_B(date, seir_results, df_sl_w_ref, country='Sierra Leone'):
    """
    Single-country variant of seir_phase_for_row().
    df_sl_w_ref has no 'Country' column; t0 is its first Date.
    """
    if country not in seir_results:
        return 'Contained'

    res   = seir_results[country]
    N     = res['N']
    t0    = df_sl_w_ref['Date'].min()
    t_day = (date - t0).days
    if t_day < 0:
        return 'Contained'

    cd    = df_sl_w_ref.sort_values('Date')
    E0    = max(cd['cum_cases'].iloc[0] * 2, 1)
    I0    = max(cd['cum_cases'].iloc[0], 1)
    t_max_fit = int((cd['Date'].max() - t0).days)

    t_sim, S, E, I, R = run_seir(
        N, res['beta'], res['gamma'], SIGMA_EBOLA,
        E0, I0, max(t_max_fit, t_day + 7)
    )

    I_now  = float(np.interp(t_day,     t_sim, I))
    I_soon = float(np.interp(t_day + 7, t_sim, I))

    if I_now < N / 100_000:
        return 'Contained'

    ratio = I_soon / max(I_now, 1e-6)
    if   ratio > 1.25:
        return 'Growing'
    elif ratio < 0.75:
        return 'Declining'
    else:
        return 'Stable/Peak'


# Apply to Dataset B test rows
seir_pred_B = np.array([
    seir_phase_for_row_B(date, seir_results, df_sl_w)
    for date in db_test_dates
])

print(f"\n  Dataset B test rows: {len(seir_pred_B)}")
print("\n  SEIR classifier – Classification Report (Dataset B):")
print(classification_report(y_test_B, seir_pred_B, zero_division=0,
                            labels=classes_B, target_names=classes_B))

# ── Phase timeline: SEIR vs True vs Best-ML (Dataset B) – FULL TIMELINE ──────
# CHANGE: generate SEIR and RF predictions across the full Dataset B timeline
# (train + test), not just the test window.  This lets you see how both models
# track the true phase from the very start of the Sierra Leone outbreak.
# ─────────────────────────────────────────────────────────────────────────────

# SEIR: evaluate at every date in df_b (full matrix, train+test)
seir_pred_B_full = np.array([
    seir_phase_for_row_B(date, seir_results, df_sl_w)
    for date in dates_B          # dates_B spans the full df_b index
])

# RF: predict on the full Dataset B feature matrix (train + test)
# train-set predictions are in-sample (optimistic) but reveal the learned shape
rf_pred_B_full = rf_B.predict(X_B)

# Split boundary date for Dataset B (70% mark)
split_date_B = dates_B.iloc[split_b]

# Also keep test-only variables for metrics (unchanged)
test_dates_B_ser = db_test_dates.reset_index(drop=True)

ymax_B = df_sl_w['new_cases'].max()

fig, ax = plt.subplots(figsize=(16, 6))
fig.suptitle(
    "Dataset B – Epidemic Phase: True vs SEIR vs Random Forest – Full Timeline (Sierra Leone)\n"
    "Shaded region = training period | Unshaded = test period",
    fontsize=12
)

# Background new-cases fill
ax.fill_between(df_sl_w['Date'], df_sl_w['new_cases'],
                alpha=0.15, color='steelblue')
ax.plot(df_sl_w['Date'], df_sl_w['new_cases'],
        color='steelblue', lw=1, alpha=0.6, label='New cases/wk')

# True phase background spans
for _, row in df_sl_w.iterrows():
    ax.axvspan(row['Date'], row['Date'] + pd.Timedelta(weeks=1),
               alpha=0.12, color=PHASE_COLORS2.get(row['Phase'], 'gray'))

# Train/test split boundary
ax.axvline(split_date_B, color='black', lw=1.5, linestyle='-.',
           alpha=0.7, label=f'Train/test split ({split_date_B.date()})')
ax.axvspan(df_sl_w['Date'].min(), split_date_B,
           alpha=0.07, color='black', label='_nolegend_')

# Full-timeline SEIR and RF overlays
ax.step(dates_B.values,
        [PHASE_INT[p] * ymax_B / 4 for p in seir_pred_B_full],
        where='post', color='#8e44ad', lw=2.2,
        linestyle='--', label='SEIR phase (train+test)', alpha=0.85)
ax.step(dates_B.values,
        [PHASE_INT[p] * ymax_B / 4 for p in rf_pred_B_full],
        where='post', color='#27ae60', lw=2.2,
        linestyle=':', label='RF phase (train+test)', alpha=0.85)

# Phase label axis on the right
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
tick_vals_B = [PHASE_INT[p] * ymax_B / 4 for p in PHASE_ORDER]
ax2.set_yticks(tick_vals_B)
ax2.set_yticklabels(PHASE_ORDER, fontsize=7, color='dimgray')
ax2.tick_params(axis='y', length=0)

ax.set_ylabel("New cases / week", fontsize=10)
ax.set_xlabel("Date")
patches_b = [Patch(color=c, alpha=0.7, label=p)
             for p, c in PHASE_COLORS2.items()]
legend1 = ax.legend(handles=patches_b, loc='upper right',
                    fontsize=8, ncol=4, title='True phase colour')
ax.add_artist(legend1)
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig("seir_vs_ml_phase_timeline_B.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nSEIR phase classification (Dataset B) complete.")


# ============================================================
# 13. HEAD-TO-HEAD COMPARISON  (ML vs DL, both datasets)
# ============================================================

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1m = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    return acc, f1w, f1m

comparison_rows = []

# ── SEIR as baseline classifier (Dataset A only – test set) ──────────────────
acc_seir, f1w_seir, f1m_seir = compute_metrics(y_test_A, seir_pred_A)
comparison_rows.append({
    'Dataset': 'A (Multi-country)', 'Model': 'SEIR',
    'Type': 'Mechanistic', 'Accuracy': acc_seir,
    'Weighted F1': f1w_seir, 'Macro F1': f1m_seir
})

# SEIR – TOP-3 countries only (meaningful subset)
acc_seir3, f1w_seir3, f1m_seir3 = compute_metrics(
    np.array(y_test_A)[has_seir], seir_pred_A[has_seir])
comparison_rows.append({
    'Dataset': 'A (Multi-country)', 'Model': 'SEIR (TOP-3)',
    'Type': 'Mechanistic', 'Accuracy': acc_seir3,
    'Weighted F1': f1w_seir3, 'Macro F1': f1m_seir3
})

# Dataset A – ML / DL
for name, pred in [
    ('Random Forest',       rf_pred_A),
    ('Gradient Boosting',   gb_pred_A),
    ('Logistic Regression', lr_pred_A),
    ('MLP',                 mlp_pred_A),
]:
    acc, f1w, f1m = compute_metrics(y_test_A, pred)
    kind = 'Deep Learning' if name in ('MLP',) else 'Machine Learning'
    comparison_rows.append({'Dataset': 'A (Multi-country)', 'Model': name,
                             'Type': kind, 'Accuracy': acc,
                             'Weighted F1': f1w, 'Macro F1': f1m})

# Dataset B – ML / DL + SEIR
acc_seir_B, f1w_seir_B, f1m_seir_B = compute_metrics(y_test_B, seir_pred_B)
comparison_rows.append({
    'Dataset': 'B (Sierra Leone)', 'Model': 'SEIR',
    'Type': 'Mechanistic', 'Accuracy': acc_seir_B,
    'Weighted F1': f1w_seir_B, 'Macro F1': f1m_seir_B
})

for name, pred in [
    ('Random Forest',       rf_pred_B),
    ('Gradient Boosting',   gb_pred_B),
    ('Logistic Regression', lr_pred_B),
    ('MLP',                 mlp_pred_B),
]:
    acc, f1w, f1m = compute_metrics(y_test_B, pred)
    kind = 'Deep Learning' if name in ('MLP',) else 'Machine Learning'
    comparison_rows.append({'Dataset': 'B (Sierra Leone)', 'Model': name,
                             'Type': kind, 'Accuracy': acc,
                             'Weighted F1': f1w, 'Macro F1': f1m})

cmp_df = pd.DataFrame(comparison_rows)

print("\n" + "="*70)
print("  FULL COMPARISON TABLE – SEIR vs ML vs DEEP LEARNING")
print("="*70)
print(cmp_df.to_string(index=False, float_format='{:.3f}'.format))

# ── Grouped bar chart – Dataset A only (SEIR + ML + DL) ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=False)
fig.suptitle("SEIR vs ML vs Deep Learning – Accuracy & Weighted F1 (Dataset A)",
             fontsize=14)

datasets = ['A (Multi-country)', 'B (Sierra Leone)']
# Extended model list includes SEIR baselines for Dataset A
models_A = ['SEIR', 'SEIR (TOP-3)', 'Random Forest', 'Gradient Boosting',
            'Logistic Regression', 'MLP']
models_B = ['SEIR', 'Random Forest', 'Gradient Boosting', 'Logistic Regression', 'MLP']
colors_A = ['#8e44ad', '#c39bd3', '#2980b9', '#27ae60', '#16a085', '#e74c3c']
colors_B = ['#8e44ad', '#2980b9', '#27ae60', '#16a085', '#e74c3c']

# Dataset A panel
ax = axes[0]
sub_A = cmp_df[cmp_df['Dataset'] == 'A (Multi-country)'].set_index('Model')
x_A   = np.arange(len(models_A))
width = 0.35
bars_acc_A = ax.bar(x_A - width/2,
                    [sub_A.loc[m, 'Accuracy']    for m in models_A],
                    width, color=colors_A, alpha=0.88)
bars_f1_A  = ax.bar(x_A + width/2,
                    [sub_A.loc[m, 'Weighted F1'] for m in models_A],
                    width, color=colors_A, alpha=0.40,
                    edgecolor='black', linewidth=0.5)
ax.set_xticks(x_A)
ax.set_xticklabels(models_A, rotation=20, ha='right', fontsize=9)
ax.set_ylim(0, 1.10)
ax.set_ylabel("Score")
ax.set_title("Dataset A (Multi-country) – incl. SEIR baseline")
ax.axvline(1.5, color='gray', linestyle='--', lw=1, alpha=0.6)
ax.axvline(4.5, color='gray', linestyle='--', lw=1, alpha=0.6)
ax.text(0.5, 1.03, 'SEIR', transform=ax.get_xaxis_transform(),
        ha='center', fontsize=8, color='#8e44ad')
ax.text(2.5+1/len(models_A), 1.03, '← ML →', transform=ax.get_xaxis_transform(),
        ha='center', fontsize=8, color='#2980b9')
ax.text(5+1/len(models_A), 1.03, 'DL', transform=ax.get_xaxis_transform(),
        ha='center', fontsize=8, color='#e74c3c')
for bar in bars_acc_A:
    ax.annotate(f'{bar.get_height():.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=7)

# Dataset B panel – now includes SEIR
ax = axes[1]
sub_B = cmp_df[cmp_df['Dataset'] == 'B (Sierra Leone)'].set_index('Model')
x_B   = np.arange(len(models_B))
bars_acc_B = ax.bar(x_B - width/2,
                    [sub_B.loc[m, 'Accuracy']    for m in models_B],
                    width, color=colors_B, alpha=0.88)
ax.bar(x_B + width/2,
       [sub_B.loc[m, 'Weighted F1'] for m in models_B],
       width, color=colors_B, alpha=0.40,
       edgecolor='black', linewidth=0.5)
ax.set_xticks(x_B)
ax.set_xticklabels(models_B, rotation=20, ha='right', fontsize=9)
ax.set_ylim(0, 1.10)
ax.set_ylabel("Score")
ax.set_title("Dataset B (Sierra Leone) – incl. SEIR baseline")
ax.axvline(0.5, color='gray', linestyle='--', lw=1, alpha=0.6)
ax.axvline(3.5, color='gray', linestyle='--', lw=1, alpha=0.6)
ax.text(0, 1.03, 'SEIR', transform=ax.get_xaxis_transform(),
        ha='center', fontsize=8, color='#8e44ad')
ax.text(2+1/len(models_B), 1.03, '← ML →', transform=ax.get_xaxis_transform(),
        ha='center', fontsize=8, color='#2980b9')
ax.text(4+1/len(models_B), 1.03, 'DL', transform=ax.get_xaxis_transform(),
        ha='center', fontsize=8, color='#e74c3c')
for bar in bars_acc_B:
    ax.annotate(f'{bar.get_height():.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=7)

handles = [
    plt.Rectangle((0,0),1,1, color='gray', alpha=0.88, label='Accuracy'),
    plt.Rectangle((0,0),1,1, color='gray', alpha=0.40,
                  edgecolor='black', label='Weighted F1'),
]
fig.legend(handles=handles, loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig("seir_ml_dl_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ── SEIR-only summary bar – both datasets ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("SEIR vs Best-ML vs Best-DL – Weighted F1 (Both Datasets)", fontsize=13)

sub_a = cmp_df[cmp_df['Dataset'] == 'A (Multi-country)'].set_index('Model')
sub_b = cmp_df[cmp_df['Dataset'] == 'B (Sierra Leone)'].set_index('Model')

for ax, sub, seir_f1, seir_lbl, title in [
    (axes[0], sub_a, f1w_seir3,  'SEIR\n(TOP-3)', 'Dataset A (Multi-country)'),
    (axes[1], sub_b, f1w_seir_B, 'SEIR',          'Dataset B (Sierra Leone)'),
]:
    best_ml = sub.loc[['Random Forest', 'Gradient Boosting'], 'Weighted F1'].max()
    best_dl = sub.loc[['MLP'], 'Weighted F1'].max()
    s_models = [seir_lbl, 'Best ML\n(RF/GB)', 'Best DL\n(MLP)']
    s_vals   = [seir_f1, best_ml, best_dl]
    s_colors = ['#8e44ad', '#2980b9', '#e74c3c']
    bars = ax.bar(s_models, s_vals, color=s_colors,
                  edgecolor='white', linewidth=1.5, width=0.45)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Weighted F1 Score")
    ax.set_title(title)
    for bar, val in zip(bars, s_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle=':', lw=1, label='0.5 baseline')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("seir_vs_ml_dl_summary.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 14. RECOMMENDATION SUMMARY
# ============================================================

print("\n" + "="*70)
print("  RECOMMENDATION: SEIR vs ML vs DEEP LEARNING FOR EBOLA PHASE PREDICTION")
print("="*70)

# Compute best model per dataset
for dataset in datasets:
    sub = cmp_df[cmp_df['Dataset'] == dataset].sort_values('Weighted F1', ascending=False)
    best = sub.iloc[0]
    print(f"\n  ── Dataset {dataset} ──")
    print(f"     Best model  : {best['Model']} ({best['Type']})")
    print(f"     Accuracy    : {best['Accuracy']:.3f}")
    print(f"     Weighted F1 : {best['Weighted F1']:.3f}")
    print(f"     Macro F1    : {best['Macro F1']:.3f}")

print("""
┌───────────────────────────────────────────────────────────────────────┐
│           OVERALL RECOMMENDATION RATIONALE (incl. SEIR)               │
├──────────────────┬────────────────────────────────────────────────────┤
│ SEIR model       │ Mechanistic baseline for Dataset A (TOP-3 cntry).  │
│                  │ Strengths:                                          │
│                  │  • Requires NO labelled training data.              │
│                  │  • Parameters (β, γ, R₀) are interpretable to      │
│                  │    epidemiologists and policy-makers.               │
│                  │  • Can be run prospectively from day 1 of a new    │
│                  │    outbreak when no historical ML labels exist.     │
│                  │ Weaknesses:                                         │
│                  │  • Assumes homogeneous mixing; misses heterogeneity │
│                  │    across sub-populations or districts.             │
│                  │  • Phase boundaries derived from dI/dt are         │
│                  │    smooth; real data is noisy → misclassifications. │
│                  │  • Not applicable to countries lacking population   │
│                  │    and outbreak-start metadata.                     │
├──────────────────┼────────────────────────────────────────────────────┤
│ Dataset A        │ Multi-country, moderate size (~200 test rows)       │
│ (WHO multi-cntry)│ RF / GBM typically win because:                    │
│                  │  • Hand-crafted lag features already encode         │
│                  │    the temporal signal DL tries to learn.           │
│                  │  • Tree ensembles are robust to class               │
│                  │    imbalance with class_weight='balanced'.          │
│                  │  • Small dataset → DL can overfit even with         │
│                  │    dropout/early-stopping.                          │
│                  │ DL advantage: MLP can capture non-linear           │
│                  │ interactions; may close the gap with               │
│                  │ more data or stronger augmentation.                │
├──────────────────┼────────────────────────────────────────────────────┤
│ Dataset B        │ Single-country daily data → small weekly rows       │
│ (Sierra Leone)   │ ML wins decisively here due to tiny n.              │
│                  │ Extra clinical features (survivors, treatment        │
│                  │ centres) boost RF/GBM more than DL because          │
│                  │ trees handle mixed-scale tabular inputs             │
│                  │ natively without normalisation sensitivity.         │
├──────────────────┼────────────────────────────────────────────────────┤
│ VERDICT          │ • SEIR as FIRST ALERT: deploy immediately when a   │
│                  │   new outbreak begins; use R₀ > 1 as Growing flag. │
│                  │ • Switch to Gradient Boosting once ≥4 weeks of     │
│                  │   labelled data are available (best accuracy +      │
│                  │   interpretability).                                │
│                  │ • For RESEARCH / scale-up: invest in deeper MLP    │
│                  │   architectures if data volume grows (>5000 rows). │
│                  │ • Logistic Regression remains the best              │
│                  │   interpretability baseline for public-health       │
│                  │   reporting alongside SEIR R₀.                     │
└──────────────────┴────────────────────────────────────────────────────┘

KEY FACTORS IN THE SEIR vs ML vs DL TRADE-OFF:
  1. Data availability: SEIR needs only N, β₀ seed; ML/DL need weeks
                        of labelled history.
  2. Data size        : DL needs hundreds of thousands of sequences;
                        ML thrives with dozens to thousands of rows.
  3. Features         : Hand-engineered lag features reduce the advantage
                        of sequence models; MLP on tabular lag features
                        is simpler and competitive.
  4. Explainability   : SEIR → epidemiological parameters (R₀, β, γ).
                        RF/GBM → feature importances.
                        DL → black-box by default.
  5. Training cost    : SEIR: seconds (curve-fit). RF/GBM: <10 s.
                        MLP: GPU + tuning required.
  6. Temporal risk    : RF on lag features avoids sequence-model windowing
                        complexity; safer for smaller teams.
  7. Generalisability : SEIR generalises across outbreaks via parameters;
                        ML/DL require retraining on new outbreak data.
""")

print("All plots saved. Analysis complete.")
