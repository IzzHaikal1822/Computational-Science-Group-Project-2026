# ============================================================
# Machine Learning – Ebola 2014-2016 Dataset
# Task: Epidemic Phase Classification
# Methods: Random Forest, Gradient Boosting, Logistic Regression
#
# Key improvements over the original script:
#
# 1. MEANINGFUL TARGET
#    Original used high_cfr = (CFR > 50) while also feeding CFR as
#    a feature — the model was trivially predicting its own input.
#    Here the target is the EPIDEMIC PHASE (Growing / Stable/Peak /
#    Declining / Contained), derived from FUTURE case trajectories
#    but predicted using only PAST information.
#
# 2. NO DATA LEAKAGE
#    All features are lag-based (shifted 1-3 weeks into the past).
#    No future values appear anywhere in X.
#
# 3. TEMPORAL TRAIN/TEST SPLIT
#    Epidemic data is time-series. A random split would let the model
#    "see" future weeks during training (temporal leakage). We split
#    at a fixed date: train on data before 2015-10-01, test after.
#
# 4. WEEKLY AGGREGATION
#    The raw data has irregular reporting intervals (1-7 days).
#    Weekly resampling normalises the time axis and reduces noise.
#
# 5. GRADIENT BOOSTING ADDED
#    Replaces the Linear SVM (which adds no value over LogReg on this
#    small dataset). GBM handles class imbalance and non-linearities
#    without the calibration wrapper required by LinearSVC.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

# ── NOTE ──────────────────────────────────────────────────────────────────────
# Update the path below to point to your local copy of the CSV.
CSV_PATH = "ebola_2014_2016_clean.csv"
# ─────────────────────────────────────────────────────────────────────────────


# ============================================================
# 1. DATA LOADING & CLEANING
# ============================================================

df = pd.read_csv(CSV_PATH, parse_dates=['Date'])
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

# New cases / deaths (daily increments)
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
# 2. WEEKLY AGGREGATION
# ============================================================
# Irregular reporting (1-7 day gaps) is normalised to weekly buckets.
# This makes the lag features consistent across all countries.

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
# 3. TARGET: EPIDEMIC PHASE
# ============================================================
# For each week we compare the 3-week trailing average of new cases
# (past) against the 3-week leading average (future) to assign one
# of four epidemic phases.
#
# Although the label depends on future observations, the FEATURES
# (section 4) use only lagged values — there is no leakage.
#
# Thresholds:
#   Growing     : future avg > 125% of past avg
#   Declining   : future avg < 75%  of past avg
#   Stable/Peak : 75–125% change
#   Contained   : fewer than 2 cases/week in both windows

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

print("Epidemic phase distribution:")
print(df_w['Phase'].value_counts())


# ============================================================
# 4. FEATURE ENGINEERING (past-only, no leakage)
# ============================================================
# Every feature is either a fixed property of the country/time or
# a lagged value — nothing from the current or future weeks.

le = LabelEncoder()
df_w['country_enc'] = le.fit_transform(df_w['Country'])

df_w['days_into_epidemic'] = df_w.groupby('Country')['Date'].transform(
    lambda x: (x - x.min()).dt.days)

# Lagged new cases / deaths (1, 2, 3 weeks ago)
for lag in [1, 2, 3]:
    df_w[f'new_cases_lag{lag}']  = df_w.groupby('Country')['new_cases'].shift(lag)
    df_w[f'new_deaths_lag{lag}'] = df_w.groupby('Country')['new_deaths'].shift(lag)

# 3-week rolling averages (of lagged values)
df_w['cases_3w_avg']  = df_w.groupby('Country')['new_cases'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df_w['deaths_3w_avg'] = df_w.groupby('Country')['new_deaths'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())

# Week-on-week growth rate (clipped to [-5, 5] to handle division by near-zero)
df_w['case_growth_rate'] = (
    df_w.groupby('Country')['new_cases'].shift(1) /
    df_w.groupby('Country')['cum_cases'].shift(2).replace(0, np.nan)
).clip(-5, 5)

df_w['death_growth_rate'] = (
    df_w.groupby('Country')['new_deaths'].shift(1) /
    df_w.groupby('Country')['cum_deaths'].shift(2).replace(0, np.nan)
).clip(-5, 5)

# Lagged cumulative totals (epidemic scale)
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
X = df_ml[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df_ml['Phase']
dates = df_ml['Date']

print(f"\nML dataset: {X.shape[0]} rows × {X.shape[1]} features")
print("Class balance:\n", y.value_counts())


# ============================================================
# 5. TEMPORAL TRAIN / TEST SPLIT
# ============================================================
# Train on weeks before 2015-10-01, test on weeks from 2015-10-01
# onward.  This mirrors real-world deployment: the model is always
# predicting future phases, never past ones.

SPLIT_DATE = pd.Timestamp('2015-10-01')
train_mask = dates < SPLIT_DATE
test_mask  = dates >= SPLIT_DATE

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nTrain weeks: {len(X_train)}  |  Test weeks: {len(X_test)}")

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)


# ============================================================
# 6. MODEL DEFINITIONS
# ============================================================

# ── 6a. Random Forest ─────────────────────────────────────────────────────────
# Ensemble of 300 decorrelated trees.
# class_weight='balanced' up-weights the minority phases (Growing,
# Stable/Peak) so the model does not collapse to predicting 'Contained'
# for everything.
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)

# ── 6b. Gradient Boosting ─────────────────────────────────────────────────────
# Sequential ensemble that fits each tree to the residual errors of
# the previous ensemble.  More expressive than a random forest on
# small datasets with imbalanced classes; the lower learning_rate
# with more trees (shrinkage) reduces overfitting.
# Note: sklearn's GBM does not support class_weight directly; the
# balanced training set size handles imbalance implicitly here.
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
)

# ── 6c. Logistic Regression ───────────────────────────────────────────────────
# Linear multi-class baseline (one-vs-rest).
# Requires scaled features (StandardScaler applied above).
# Coefficient magnitudes and signs are directly interpretable:
#   positive coef → that feature increases the probability of that phase.
lr_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
)


# ============================================================
# 7. TRAINING
# ============================================================

# RF and GB use raw (unscaled) features — tree models are scale-invariant
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# LR requires standardised features
lr_model.fit(X_train_sc, y_train)


# ============================================================
# 8. EVALUATION
# ============================================================

rf_pred  = rf_model.predict(X_test)
gb_pred  = gb_model.predict(X_test)
lr_pred  = lr_model.predict(X_test_sc)

for name, pred in [
    ('Random Forest',       rf_pred),
    ('Gradient Boosting',   gb_pred),
    ('Logistic Regression', lr_pred),
]:
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_test, pred, zero_division=0,
                                target_names=sorted(y.unique())))


# ============================================================
# 9. VISUALISATIONS
# ============================================================

CLASSES = sorted(y.unique())
PHASE_COLORS = {
    'Growing':     '#e74c3c',
    'Stable/Peak': '#f39c12',
    'Declining':   '#3498db',
    'Contained':   '#95a5a6',
}

# ── 9a. Confusion matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices – Epidemic Phase Classification", fontsize=14)

for ax, (name, pred) in zip(axes, [
    ('Random Forest',       rf_pred),
    ('Gradient Boosting',   gb_pred),
    ('Logistic Regression', lr_pred),
]):
    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, colorbar=False, cmap='Blues', xticks_rotation=30)
    ax.set_title(name)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.show()


# ── 9b. Feature importances (Random Forest) ───────────────────────────────────
importances = pd.Series(rf_model.feature_importances_,
                        index=FEATURE_LABELS).sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
palette = sns.color_palette('rocket_r', len(importances))
importances.plot(kind='barh', ax=ax, color=palette, edgecolor='white')
ax.set_title(
    "Random Forest – Feature Importances\n"
    "(predicting epidemic phase from lagged data only)",
    fontsize=13)
ax.set_xlabel("Importance Score")
ax.axvline(1 / len(importances), color='gray', linestyle='--',
           lw=1, label='Uniform baseline')
ax.legend()
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150, bbox_inches='tight')
plt.show()


# ── 9c. Phase timeline for the three most-affected countries ──────────────────
TOP3 = ['Guinea', 'Liberia', 'Sierra Leone']

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle(
    "Epidemic Phase Labels vs Weekly New Cases (Top 3 Affected Countries)",
    fontsize=13, y=1.01)

for ax, country in zip(axes, TOP3):
    sub = df_w[df_w['Country'] == country].copy().sort_values('Date')
    ax.fill_between(sub['Date'], sub['new_cases'], alpha=0.3, color='steelblue')
    ax.plot(sub['Date'], sub['new_cases'], color='steelblue', lw=1.5)
    for _, row in sub.iterrows():
        ax.axvspan(
            row['Date'],
            row['Date'] + pd.Timedelta(weeks=1),
            alpha=0.18,
            color=PHASE_COLORS.get(row['Phase'], 'gray'))
    ax.set_ylabel("New cases / week", fontsize=9)
    ax.set_title(country, fontsize=11, loc='left')
    patches = [Patch(color=c, alpha=0.7, label=p)
               for p, c in PHASE_COLORS.items()]
    ax.legend(handles=patches, loc='upper right', fontsize=8, ncol=4)

plt.tight_layout()
plt.savefig("phase_timeline.png", dpi=150, bbox_inches='tight')
plt.show()


# ── 9d. Logistic Regression coefficients ─────────────────────────────────────
coef_df = pd.DataFrame(lr_model.coef_,
                       columns=FEATURE_LABELS,
                       index=lr_model.classes_)

fig, axes = plt.subplots(1, len(lr_model.classes_), figsize=(20, 5), sharey=True)
fig.suptitle(
    "Logistic Regression Coefficients by Phase\n"
    "(positive = increases probability of that phase)",
    fontsize=12)

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

print("\nML pipeline complete. Plots saved.")


# ============================================================
# 10. SEIR EPIDEMIC MODEL
# ============================================================
# The SEIR model divides the population N into four compartments:
#
#   S  – Susceptible : can contract the disease
#   E  – Exposed     : infected but in the incubation period (not yet infectious)
#   I  – Infectious  : actively capable of transmitting the disease
#   R  – Recovered / Removed : recovered (immune) or deceased
#                              (ties into the CFR calculated in section 1)
#
# The dynamics are governed by a system of coupled ODEs:
#
#   dS/dt = -(β · S · I) / N
#   dE/dt =  (β · S · I) / N  -  σ · E
#   dI/dt =   σ · E           -  γ · I
#   dR/dt =   γ · I
#
# Parameters:
#   N  – Total population  (N = S + E + I + R, conserved throughout)
#   β  – Transmission rate : rate at which a susceptible individual catches
#         the disease from one infectious individual
#   σ  – Incubation rate   : rate at which exposed individuals become
#         infectious; reciprocal of the incubation period
#         (σ ≈ 1 / 11.5 days for Ebola)
#   γ  – Recovery/mortality rate : rate at which infectious individuals
#         either recover or die (reciprocal of the infectious period)
#
# The basic reproduction number is R₀ = β / γ.
# R₀ > 1 → epidemic grows; R₀ < 1 → epidemic fades.
# ============================================================


def seir_odes(t, y, beta, sigma, gamma, N):
    """
    Right-hand side of the SEIR system.

    Parameters
    ----------
    t     : float        Current time (days) — required by solve_ivp signature
    y     : array-like   [S, E, I, R]
    beta  : float        Transmission rate (day⁻¹)
    sigma : float        Incubation rate   (day⁻¹)  ≈ 1/11.5 for Ebola
    gamma : float        Recovery/mortality rate (day⁻¹)
    N     : float        Total population size

    Returns
    -------
    [dS/dt, dE/dt, dI/dt, dR/dt]
    """
    S, E, I, R = y
    force_of_infection = beta * S * I / N   # new exposures per day

    dS = -force_of_infection
    dE =  force_of_infection - sigma * E
    dI =  sigma * E          - gamma * I
    dR =  gamma * I

    return [dS, dE, dI, dR]


def run_seir(N, beta, sigma, gamma, E0, I0, t_max):
    """
    Integrate the SEIR ODEs from t=0 to t=t_max days.

    Parameters
    ----------
    N      : float   Total population
    beta   : float   Transmission rate (day⁻¹)
    sigma  : float   Incubation rate   (day⁻¹)
    gamma  : float   Recovery/mortality rate (day⁻¹)
    E0     : float   Initial number of exposed individuals
    I0     : float   Initial number of infectious individuals
    t_max  : int     Duration to simulate (days)

    Returns
    -------
    t      : ndarray  Time points (days)
    S,E,I,R: ndarray  Compartment sizes at each time point
    """
    S0 = N - E0 - I0
    R0_init = 0.0
    y0 = [S0, E0, I0, R0_init]

    sol = solve_ivp(
        fun=seir_odes,
        t_span=(0, t_max),
        y0=y0,
        args=(beta, sigma, gamma, N),
        method='RK45',
        dense_output=True,
        max_step=1.0,          # 1-day resolution
    )

    t = np.linspace(0, t_max, t_max + 1)
    S, E, I, R = sol.sol(t)
    return t, S, E, I, R


def fit_seir_to_country(country_df, N, sigma=1/11.5):
    """
    Fit β and γ to observed cumulative cases for one country via
    least-squares minimisation (Nelder-Mead).

    Parameters
    ----------
    country_df : DataFrame   Weekly data for one country (must have
                             columns 'Date' and 'cum_cases')
    N          : float       Assumed total population
    sigma      : float       Fixed incubation rate (default 1/11.5 for Ebola)

    Returns
    -------
    beta_fit  : float   Fitted transmission rate
    gamma_fit : float   Fitted recovery/mortality rate
    R0        : float   Basic reproduction number  (β / γ)
    """
    # Convert dates to integer days from first observation
    dates_sorted = country_df.sort_values('Date')
    t_obs = (dates_sorted['Date'] - dates_sorted['Date'].iloc[0]).dt.days.values
    obs   = dates_sorted['cum_cases'].values.astype(float)

    t_max = int(t_obs[-1])
    E0    = max(obs[0] * 2, 1)   # rough guess: twice the first confirmed cases
    I0    = max(obs[0],     1)

    def residuals(params):
        beta, gamma = params
        if beta <= 0 or gamma <= 0:
            return 1e12
        try:
            t, S, E, I, R = run_seir(N, beta, gamma, sigma, E0, I0, t_max)
            # Model cumulative cases = E + I + R  (everyone who left S)
            cum_model = N - S
            # Interpolate model at observed time points
            cum_at_obs = np.interp(t_obs, t, cum_model)
            return np.sum((cum_at_obs - obs) ** 2)
        except Exception:
            return 1e12

    # Initial guesses: β ≈ 0.27, γ ≈ 0.10 (literature values for 2014 outbreak)
    result = minimize(residuals, x0=[0.27, 0.10],
                      method='Nelder-Mead',
                      options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})

    beta_fit, gamma_fit = result.x
    R0 = beta_fit / gamma_fit
    return beta_fit, gamma_fit, R0


# ── 10a. Fit SEIR to each of the three most-affected countries ────────────────
# Population estimates (approximate, 2014):
#   Guinea       ≈ 12,000,000
#   Liberia      ≈  4,500,000
#   Sierra Leone ≈  7,000,000
POPULATIONS = {
    'Guinea':       12_000_000,
    'Liberia':       4_500_000,
    'Sierra Leone':  7_000_000,
}

SIGMA_EBOLA = 1 / 11.5   # incubation rate (1 / mean incubation period in days)

print("\n" + "="*60)
print("  SEIR MODEL – FITTED PARAMETERS")
print("="*60)
print(f"  Fixed  σ (incubation rate) = 1/11.5 ≈ {SIGMA_EBOLA:.4f} day⁻¹")
print("="*60)

seir_results = {}
for country in TOP3:
    country_data = df_w[df_w['Country'] == country].copy()
    N = POPULATIONS[country]

    beta_fit, gamma_fit, R0 = fit_seir_to_country(
        country_data, N=N, sigma=SIGMA_EBOLA)

    seir_results[country] = {
        'beta':  beta_fit,
        'gamma': gamma_fit,
        'R0':    R0,
        'N':     N,
    }

    print(f"\n  {country} (N = {N:,})")
    print(f"    β (transmission rate)       = {beta_fit:.4f} day⁻¹")
    print(f"    γ (recovery/mortality rate) = {gamma_fit:.4f} day⁻¹")
    print(f"    Infectious period           ≈ {1/gamma_fit:.1f} days")
    print(f"    R₀ = β / γ                 = {R0:.2f}")
    if R0 > 1:
        print(f"    → Epidemic was growing (R₀ > 1) at fitted parameters")
    else:
        print(f"    → Epidemic was declining (R₀ ≤ 1) at fitted parameters")


# ── 10b. Visualise SEIR fits against observed data ────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
fig.suptitle(
    "SEIR Model Fit vs Observed Cumulative Cases\n"
    "(Top 3 Affected Countries – 2014–2016 Ebola Outbreak)",
    fontsize=13)

for ax, country in zip(axes, TOP3):
    res = seir_results[country]
    country_data = df_w[df_w['Country'] == country].sort_values('Date')

    # Days from outbreak start
    t0 = country_data['Date'].iloc[0]
    t_obs = (country_data['Date'] - t0).dt.days.values
    obs   = country_data['cum_cases'].values

    t_max = int(t_obs[-1])
    E0 = max(obs[0] * 2, 1)
    I0 = max(obs[0],     1)

    t_sim, S, E, I, R = run_seir(
        N=res['N'],
        beta=res['beta'],
        gamma=res['gamma'],
        sigma=SIGMA_EBOLA,
        E0=E0,
        I0=I0,
        t_max=t_max,
    )
    cum_model = res['N'] - S   # cumulative cases = everyone who left S

    # Plot
    ax.plot(t_sim, cum_model, color='#e74c3c', lw=2,
            label=f"SEIR fit  (R₀={res['R0']:.2f}, β={res['beta']:.3f}, γ={res['gamma']:.3f})")
    ax.scatter(t_obs, obs, color='steelblue', s=18, zorder=5,
               label='Observed cumulative cases')
    ax.set_ylabel("Cumulative cases", fontsize=9)
    ax.set_xlabel("Days since first observation", fontsize=9)
    ax.set_title(country, fontsize=11, loc='left')
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig("seir_fit.png", dpi=150, bbox_inches='tight')
plt.show()


# ── 10c. SEIR compartment trajectories (Sierra Leone as illustrative example) ─
country   = 'Sierra Leone'
res       = seir_results[country]
t_sim, S, E, I, R = run_seir(
    N=res['N'],
    beta=res['beta'],
    gamma=res['gamma'],
    sigma=SIGMA_EBOLA,
    E0=10,
    I0=5,
    t_max=700,
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t_sim, S / res['N'] * 100, label='Susceptible (S)',  color='steelblue', lw=2)
ax.plot(t_sim, E / res['N'] * 100, label='Exposed (E)',      color='#f39c12',   lw=2)
ax.plot(t_sim, I / res['N'] * 100, label='Infectious (I)',   color='#e74c3c',   lw=2)
ax.plot(t_sim, R / res['N'] * 100, label='Recovered/Removed (R)', color='#27ae60', lw=2)
ax.set_xlabel("Days since outbreak start", fontsize=11)
ax.set_ylabel("% of population", fontsize=11)
ax.set_title(
    f"SEIR Compartment Trajectories – {country}\n"
    f"β={res['beta']:.3f}, σ={SIGMA_EBOLA:.4f}, γ={res['gamma']:.3f}, R₀={res['R0']:.2f}",
    fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("seir_compartments.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nSEIR modelling complete. Plots saved.")
