import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf
from plotly.offline import plot
from scipy.stats import norm


# === CONFIGURATION ===

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\Vesely_106_202403141131.csv"
#OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\D) ACF-PACF randtest\D) real" # script adds differnet extensions for ACF and PACF plot 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NC) sim_NOBIAS_Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\D) ACF-PACF randtest\D-NC) sim NOBIAS" # script adds differnet extensions for ACF and PACF plot 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NP) sim_MINBIAS deathday_gr_doseday_random Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\D) ACF-PACF randtest\B-NP) sim MINBIAS deathday_gr_doseday_random" # script adds differnet extensions for ACF and PACF plot 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NK) sim_MINBIAS_deathday_gr_doseday_Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\D) ACF-PACF randtest\D-NK) sim MINBIAS deathday_gr_doseday" # script adds differnet extensions for ACF and PACF plot 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NO) sim minbias deathday_gr_doseday Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\D) ACF-PACF randtest\D-NO) sim MINBIAS deathday_gr_doseday" # script adds differnet extensions for ACF and PACF plot 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NO) sim randomly_assign_first_doses deathday_gr_doseday Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\D) ACF-PACF randtest\D-NO) sim randomly_assign_first_doses deathday_gr_doseday" # script adds differnet extensions for ACF and PACF plot 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NQ) sim link nearest random neighbour only death Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\D) ACF-PACF randtest\D-NQ) sim link nearest random neigbour only death vx uvx norm.html"

INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NU) sim link nearest random neighbour whole pop Vesely_106_202403141131.csv"
OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\D) ACF-PACF randtest\D-NU) sim link nearest random neighbour whole pop vx uvx norm.html"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NV) real not link nearest random neighbour only death Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\D) ACF-PACF randtest\D-NV) real not link nearest random neighbour only death vx uvx norm.html"

START_DATE = pd.Timestamp('2020-01-01')
MAX_AGE = 113
REFERENCE_YEAR = 2023
MAX_LAG = 547
AGE_SELECTED = [70]  # 👈 Set to [] to run for all ages

print("🔹 Loading data...")
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
needed_cols = ['Rok_narozeni', 'DatumUmrti'] + dose_date_cols

df = pd.read_csv(
    INPUT_CSV,
    usecols=needed_cols,
    parse_dates=['DatumUmrti'] + dose_date_cols,
    dayfirst=False,
    low_memory=False
)

df.columns = [col.strip().lower() for col in df.columns]
dose_date_cols_lower = [col.lower() for col in dose_date_cols]

print("🔹 Preprocessing date and age columns...")
df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
df['age'] = REFERENCE_YEAR - df['birth_year']
df = df[df['age'].between(0, MAX_AGE)].copy()

# Apply age filter if AGE_SELECTED is defined
if AGE_SELECTED:
    df = df[df['age'].isin(AGE_SELECTED)]

def to_day_number(date_series):
    return (date_series - START_DATE).dt.days

df['death_day'] = to_day_number(df['datumumrti'])
for col in dose_date_cols_lower:
    df[col + '_day'] = to_day_number(df[col])

df['first_dose_day'] = df[[col + '_day' for col in dose_date_cols_lower]].min(axis=1, skipna=True)
df['has_any_dose'] = df[[col + '_day' for col in dose_date_cols_lower]].notna().any(axis=1)

END_MEASURE = int(df['death_day'].dropna().max())
days = np.arange(0, END_MEASURE + 1)
ages = sorted(df['age'].unique())  # Only relevant ages based on AGE_SELECTED

death_series_total = {}
death_series_vx = {}
death_series_uvx = {}

print("🔹 Aggregating deaths by age and vaccination status...")
df_age_groups = [df[df['age'] == age] for age in ages]
for age, sub in zip(ages, df_age_groups):
    if sub.empty:
        continue

    death_days = sub['death_day'].values
    first_dose_days = sub['first_dose_day'].values
    has_any_dose = sub['has_any_dose'].values

    d_total = np.zeros_like(days, dtype=int)
    d_vx = np.zeros_like(days, dtype=int)
    d_uvx = np.zeros_like(days, dtype=int)

    for i, day in enumerate(days):
        death_today = (death_days == day)
        is_vaxed = (day >= first_dose_days) & has_any_dose
        is_uvx = ~is_vaxed

        d_total[i] = np.sum(death_today)
        d_vx[i] = np.sum(death_today & is_vaxed)
        d_uvx[i] = np.sum(death_today & is_uvx)

    death_series_total[age] = d_total
    death_series_vx[age] = d_vx
    death_series_uvx[age] = d_uvx

print("🔹 Computing ACF/PACF for each age with ≥10 non-zero total death days...")
acf_traces = []
pacf_traces = []

for age in ages:
    if age not in death_series_total:
        continue

    s_total = death_series_total[age]
    s_vx = death_series_vx[age]
    s_uvx = death_series_uvx[age]

    if np.count_nonzero(s_total) < 10:
        print(f"⚠️ Skipping age {age}: not enough non-zero total deaths")
        continue

    print(f"✅ Processing age {age}...")

    acf_total_vals, confint_total = acf(s_total, nlags=MAX_LAG, fft=True, alpha=0.05)
    acf_vx_vals, confint_vx = acf(s_vx, nlags=MAX_LAG, fft=True, alpha=0.05)
    acf_uvx_vals, confint_uvx = acf(s_uvx, nlags=MAX_LAG, fft=True, alpha=0.05)

    pacf_total_vals, confint_pacf_total = pacf(s_total, nlags=MAX_LAG, method='ywm', alpha=0.05)
    pacf_vx_vals, confint_pacf_vx = pacf(s_vx, nlags=MAX_LAG, method='ywm', alpha=0.05)
    pacf_uvx_vals, confint_pacf_uvx = pacf(s_uvx, nlags=MAX_LAG, method='ywm', alpha=0.05)

    lags = np.arange(MAX_LAG + 1)

    def ci_band(y_upper, y_lower, label, fill_color='rgba(200,200,200,0.2)'):
        return [
            go.Scatter(
                x=lags,
                y=y_upper,
                mode='lines',
                line=dict(width=0),
                name=f"{label} Upper CI",
                showlegend=False
            ),
            go.Scatter(
                x=lags,
                y=y_lower,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=fill_color,
                name=f"{label} 95% CI",
                showlegend=True
            )
        ]

    acf_traces.extend([
        go.Scatter(x=lags, y=acf_total_vals, name=f'ACF Age {age} - Total', line=dict(width=1)),
        *ci_band(confint_total[:, 1], confint_total[:, 0], f'ACF Age {age} - Total'),

        go.Scatter(x=lags, y=acf_vx_vals, name=f'ACF Age {age} - Vaxed', line=dict(width=1)),
        *ci_band(confint_vx[:, 1], confint_vx[:, 0], f'ACF Age {age} - Vaxed'),

        go.Scatter(x=lags, y=acf_uvx_vals, name=f'ACF Age {age} - Unvaxed', line=dict(width=1)),
        *ci_band(confint_uvx[:, 1], confint_uvx[:, 0], f'ACF Age {age} - Unvaxed'),
    ])

    pacf_traces.extend([
        go.Scatter(x=lags, y=pacf_total_vals, name=f'PACF Age {age} - Total', line=dict(width=1)),
        *ci_band(confint_pacf_total[:, 1], confint_pacf_total[:, 0], f'PACF Age {age} - Total'),

        go.Scatter(x=lags, y=pacf_vx_vals, name=f'PACF Age {age} - Vaxed', line=dict(width=1)),
        *ci_band(confint_pacf_vx[:, 1], confint_pacf_vx[:, 0], f'PACF Age {age} - Vaxed'),

        go.Scatter(x=lags, y=pacf_uvx_vals, name=f'PACF Age {age} - Unvaxed', line=dict(width=1)),
        *ci_band(confint_pacf_uvx[:, 1], confint_pacf_uvx[:, 0], f'PACF Age {age} - Unvaxed'),
    ])

print("📊 Creating Plotly figures...")

fig_acf = go.Figure(data=acf_traces)
fig_acf.update_layout(
    title="ACF per Age (Total, Vaxed, Unvaxed) with 95% CI",
    xaxis_title="Lag (days)",
    yaxis_title="Autocorrelation",
    height=800,
    template='plotly_white'
)

fig_pacf = go.Figure(data=pacf_traces)
fig_pacf.update_layout(
    title="PACF per Age (Total, Vaxed, Unvaxed) with 95% CI",
    xaxis_title="Lag (days)",
    yaxis_title="Partial Autocorrelation",
    height=800,
    template='plotly_white'
)

print("💾 Saving plots to HTML...")

plot(fig_acf, filename=f"{OUTPUT_HTML}_ACF.html", auto_open=False)
plot(fig_pacf, filename=f"{OUTPUT_HTML}_PACF.html", auto_open=False)

print("✅ All done.")
