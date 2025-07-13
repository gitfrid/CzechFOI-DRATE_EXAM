import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from pyts.decomposition import SingularSpectrumAnalysis


# It's a histogram of death count frequencies, not a timeline.


# === CONFIGURATION ===

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\DB) Death Histogram\DB) real" # script adds differnet extensions for ACF and PACF plot 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NC) sim_NOBIAS_Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\DB) Death Histogram\DB-NC) sim NOBIAS" # script adds differnet extensions for ACF and PACF plot 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NP) sim_MINBIAS deathday_gr_doseday_random Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\DB) Death Histogram\DB-NP) sim MINBIAS deathday_gr_doseday_random" # script adds differnet extensions for ACF and PACF plot 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NK) sim_MINBIAS_deathday_gr_doseday_Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\DB) Death Histogram\DB-NK) sim MINBIAS deathday_gr_doseday" # script adds differnet extensions for ACF and PACF plot 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NO) sim minbias deathday_gr_doseday Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\DB) Death Histogram\DB-NO) sim MINBIAS deathday_gr_doseday" # script adds differnet extensions for ACF and PACF plot 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NO) sim randomly_assign_first_doses deathday_gr_doseday Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\DB) Death Histogram\DB-NO) sim randomly_assign_first_doses deathday_gr_doseday" # script adds differnet extensions for ACF and PACF plot 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NQ) sim link nearest random neighbour only death Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\DB) Death Histogram\DB-NQ) sim link nearest random neigbour only death vx uvx norm.html"

INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NU) sim link nearest random neighbour whole pop Vesely_106_202403141131.csv"
OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\DB) Death Histogram\DB-NU) sim link nearest random neighbour whole pop vx uvx norm.html"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NV) real not link nearest random neighbour only death Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\DB) Death Histogram\DB-NV) real not link nearest random neighbour only death vx uvx norm.html"


START_DATE = pd.Timestamp('2020-01-01')
MAX_AGE = 113
REFERENCE_YEAR = 2023
agevar = [70]  # 👈 Set to list(range(0, MAX_AGE + 1)) for all ages

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

print("🔹 Preprocessing...")
df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
df['age'] = REFERENCE_YEAR - df['birth_year']
df = df[df['age'].between(0, MAX_AGE)].copy()

def to_day_number(date_series):
    return (date_series - START_DATE).dt.days

df['death_day'] = to_day_number(df['datumumrti'])
for col in dose_date_cols_lower:
    df[col + '_day'] = to_day_number(df[col])

df['first_dose_day'] = df[[col + '_day' for col in dose_date_cols_lower]].min(axis=1, skipna=True)
df['has_any_dose'] = df[[col + '_day' for col in dose_date_cols_lower]].notna().any(axis=1)

END_MEASURE = int(df['death_day'].dropna().max())
days = np.arange(0, END_MEASURE + 1)

death_series_total = {}
death_series_vx = {}
death_series_uvx = {}

print("🔹 Aggregating deaths by age and vaccination status...")
for age in agevar:
    sub = df[df['age'] == age]
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

# === HISTOGRAM PLOTS ===
print(f"🔹 Creating raw death count histograms for ages: {agevar}")

# Flatten selected age data into 1D arrays
all_total = np.concatenate([death_series_total[age] for age in agevar if age in death_series_total])
all_vx = np.concatenate([death_series_vx[age] for age in agevar if age in death_series_vx])
all_uvx = np.concatenate([death_series_uvx[age] for age in agevar if age in death_series_uvx])

# Create Plotly histogram traces
hist_total = go.Histogram(
    x=all_total,
    name='Total Deaths',
    opacity=0.75,
    marker_color='blue',
    nbinsx=int(all_total.max()) + 1 if all_total.size else 1,
)

hist_vx = go.Histogram(
    x=all_vx,
    name='Vaccinated Deaths',
    opacity=0.75,
    marker_color='green',
    nbinsx=int(all_vx.max()) + 1 if all_vx.size else 1,
)

hist_uvx = go.Histogram(
    x=all_uvx,
    name='Unvaccinated Deaths',
    opacity=0.75,
    marker_color='red',
    nbinsx=int(all_uvx.max()) + 1 if all_uvx.size else 1,
)

fig_hist = go.Figure(data=[hist_total, hist_vx, hist_uvx])
fig_hist.update_layout(
    barmode='overlay',
    title=f'Histogram of Raw Death Counts (Ages: {agevar})',
    xaxis_title='Deaths per Day',
    yaxis_title='Frequency',
    template='plotly_white',
    height=600,
    width=900
)

plot(fig_hist, filename=OUTPUT_HTML + "_RawDeaths_Histogram.html", auto_open=False)
print(f"✅ Saved histogram plot: {OUTPUT_HTML}_RawDeaths_Histogram.html")

print("✅ Done.")
