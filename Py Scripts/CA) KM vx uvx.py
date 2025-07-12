import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import plotly.graph_objects as go
import os  # Add this at the top if not already present

# === Constants ===

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA) real vx uvx.html" 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NC) sim_NOBIAS_Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-NC) sim NOBIAS vx uvx.html" 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NP) sim_MINBIAS deathday_gr_doseday_random Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-NP) sim MINBIAS deathday_gr_doseday_random vx uvx.html" 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NK) sim_MINBIAS_deathday_gr_doseday_Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-NK) sim MINBIAS deathday_gr_doseday vx uvx.html"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NO) sim minbias deathday_gr_doseday Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-NO) sim MINBIAS deathday_gr_doseday vx uvx.html"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NO) sim randomly_assign_first_doses deathday_gr_doseday Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-NO) sim randomly_assign_first_doses deathday_gr_doseday vx uvx.html" 

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NQ) sim link nearest random neighbour only death Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-NQ) sim link nearest random neigbour only death vx uvx.html"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NU) sim link nearest random neighbour whole pop Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-NU) sim link nearest random neighbour whole pop vx uvx.html"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NW) sim link nearest random neighbour only deaths Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-NW) sim link nearest random neighbour only deaths vx uvx.html"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NV) real not link nearest random neighbour only death Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-NV) real not link nearest random neighbour only death vx uvx.html"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NR) sim random doses Vesely_106_202403141131_doses.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-NR) sim random doses vx uvx.html"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NT) sim random deaths Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-NT) sim random deaths vx uvx.html"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NX) sim random deaths and doses Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-NX) sim random deaths and doses vx uvx.html"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\NY) sim random all doses Vesely_106_202403141131_doses.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-NY) sim random all doses vx uvx.html"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\AA) sim minbias deathday_gr_doseday Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-AA) sim minbias deathday_gr_doseday vx uvx.html"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\AA) sim randomly_assign_first_doses deathday_gr_doseday Vesely_106_202403141131.csv"
# OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-AA) sim randomly_assign_first_doses deathday_gr_doseday vx uvx.html"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\AA) sim randomly_assign_first_doses deathday_gr_doseday Vesely_106_202403141131.csv"
#OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\AA) sim randomly assign_first_doses deathday_gr_doseday.html"


#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\#case2_sim_deaths_real_doses_no_constraint.csv"
#OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA) #case2_sim_deaths_real_doses_no_constraint.csv.html"


# 6 input CSV files (replace paths if needed)
#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AA) case1_real_deaths_real_doses.csv"
#OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-AA) case1_real_deaths_real_doses.html"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AA) case2_sim_deaths_real_doses_no_constraint.csv"
#OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-AA) case2_sim_deaths_real_doses_no_constraint.html"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AA) case3_sim_deaths_sim_real_doses_with_constraint.csv"
#OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-AA) case3_sim_deaths_sim_real_doses_with_constraint.html"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AA) case4_sim_deaths_sim_real_doses_no_constraint.csv"
#OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-AA) case4_sim_deaths_sim_real_doses_no_constraint.html"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AA) case5_sim_deaths_sim_flat_random_doses_with_constraint.csv"
#OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-AA) case5_sim_deaths_sim_flat_random_doses_with_constraint.html"

INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AA) case6_sim_deaths_sim_flat_random_doses_no_constraint.csv"
OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\CA) KM vx uvx\CA-AA) case6_sim_deaths_sim_flat_random_doses_no_constraint.html"


# CONFIG

START_DATE = pd.Timestamp('2020-01-01')
MAX_AGE = 113
REFERENCE_YEAR = 2023

# === Age Filter ===
AGE_SELECTED = [70]  # set [] for all ages but don't touch rest of logic

# === Load and Prepare Data ===
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

df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
df['age'] = REFERENCE_YEAR - df['birth_year']
df = df[df['age'].between(0, MAX_AGE)].copy()

# === Apply Age Filter if any ===
if AGE_SELECTED:
    df = df[df['age'].isin(AGE_SELECTED)]

def to_day_number(date_series):
    return (date_series - START_DATE).dt.days

df['death_day'] = to_day_number(df['datumumrti'])
for col in dose_date_cols_lower:
    df[col + '_day'] = to_day_number(df[col])

df['first_dose_day'] = df[[col + '_day' for col in dose_date_cols_lower]].min(axis=1, skipna=True)
df['has_any_dose'] = df[[col + '_day' for col in dose_date_cols_lower]].notna().any(axis=1)

# === Add censoring ===
df['censor_day'] = df['death_day'].isna()
# df['death_day'].fillna(df['death_day'].max() + 1, inplace=True)
df['death_day'] = df['death_day'].fillna(df['death_day'].max() + 1)
df['event'] = (~df['censor_day']).astype(int)

# === Group assignment ===
df['group'] = 'uvx'
df.loc[df['has_any_dose'], 'group'] = 'vx'
df['group'] = df['group'].astype('category')

print("Data shape after filtering by age:", df.shape)
print("Group counts:\n", df['group'].value_counts())
print("Death day stats:")
print(df['death_day'].describe())
print("Event counts:")
print(df['event'].value_counts())
print("Any NaNs in death_day?", df['death_day'].isna().sum())


# === Fit KM curves ===
kmf_total = KaplanMeierFitter()
kmf_vx = KaplanMeierFitter()
kmf_uvx = KaplanMeierFitter()

T_total = df['death_day']
E_total = df['event']

T_vx = df[df['group'] == 'vx']['death_day']
E_vx = df[df['group'] == 'vx']['event']

T_uvx = df[df['group'] == 'uvx']['death_day']
E_uvx = df[df['group'] == 'uvx']['event']

kmf_total.fit(T_total, event_observed=E_total, label='Total')
kmf_vx.fit(T_vx, event_observed=E_vx, label='Vaccinated')
kmf_uvx.fit(T_uvx, event_observed=E_uvx, label='Unvaccinated')

input_filename = os.path.basename(INPUT_CSV)

# === Plotly Output ===
fig = go.Figure()

for kmf in [kmf_total, kmf_vx, kmf_uvx]:
    fig.add_trace(go.Scatter(
        x=kmf.survival_function_.index,
        y=kmf.survival_function_[kmf._label],
        mode='lines',
        name=kmf._label
    ))

fig.update_layout(
    title=f'Kaplan-Meier Survival Curves: Total vs Vaccinated vs Unvaccinated AGE:{AGE_SELECTED}<br><sub>Input CSV: {input_filename}</sub>',
    xaxis_title='Days Since Jan 1, 2020',
    yaxis_title='Survival Probability',
    template='plotly_white'
)

fig.write_html(OUTPUT_HTML)
print(f"Plot saved to: {OUTPUT_HTML}")
