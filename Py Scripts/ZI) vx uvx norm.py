import pandas as pd
import numpy as np
import plotly.graph_objs as go


# Vaccine Death Rate Comparison Script

# Description:
# This Python script analyzes simulated or real data on individual vaccination and death records
# to calculate and visualize death rates for vaccinated (vx), unvaccinated (uvx), and total populations.
# It handles smoothing, normalization per 100,000 people, and separation of deaths by vaccination status.
# Dose counts (first and all doses) are also calculated and optionally smoothed.

# Visualization is performed using Plotly and saved as interactive html file


# 6 input CSV files (replace paths if needed)

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AA) case1_real_deaths_real_doses.csv"
#OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\ZI) vx uvx norm\ZI-AA) case1_real_deaths_real_doses.html"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AA) case2_sim_deaths_real_doses_no_constraint.csv"
#OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\ZI) vx uvx norm\ZI-AA) case2_sim_deaths_real_doses_no_constraint.html"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AA) case3_sim_deaths_sim_real_doses_with_constraint.csv"
#OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\ZI) vx uvx norm\ZI-AA) case3_sim_deaths_sim_real_doses_with_constraint.html"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AA) case4_sim_deaths_sim_real_doses_no_constraint.csv"
#OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\ZI) vx uvx norm\ZI-AA) case4_sim_deaths_sim_real_doses_no_constraint.html"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AA) case5_sim_deaths_sim_flat_random_doses_with_constraint.csv"
#OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\ZI) vx uvx norm\ZI-AA) case5_sim_deaths_sim_flat_random_doses_with_constraint.html"

INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AA) case6_sim_deaths_sim_flat_random_doses_no_constraint.csv"
OUTPUT_HTML = r"C:\CzechFOI-DRATE_EXAM\Plot Results\ZI) vx uvx norm\ZI-AA) case6_sim_deaths_sim_flat_random_doses_no_constraint.html"


START_DATE = pd.Timestamp('2020-01-01')  # Start of simulation
MAX_AGE = 113                            # Maximum considered age
REFERENCE_YEAR = 2023                    # Used to calculate age from birth year

# === Load and Preprocess Data ===

# Define dose date column names (up to 7 doses)
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
needed_cols = ['Rok_narozeni', 'DatumUmrti'] + dose_date_cols

# Read CSV while parsing date columns
df = pd.read_csv(
    INPUT_CSV,
    usecols=needed_cols,
    parse_dates=['DatumUmrti'] + dose_date_cols,
    dayfirst=False,
    low_memory=False
)

# Lowercase all column names for consistency
df.columns = [col.strip().lower() for col in df.columns]
dose_date_cols_lower = [col.lower() for col in dose_date_cols]

# Calculate individual's age
df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
df['age'] = REFERENCE_YEAR - df['birth_year']
df = df[df['age'].between(0, MAX_AGE)].copy()  # Keep only valid age range

# Convert all date columns to "days since START_DATE"
def to_day_number(date_series):
    return (date_series - START_DATE).dt.days

df['death_day'] = to_day_number(df['datumumrti'])
for col in dose_date_cols_lower:
    df[col + '_day'] = to_day_number(df[col])

# Compute first dose day and whether any dose was taken
df['first_dose_day'] = df[[col + '_day' for col in dose_date_cols_lower]].min(axis=1, skipna=True)
df['has_any_dose'] = df[[col + '_day' for col in dose_date_cols_lower]].notna().any(axis=1)

# === Prepare Simulation Structure ===

END_MEASURE = int(df['death_day'].dropna().max())  # Last day with any death
days = np.arange(0, END_MEASURE + 1)               # Simulation days
ages = np.arange(0, MAX_AGE + 1)                   # Age groups 0–113

# Result storage dictionary
results = {
    'day': [], 'age': [],
    'pop_vx': [], 'pop_uvx': [], 'pop_total': [],
    'death_vx': [], 'death_uvx': [], 'death_total': []
}

# Pre-split data into age-specific subgroups
df_age_groups = [df[df['age'] == age] for age in ages]

# === Main Analysis Loop: Calculate Population and Deaths ===

for age, sub in zip(ages, df_age_groups):
    if sub.empty:
        continue

    death_days = sub['death_day'].values
    first_dose_days = sub['first_dose_day'].values
    has_any_dose = sub['has_any_dose'].values

    for day in days:
        alive_mask = np.isnan(death_days) | (death_days > day)  # Alive if no death or future death
        death_today_mask = (death_days == day)                   # Died today

        is_vaxed = (day >= first_dose_days) & has_any_dose      # Considered vaccinated today
        is_uvx = ~is_vaxed                                       # Otherwise unvaccinated

        pop_vx = np.sum(alive_mask & is_vaxed)
        pop_uvx = np.sum(alive_mask & is_uvx)
        pop_total = pop_vx + pop_uvx

        death_vx = np.sum(death_today_mask & is_vaxed)
        death_uvx = np.sum(death_today_mask & is_uvx)
        death_total = death_vx + death_uvx

        # Append values to results
        results['day'].append(day)
        results['age'].append(age)
        results['pop_vx'].append(pop_vx)
        results['pop_uvx'].append(pop_uvx)
        results['pop_total'].append(pop_total)
        results['death_vx'].append(death_vx)
        results['death_uvx'].append(death_uvx)
        results['death_total'].append(death_total)

# === Normalize and Smooth Death Rates ===

result_df = pd.DataFrame(results)

# Calculate raw and normalized differences between vx and uvx deaths
result_df['deathdiff_uvx_vx'] = result_df['death_uvx'] - result_df['death_vx']
result_df['death_vx_norm'] = (result_df['death_vx'] / result_df['pop_vx'].replace(0, np.nan)) * 100_000
result_df['death_uvx_norm'] = (result_df['death_uvx'] / result_df['pop_uvx'].replace(0, np.nan)) * 100_000
result_df['death_total_norm'] = (result_df['death_total'] / result_df['pop_total'].replace(0, np.nan)) * 100_000
result_df['deathdiff_uvx_vx_norm'] = result_df['death_uvx_norm'] - result_df['death_vx_norm']

# Fill NaNs (e.g., due to divide-by-zero) with 0
result_df.fillna(0, inplace=True)

# Apply 7-day centered rolling mean smoothing
window_size = 7
for col in [
    'death_vx_norm', 'death_uvx_norm', 'death_total_norm', 'deathdiff_uvx_vx_norm',
    'death_vx', 'death_uvx', 'death_total', 'deathdiff_uvx_vx'
]:
    result_df[col + '_smooth'] = result_df.groupby('age')[col].transform(
        lambda x: x.rolling(window_size, center=True, min_periods=1).mean()
    )

# === Count Doses Per Day ===

# Initialize count holders
first_dose_counts_age = {age: pd.Series(0, index=days, dtype=float) for age in ages}
all_dose_counts_age = {age: pd.Series(0, index=days, dtype=float) for age in ages}

# Count first and all doses for each day and age
for age, sub in zip(ages, df_age_groups):
    if sub.empty:
        continue

    # Count first doses
    first_counts = sub['first_dose_day'].value_counts().dropna().astype(int)
    s_first = pd.Series(0, index=days, dtype=float)
    s_first.update(first_counts)
    first_dose_counts_age[age] = s_first

    # Count all doses (flatten across all dose columns)
    all_dose_days = pd.concat([sub[col + '_day'] for col in dose_date_cols_lower])
    all_counts = all_dose_days.value_counts().dropna().astype(int)
    s_all = pd.Series(0, index=days, dtype=float)
    s_all.update(all_counts)
    all_dose_counts_age[age] = s_all

# Convert dictionaries to DataFrames
first_dose_df = pd.DataFrame(first_dose_counts_age)
all_dose_df = pd.DataFrame(all_dose_counts_age)

# Smooth dose counts
first_dose_df_smooth = first_dose_df.rolling(window=window_size, center=True, min_periods=1).mean()
all_dose_df_smooth = all_dose_df.rolling(window=window_size, center=True, min_periods=1).mean()

# === Plot with Plotly ===

fig = go.Figure()

# Define colors
colors_vx = 'rgba(0,100,255,0.3)'     # Blue
colors_uvx = 'rgba(255,0,0,0.3)'      # Red
colors_total = 'rgba(0,0,0,0.3)'      # Black
colors_diff = 'rgba(0,200,0,0.5)'     # Green

# Add smoothed normalized death traces per age
for age in ages:
    df_age = result_df[result_df['age'] == age]
    if df_age.empty:
        continue

    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_vx_norm_smooth'],
                             name=f'death_vx_norm_smooth age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_vx), visible='legendonly'))
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_uvx_norm_smooth'],
                             name=f'death_uvx_norm_smooth age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_uvx), visible='legendonly'))
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_total_norm_smooth'],
                             name=f'death_total_norm_smooth age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_total), visible='legendonly'))
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['deathdiff_uvx_vx_norm_smooth'],
                             name=f'deathdiff_uvx_vx_norm_smooth age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_diff), visible='legendonly'))

# === Final Layout and Save ===

fig.update_layout(
    title='Smoothed Death Rates per Age Group',
    xaxis=dict(title='Day since 2020-01-01'),
    yaxis=dict(title='Death Rate per 100,000'),
    legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5),
    height=800
)

# Save to HTML file
fig.write_html(OUTPUT_HTML)
print(f"Plot saved to {OUTPUT_HTML}")
