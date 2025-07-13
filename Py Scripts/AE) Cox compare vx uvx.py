import pandas as pd
import numpy as np
from lifelines import CoxTimeVaryingFitter
import warnings
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import sys

# ----------------------------
# Configuration parameters
# ----------------------------


# AA) 6 input CSV files (replace paths if needed)

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AF) case1_real_deaths_real_doses.csv"
#OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case1_real_deaths_real_doses.TXT"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AF) case2_sim_deaths_real_doses_no_constraint.csv"
#OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case2_sim_deaths_real_doses_no_constraint.TXT"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AF) case3_sim_deaths_sim_real_doses_with_constraint.csv"
#OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case3_sim_deaths_sim_real_doses_with_constraint.TXT"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AF) case4_sim_deaths_sim_real_doses_no_constraint.csv"
#OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case4_sim_deaths_sim_real_doses_no_constraint.TXT"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AF) case5_sim_deaths_sim_flat_random_doses_with_constraint.csv"
# OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case5_sim_deaths_sim_flat_random_doses_with_constraint.TXT"


# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AF) case6_sim_deaths_sim_flat_random_doses_no_constraint.csv"
# OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case6_sim_deaths_sim_flat_random_doses_no_constraint.TXT"


# AF) 10 cases

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AF) case1_real_deaths_real_doses.csv"
#OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case1_real_deaths_real_doses.TXT"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AF) case2_sim_deaths_real_doses_no_constraint.csv"
#OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case2_sim_deaths_real_doses_no_constraint.TXT"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AF) case3_sim_deaths_sim_real_doses_with_constraint.csv"
#OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case3_sim_deaths_sim_real_doses_with_constraint.TXT"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AF) case4_sim_deaths_sim_real_doses_no_constraint.csv"
# OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case4_sim_deaths_sim_real_doses_no_constraint.TXT"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AF) case5_sim_deaths_sim_flat_random_doses_with_constraint.csv"
#OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case5_sim_deaths_sim_flat_random_doses_with_constraint.TXT"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES\AF) case6_sim_deaths_sim_flat_random_doses_no_constraint.csv"
# OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case6_sim_deaths_sim_flat_random_doses_no_constraint.TXT"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\Terra\SIM_CASES\AF) case7_sim_deaths_sim_beelcurve_random_doses_with_constraint.csv"
# OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case7_sim_deaths_sim_beelcurve_random_doses_with_constraint.TXT"

#INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\Terra\SIM_CASES\AF) case8_sim_deaths_sim_beelcurve_random_doses_no_constraint.csv"
#OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case8_sim_deaths_sim_beelcurve_random_doses_no_constraint.TXT"

# INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\Terra\SIM_CASES\AF) case9_sim_deaths_sim_realcurve_random_doses_with_constraint.csv"
# OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case9_sim_deaths_sim_realcurve_random_doses_with_constraint.TXT"

INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\Terra\SIM_CASES\AF) case10_sim_deaths_sim_realcurve_random_doses_no_constraint.csv"
OUTPUT_TXT = r"C:\CzechFOI-DRATE_EXAM\Plot Results\AE) Cox compare vx uvx\AE-AF) case10_sim_deaths_sim_realcurve_random_doses_no_constraint.TXT"


REFERENCE_DATE = datetime(2021, 1, 1)
REFERENCE_YEAR = 2023

AG70 = 70
T_MAX = 1095
np.random.seed(42)



class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

# Save original stdout/stderr to restore later if needed
original_stdout = sys.stdout
original_stderr = sys.stderr

log_file_path = fr"{OUTPUT_TXT}"
log_file = open(log_file_path, "w", encoding="utf-8")

# Set up teeing
tee = Tee(sys.stdout, log_file)
sys.stdout = tee
sys.stderr = tee


# ----------------------------
# Helper function to convert absolute date string to relative day integer
# ----------------------------
def date_to_day(d):
    try:
        return (pd.to_datetime(d) - REFERENCE_DATE).days
    except Exception:
        return None


# ----------------------------
# Load data based on DOSE_SCHEDULE
# ----------------------------
if os.path.isfile(INPUT_CSV):

    # Read real data from CSV (like USE_REAL_DATA=True before)
    print(f"Loading real data from: {INPUT_CSV}")
    df_csv = pd.read_csv(INPUT_CSV, low_memory=False)
    df_csv["age"] = REFERENCE_YEAR - df_csv["Rok_narozeni"]
    df_csv = df_csv[df_csv["age"] == AG70].copy()
    print(f"Loaded {len(df_csv)} rows for age {AG70}")

    data = []
    for idx, row in df_csv.iterrows():
        person_id = idx
        vx_dates = [row[f"Datum_{i}"] for i in range(1, 8)]
        vx_days = sorted([date_to_day(d) for d in vx_dates if pd.notna(d)])
        death_day = date_to_day(row["DatumUmrti"]) if pd.notna(row["DatumUmrti"]) else None

        if not vx_days:
            data.append({
                'id': person_id,
                'start': 0,
                'stop': death_day if death_day is not None else T_MAX,
                'event': int(death_day is not None and death_day <= T_MAX),
                'vx': 0,
                'age': AG70
            })
        else:
            first_vx = vx_days[0]
            if death_day is not None and death_day <= first_vx:
                data.append({
                    'id': person_id,
                    'start': 0,
                    'stop': death_day,
                    'event': 1,
                    'vx': 0,
                    'age': AG70
                })
            else:
                data.append({
                    'id': person_id,
                    'start': 0,
                    'stop': first_vx,
                    'event': 0,
                    'vx': 0,
                    'age': AG70
                })
                # Change here: only assign vaccinated interval if death_day > first_vx day
                if death_day is None or death_day > first_vx:
                    data.append({
                        'id': person_id,
                        'start': first_vx,
                        'stop': death_day if death_day is not None else T_MAX,
                        'event': int(death_day is not None and death_day <= T_MAX),
                        'vx': 1,
                        'age': AG70
                    })

    df = pd.DataFrame(data)


# ----------------------------
# Diagnostics and summary
# ----------------------------
print("Summary of prepared data:")
print(df['vx'].value_counts())
print(df['event'].value_counts())

# ----------------------------
# Clean data: remove zero or negative duration intervals
# ----------------------------
df = df[df['stop'] > df['start']].copy()

# Add a baseline covariate column of 1.0 for model intercept
df['baseline'] = 1.0

# Define covariates for modeling
covariate_cols = ['vx', 'age', 'baseline']

# Convert covariates to numeric types, coercing errors to NaN
for col in covariate_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check and drop rows with missing values in covariates after conversion
if df[covariate_cols].isna().any().any():
    print("\nWarning: NaNs found in covariates after conversion. Dropping such rows.")
    df.dropna(subset=covariate_cols, inplace=True)

# Scale covariates to zero mean and unit variance before modeling
scaler = StandardScaler()
df[covariate_cols] = scaler.fit_transform(df[covariate_cols])

# ----------------------------
# Diagnostic printouts to verify data integrity
# ----------------------------
print("\nData snapshot:")
print(df.head(10))

print("\nCovariate unique values (scaled):")
for col in covariate_cols:
    print(f"{col}: min={df[col].min():.3f}, max={df[col].max():.3f}")

print("\nEvent counts and total per vx group:")
print(df.groupby('vx')['event'].agg(['sum', 'count']))

print("\nMissing values per column:")
print(df.isna().sum())

print("\nChecking for duplicated intervals (id, start, stop):")
duplicates = df.duplicated(subset=['id', 'start', 'stop'])
print(f"Number of duplicated intervals: {duplicates.sum()}")

print("\nChecking for zero-duration intervals:")
zero_duration = (df['stop'] - df['start']) == 0
print(f"Number of zero-duration intervals: {zero_duration.sum()}")

print("\nEvent rate by vaccination status:")
event_counts = df.groupby('vx')['event'].sum()
total_counts = df.groupby('vx').size()
event_rates = (event_counts / total_counts).to_frame('event_rate')
print(event_rates)

# ----------------------------
# Fit Cox proportional hazards model with time-varying covariate 'vx'
# Try several penalizer values for numerical stability
# ----------------------------
penalizers_to_try = [0.01, 0.1, 1, 10]
success = False

for penalizer in penalizers_to_try:
    try:
        print(f"\nTrying CoxTimeVaryingFitter with penalizer={penalizer} and formula='vx'")
        ctv = CoxTimeVaryingFitter(penalizer=penalizer)
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            # Fit model using id, start, stop, event columns and formula for covariates
            ctv.fit(df, id_col='id', start_col='start', stop_col='stop', event_col='event',
                    show_progress=True, formula="vx")
        ctv.print_summary()
        success = True
        break
    except Exception as e:
        print(f"Failed with penalizer={penalizer}: {e}")

if not success:
    print("All attempts to fit CoxTimeVaryingFitter failed. Check data integrity.")

# close logging console and restore original streams at end
sys.stdout = original_stdout
sys.stderr = original_stderr
log_file.close()


