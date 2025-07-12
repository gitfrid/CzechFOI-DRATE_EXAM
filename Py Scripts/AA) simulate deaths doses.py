import pandas as pd
import numpy as np
import os

# This simulation shows how the timing of vaccination creates a "non-random boundary condition" bias, making vaccines appear more effective than they actually are.
# By randomizing deaths or vaccination dates, we can test whether this bias is always—unavoidably—present in real observational data

# Only a few scientists seem to be aware of this statistical artefact,
# and widely used methods such as Cox regression or Kaplan-Meier analysis do not correct for it - which often leads to misleading results.

# Non-random boundary condition bias is distinct from better-known biases 
# such as healthy user bias, treatment lag bias, immortal time bias, and selection bias, though it may overlap with them in effect

# Charles Sanders Peirce recognized over a century ago that improper randomization and boundary conditions could distort conclusions
# a warning still relevant in today’s observational vaccine-effect studies

# === CONFIGURABLE CONSTANTS ===
INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\Vesely_106_202403141131.csv"  # Input data file
OUTPUT_FOLDER = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES"                # Output folder

START_DATE = pd.Timestamp('2020-01-01')                         # Reference day 0
REFERENCE_YEAR = 2023                                           # Reference year for age calculation
MAX_AGE = 113                                                   # Maximum age considered
DOSE_DATE_COLS = [f'Datum_{i}' for i in range(1, 8)]            # Column names for dose dates
NEEDED_COLS = ['Rok_narozeni', 'DatumUmrti'] + DOSE_DATE_COLS   # Required columns from input

FLAT_START_DAY = 350                            # Flat random dose start day (relative to START_DATE)
FLAT_END_DAY = 500                              # Flat random dose end day

RETRIES = 10000                                 # Max retries for constraint-based dose assignment
BASE_RNG_SEED = 42                              # Base seed for reproducible randomness
SEL_AGE = 70                                    # Single age selected for simulation

np.random.seed(BASE_RNG_SEED)

# === UTILITIES ===

def to_day_number(date_series):
    # Convert datetime series to day numbers since START_DATE.
    date_series = pd.to_datetime(date_series, errors='coerce')
    return (date_series - START_DATE).dt.days

def parse_dates(df):
    # Parse all relevant columns to datetime.
    for col in DOSE_DATE_COLS + ['DatumUmrti']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def calculate_age(df):
    # Calculate age at REFERENCE_YEAR.
    df["Age"] = REFERENCE_YEAR - df["Rok_narozeni"].astype(int)
    return df

def estimate_death_rate(df):
    # df is for a single age only
    deaths = df["DatumUmrti"]
    death_rate = np.clip(deaths.notna().sum() / len(deaths), 1e-4, 0.999)
    return death_rate  # return just a float, not a dict


def simulate_deaths(df, end_measure, death_rate):
    # Simulate deaths using estimated probability for the single age group.
    df = df.copy()
    df['DatumUmrti'] = pd.NaT
    df['death_day'] = np.nan

    n = len(df)
    will_die = np.random.rand(n) < death_rate
    death_days = np.full(n, np.nan)
    death_days[will_die] = np.random.randint(0, end_measure + 1, size=will_die.sum())

    df.loc[:, "death_day"] = death_days
    df.loc[will_die, "DatumUmrti"] = START_DATE + pd.to_timedelta(death_days[will_die], unit='D')

    return df


# === DOSE ASSIGNMENT FUNCTIONS ===

def assign_doses_per_age(dose_sets, death_day_arr, rng_seed, constrained):
    # Assign doses randomly to individuals, optionally only to those alive after last dose day.
    rng = np.random.default_rng(rng_seed)
    updates = []           # store (index, dose_dates) assignments
    skip_count = 0         # count how many doses could not be assigned

    if len(dose_sets) == 0 or len(death_day_arr) == 0:
        return updates, skip_count

    vax_stat_arr = np.zeros(len(death_day_arr), dtype=np.int8)  # mark assigned individuals (0=not assigned,1=assigned)

    for dose_dates in dose_sets:
        # Extract valid (non-NaT) dose dates
        valid_dates = [d for d in dose_dates if pd.notna(d)]
        if not valid_dates:
            continue

        # Convert valid dose dates to day numbers (int), get last dose day
        valid_days = np.array(to_day_number(pd.Series(valid_dates)))
        last_dose_day = valid_days.max()

        # Find indices of individuals not yet assigned doses
        eligible_mask = (vax_stat_arr == 0)
        eligible_indices = np.where(eligible_mask)[0]
        if eligible_indices.size == 0:
            skip_count += 1
            continue

        selected_pos = None
        if constrained:
            # Try multiple times to assign dose only if person survives past last dose day
            for _ in range(RETRIES):
                trial_pos = rng.choice(eligible_indices)
                if np.isnan(death_day_arr[trial_pos]) or death_day_arr[trial_pos] > last_dose_day:
                    selected_pos = trial_pos
                    break
            if selected_pos is None:
                skip_count += 1
        else:
            # Unconstrained: assign randomly without checking death day
            selected_pos = rng.choice(eligible_indices)

        if selected_pos is not None:
            updates.append((selected_pos, dose_dates))  # record assignment
            vax_stat_arr[selected_pos] = 1              # mark as assigned
        else:
            skip_count += 1

    return updates, skip_count


def assign_doses_real_curve_random(df_target, df_source, constrained):
    # Assign real dose curves from df_source to df_target, respecting constraints if set.
    df_target = df_target.copy()
    df_target[DOSE_DATE_COLS] = pd.NaT  # reset dose date columns in target

    # Extract dose sets (list of dose date lists) from source
    dose_sets = df_source[DOSE_DATE_COLS].dropna(how='all').values.tolist()
    death_day_arr = df_target["death_day"].to_numpy()
    rng_seed = BASE_RNG_SEED + SEL_AGE  # seed rng with base + age for reproducibility

    # Assign doses per age group with optional constraint
    updates, skip_count = assign_doses_per_age(dose_sets, death_day_arr, rng_seed, constrained)

    group_idx = df_target.index.to_numpy()
    for pos, dose_dates in updates:
        row_idx = group_idx[pos]
        # Update dose dates in target dataframe for assigned individual
        for j, d in enumerate(dose_dates):
            if pd.notna(d):
                df_target.at[row_idx, DOSE_DATE_COLS[j]] = pd.Timestamp(d)

    print(f"Assigned {len(updates)} doses, Skipped {skip_count} doses (constrained={constrained})")
    return df_target


def assign_doses_flat_curve_random(df_target, df_source, start_day, end_day, constrained):
    # Assign flat random first doses (only first dose) matching total real doses, with optional constraint.
    df_target = df_target.copy()
    df_target[DOSE_DATE_COLS] = pd.NaT  # reset dose date columns

    # Count total real first doses in source data for this age
    total_doses = df_source[DOSE_DATE_COLS[0]].notna().sum()

    num_days = end_day - start_day + 1
    rng = np.random.default_rng(BASE_RNG_SEED)

    age_arr = df_target["Age"].to_numpy()
    death_day_arr = df_target["death_day"].to_numpy()
    dose_date_arr = df_target[DOSE_DATE_COLS[0]].to_numpy(dtype="datetime64[ns]")
    assigned_mask = pd.isna(dose_date_arr)  # True where dose not assigned yet

    # Eligible indices for assignment: correct age and dose not assigned
    eligible_indices = np.where((age_arr == SEL_AGE) & assigned_mask)[0]
    rng.shuffle(eligible_indices)  # shuffle eligible indices randomly

    pos = 0  # pointer in eligible indices
    doses_per_day = total_doses // num_days
    remainder = total_doses % num_days

    assigned_total = 0
    skipped_total = 0

    # Assign doses evenly over the days (with remainder handled in first days)
    for i, day in enumerate(range(start_day, end_day + 1)):
        if assigned_total >= total_doses:
            break

        n_to_assign = doses_per_day + (1 if i < remainder else 0)
        assigned_this_day = 0

        for _ in range(n_to_assign):
            found = False
            retries = 0
            # Find eligible individual who survives past dose day if constrained
            while pos < len(eligible_indices) and retries < RETRIES:
                idx = eligible_indices[pos]
                retries += 1
                pos += 1
                if (not constrained) or (np.isnan(death_day_arr[idx]) or death_day_arr[idx] > day):
                    # Assign dose date as START_DATE + day offset
                    dose_date_arr[idx] = START_DATE + pd.to_timedelta(day, unit='D')
                    assigned_mask[idx] = False  # mark assigned
                    assigned_total += 1
                    assigned_this_day += 1
                    found = True
                    break
            if not found:
                skipped_total += 1

    df_target[DOSE_DATE_COLS[0]] = dose_date_arr  # update target df with assigned first doses
    print(f"Assigned {assigned_total}/{total_doses} flat random doses, Skipped {skipped_total} (constrained={constrained})")
    return df_target

# === OUTPUT FORMATTING ===

def format_and_save(df, out_path):
    # Format date columns and export to CSV.
    for col in DOSE_DATE_COLS + ['DatumUmrti']:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d').fillna('')
    df.to_csv(out_path, index=False)

def save_case(df, filename):
    out_path = os.path.join(OUTPUT_FOLDER, filename)
    format_and_save(df, out_path)
    print(f"Saved: {out_path}")
    return out_path

# === MAIN PROCESS ===

def run_all_cases():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("📥 Loading full data...")
    df_full = pd.read_csv(INPUT_CSV, usecols=NEEDED_COLS, dtype=str)
    # Parse dates and calculate age before filtering by age
    df_full = parse_dates(df_full)
    df_full = calculate_age(df_full)
    
    # Determine max observation window in days
    max_death_day = to_day_number(df_full["DatumUmrti"]).max()
    END_MEASURE = max_death_day if not np.isnan(max_death_day) else 1533  # fallback
    print(f"Measurement window (END_MEASURE): {END_MEASURE} days")

    # Filter for selected age only
    df_age = df_full[df_full["Age"] == SEL_AGE].copy()
    print(f"Filtered data to Age={SEL_AGE}, total rows = {len(df_age)}")

    # Simulate deaths based on age death rate
    death_rate = estimate_death_rate(df_age)  # Estimate death rate *for the filtered age only*
    df_age_sim_deaths = simulate_deaths(df_age, end_measure=END_MEASURE, death_rate=death_rate)

    # Case 1: Real deaths, real doses (no reassignment, filtered by age)
    df_case1 = df_age.copy()
    out_case1 = save_case(df_case1, "AA) case1_real_deaths_real_doses.csv")    
    print(f"CASE 1 saved: {out_case1}")

    # Case 2: Sim deaths, real doses, NO constraint      
    df_case2 = df_age_sim_deaths.copy()
    out_case2 = save_case(df_case2, "AA) case2_sim_deaths_real_doses_no_constraint.csv")
    print(f"CASE 2 saved: {out_case2}")

    # Case 3: Sim deaths, simulated doses (real schedule), WITH constraint (death_day > last dose day)
    df_case3 = df_age_sim_deaths.copy()
    df_case3 = assign_doses_real_curve_random(df_case3, df_age, constrained=True)
    out_case3 = save_case(df_case3, "AA) case3_sim_deaths_sim_real_doses_with_constraint.csv")
    print(f"CASE 3 saved: {out_case3}")

    # Case 4: Sim deaths, simulated doses (real schedule), NO constraint
    df_case4 = df_age_sim_deaths.copy()
    df_case4 = assign_doses_real_curve_random(df_case4, df_age, constrained=False)
    out_case4 = save_case(df_case4, "AA) case4_sim_deaths_sim_real_doses_no_constraint.csv")
    print(f"CASE 4 saved: {out_case4}")

    # Case 5: Sim deaths, sim flat-random doses - same number as real, WITH constraint (death_day > dose day)
    df_case5 = df_age_sim_deaths.copy()
    df_case5 = assign_doses_flat_curve_random(df_case5, df_age, FLAT_START_DAY, FLAT_END_DAY, constrained=True)
    out_case5 = save_case(df_case5, "AA) case5_sim_deaths_sim_flat_random_doses_with_constraint.csv")
    print(f"CASE 5 saved: {out_case5}")

    # Case 6: Sim deaths, sim flat-random doses - same number as real, NO constraint
    df_case6 = df_age_sim_deaths.copy()
    df_case6 = assign_doses_flat_curve_random(df_case6, df_age, FLAT_START_DAY, FLAT_END_DAY, constrained=False)
    out_case6 = save_case(df_case6, "AA) case6_sim_deaths_sim_flat_random_doses_no_constraint.csv")
    print(f"CASE 6 saved: {out_case6}")


    print("✅ All cases processed and saved.")

if __name__ == "__main__":
    run_all_cases()
