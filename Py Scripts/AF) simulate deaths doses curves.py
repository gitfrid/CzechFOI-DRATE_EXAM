import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

# This simulation shows how the timing of vaccination creates a "non-random boundary condition" bias, making vaccines appear more effective than they actually are.
# By randomizing deaths or vaccination dates, we can test whether this bias is always—unavoidably—present in real observational data

# Only a few scientists seem to be aware of this statistical artefact,
# and widely used methods such as Cox regression or Kaplan-Meier analysis do not correct for it - which often leads to misleading results.

# Non-random boundary condition bias is distinct from better-known biases 
# such as healthy user bias, treatment lag bias, immortal time bias, and selection bias, though it may overlap with them in effect

# Charles Sanders Peirce recognized over a century ago that improper randomization and boundary conditions could distort conclusions
# a warning still relevant in today’s observational vaccine-effect studies

# Simulation of Boundary Condition Bias in Observational Vaccine Studies
# -----------------------------------------------------------------------

# This script demonstrates how non-random boundary conditions—arising from the timing of vaccination relative to death events—
# can introduce a **systematic statistical bias** in observational vaccine effectiveness studies. Even when deaths or vaccinations 
# are randomly distributed, the imposed survival constraints (e.g., needing to be alive to receive a dose) can make vaccines appear 
# more protective than they are.

# Key Objectives:
#---------------
# 1. Simulate a synthetic population (single age group) based on real Czech vaccine registry data.
# 2. Randomly assign deaths over a fixed follow-up period using empirically estimated constant homogen death rates.
# 3. Assign vaccination schedules to individuals in two ways:
#    - Randomized from real-world dose curves (with or without survival constraints).
#    - Generated dose curves (flat or bell-shaped) over a specified range.
# 4. Compare how the choice of dose assignment and timing affects the apparent association between vaccination and death.

# Case 3 Highlight:
# -----------------
# In Case 3  we enforce a strong constraint: death_day > last_dose_day
# Individuals are only included if their death occurs after their latest vaccine dose (e.g., Dose1 to Dose7)
# This amplifies the bias dramatically — (simply by how they're selected), creating a huge artificial survival benefit

# === CONFIGURABLE CONSTANTS ===
INPUT_CSV = r"C:\CzechFOI-DRATE_EXAM\TERRA\Vesely_106_202403141131.csv"  # Input data file
OUTPUT_FOLDER = r"C:\CzechFOI-DRATE_EXAM\TERRA\SIM_CASES"                # Output folder

START_DATE = pd.Timestamp('2020-01-01')                         # Reference day 0
REFERENCE_YEAR = 2023                                           # Reference year for age calculation
MAX_AGE = 113                                                   # Maximum age considered
DOSE_DATE_COLS = [f'Datum_{i}' for i in range(1, 8)]            # Column names for dose dates
NEEDED_COLS = ['Rok_narozeni', 'DatumUmrti'] + DOSE_DATE_COLS   # Required columns from input

DOSE_START_DAY = 460                            # Dose start day (relative to START_DATE) for simulated curve window (flat,bell,real)  
DOSE_END_DAY = 530                              # Dose end day for simulated curve window (flat,bell,real)  

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

def assign_doses_per_age(dose_sets, death_day_arr, rng_seed, constrained, retries=10):
    
    #Randomly assigns dose date sets to individuals in df_target.
    #If `constrained` is True, ensures each assigned individual survives beyond the last dose day.
    #Returns a list of (index, dose_dates) updates and a skip count.
    
    rng = np.random.default_rng(rng_seed)
    updates = []
    skip_count = 0

    if len(dose_sets) == 0 or len(death_day_arr) == 0:
        return updates, skip_count

    vax_stat_arr = np.zeros(len(death_day_arr), dtype=np.int8)

    for dose_dates in dose_sets:
        valid_dates = [d for d in dose_dates if pd.notna(d)]
        if not valid_dates:
            continue

        valid_days = np.array(to_day_number(pd.Series(valid_dates)))
        last_dose_day = valid_days.max()

        eligible_indices = np.where(vax_stat_arr == 0)[0]
        if eligible_indices.size == 0:
            skip_count += 1
            continue

        rng.shuffle(eligible_indices)
        selected_pos = None

        if constrained:
            trial_pool = rng.choice(eligible_indices, size=min(retries, len(eligible_indices)), replace=False)
            for trial_pos in trial_pool:
                # constaint death day > first dose day
                if np.isnan(death_day_arr[trial_pos]) or death_day_arr[trial_pos] > valid_days.min():
                # constaint death day > last dose day
                #if np.isnan(death_day_arr[trial_pos]) or death_day_arr[trial_pos] > last_dose_day:
                    selected_pos = trial_pos
                    break
        else:
            selected_pos = rng.choice(eligible_indices)

        if selected_pos is not None:
            # Store the whole sequence dose1-7 dose dates, in a list to match expected structure
            updates.append((selected_pos, dose_dates))
            # Store only the first valid dose date, in a list to match expected structure
            # updates.append((selected_pos, [valid_dates[0]]))            
            vax_stat_arr[selected_pos] = 1
        else:
            skip_count += 1

    return updates, skip_count


def assign_doses_real_curve_random(df_target, df_source, constrained, retries=10):
    # Assigns real-world dose curves from df_source to df_target individuals.
    # Dose dates are assigned randomly, optionally constrained by survival after last dose.
    
    df_target = df_target.copy()
    df_target[DOSE_DATE_COLS] = pd.NaT  # Reset all dose date columns

    dose_sets = df_source[DOSE_DATE_COLS].dropna(how='all').values.tolist()
    death_day_arr = df_target["death_day"].to_numpy()
    rng_seed = BASE_RNG_SEED + SEL_AGE  # Customize per age for reproducibility

    updates, skip_count = assign_doses_per_age(dose_sets, death_day_arr, rng_seed, constrained, retries=retries)

    for pos, dose_dates in updates:
        for j, d in enumerate(dose_dates):
            if pd.notna(d):
                df_target.iloc[pos, df_target.columns.get_loc(DOSE_DATE_COLS[j])] = pd.Timestamp(d)

    print(f"Assigned {len(updates)} doses, Skipped {skip_count} doses (constrained={constrained})")
    return df_target



def generate_flat_dose_curve(df_real, dose_col, start_day, end_day):
    
    # Generate a flat (uniform) dose schedule from start_day to end_day,
    # based on the number of non-null entries in a real dose column.

    # Parameters:
    #     df_real (pd.DataFrame): DataFrame containing real dose data.
    #    dose_col (str): Column name containing dose dates.
    #    start_day (int): Start day of dose rollout.
    #    end_day (int): End day of dose rollout.

    # Returns:
    #    np.array of daily dose counts (length = end_day - start_day + 1)
    
    total_doses = df_real[dose_col].notna().sum()
    num_days = end_day - start_day + 1
    doses_schedule = [total_doses // num_days] * num_days
    for i in range(total_doses % num_days):
        doses_schedule[i] += 1
    return np.array(doses_schedule)

import numpy as np
from scipy.stats import norm

def generate_bell_dose_curve(df_real, dose_col, start_day, end_day, std_factor=0.2):
    
    # Generate a bell-shaped (Gaussian-like) dose schedule from start_day to end_day,
    # centered in the middle of the range.

    # Parameters:
    #    df_real (pd.DataFrame): DataFrame containing real dose data.
    #    dose_col (str): Column name containing dose dates.
    #    start_day (int): Start day of dose rollout.
    #    end_day (int): End day of dose rollout.
    #    std_factor (float): Width of the bell curve (as fraction of duration, default=0.2).

    # Returns:
    #     np.array of daily dose counts (length = end_day - start_day + 1)
    
    total_doses = df_real[dose_col].notna().sum()
    num_days = end_day - start_day + 1
    days = np.arange(num_days)

    # Center the bell in the middle of the period
    mean = num_days // 2
    std = num_days * std_factor

    # Create a normalized bell curve (probability distribution)
    bell_curve = norm.pdf(days, loc=mean, scale=std)
    bell_curve /= bell_curve.sum()  # Normalize to sum to 1

    # Scale to match total_doses
    dose_counts = (bell_curve * total_doses).round().astype(int)

    # Adjust to make sure the total exactly matches (due to rounding)
    diff = total_doses - dose_counts.sum()
    if diff > 0:
        dose_counts[:diff] += 1
    elif diff < 0:
        dose_counts[:abs(diff)] -= 1

    return dose_counts


def generate_real_dose_curve(df_real, dose_col, start_day, end_day):
    
    # Generate a real dose schedule by counting actual doses from the real data
    #between start_day and end_day.

    # Parameters:
    #    df_real (pd.DataFrame): DataFrame containing real dose data.
    #    dose_col (str): Column name containing dose days (integers).
    #    start_day (int): Start day of dose rollout.
    #    end_day (int): End day of dose rollout.

    #Returns:
    #    np.array of daily dose counts (length = end_day - start_day + 1)
    
    # Filter dose dates within the desired range
    doses_in_range = df_real[dose_col].dropna()
    # Convert to day numbers relative to START_DATE
    doses_in_range = to_day_number(doses_in_range)
    doses_in_range = doses_in_range[(doses_in_range >= start_day) & (doses_in_range <= end_day)]

    # Count number of doses per day
    daily_counts = doses_in_range.value_counts().sort_index()

    # Create array of dose counts for each day in the range
    num_days = end_day - start_day + 1
    dose_schedule = np.zeros(num_days, dtype=int)

    for day, count in daily_counts.items():
        dose_schedule[day - start_day] = count

    return dose_schedule


def assign_doses_curve_random_constrained(df_target, daily_doses, start_day, constrained=True, seed=BASE_RNG_SEED):
    
    # Assign dose dates based on a daily dose curve, with optional death-day constraint.

    # Parameters:
    #     df_target (pd.DataFrame): Target individuals to assign doses.
    #     daily_doses (np.array): Doses per day, starting from start_day.
    #     start_day (int): Offset to start assigning.
    #     constrained (bool): If True, only assign to people not dead before dose.
    #     seed (int): RNG seed.

    # Returns:
    #     pd.DataFrame: With assigned dose dates in DOSE_DATE_COLS[0].
    
    df_target = df_target.copy()
    df_target[DOSE_DATE_COLS] = pd.NaT

    rng = np.random.default_rng(seed)
    total_doses = daily_doses.sum()

    death_day_arr = df_target["death_day"].to_numpy()
    dose_date_arr = df_target[DOSE_DATE_COLS[0]].to_numpy(dtype="datetime64[ns]")
    assigned_mask = pd.isna(dose_date_arr)
    eligible_indices = np.where(assigned_mask)[0]

    assigned_total = 0
    skipped_total = 0

    for offset, doses_today in enumerate(daily_doses):
        day = start_day + offset
        if assigned_total >= total_doses or len(eligible_indices) == 0:
            break

        rng.shuffle(eligible_indices)

        for _ in range(doses_today):
            if len(eligible_indices) == 0:
                break

            selected_pos = None
            if constrained:
                for _ in range(RETRIES):
                    trial_pos = rng.choice(eligible_indices)
                    if np.isnan(death_day_arr[trial_pos]) or death_day_arr[trial_pos] > day:
                        selected_pos = trial_pos
                        break
            else:
                selected_pos = rng.choice(eligible_indices)

            if selected_pos is not None:
                dose_date_arr[selected_pos] = START_DATE + pd.to_timedelta(day, unit='D')
                assigned_mask[selected_pos] = False
                eligible_indices = eligible_indices[assigned_mask[eligible_indices]]
                assigned_total += 1
            else:
                skipped_total += 1

    df_target[DOSE_DATE_COLS[0]] = dose_date_arr
    print(f"Assigned ~{assigned_total}/{total_doses} flat random doses, Skipped {skipped_total} (constrained={constrained})")
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
    out_case1 = save_case(df_case1, "AF) case1_real_deaths_real_doses.csv")    
    print(f"CASE 1 saved: {out_case1}")

    # Case 2: Sim deaths, real doses, NO constraint      
    df_case2 = df_age_sim_deaths.copy()
    out_case2 = save_case(df_case2, "AF) case2_sim_deaths_real_doses_no_constraint.csv")
    print(f"CASE 2 saved: {out_case2}")

    # Case 3: Sim deaths, simulated doses (real schedule), WITH constraint (death_day > last dose day)
    df_case3 = df_age_sim_deaths.copy()
    df_case3 = assign_doses_real_curve_random(df_case3, df_age, constrained=True)
    out_case3 = save_case(df_case3, "AF) case3_sim_deaths_sim_real_doses_with_constraint.csv")
    print(f"CASE 3 saved: {out_case3}")

    # Case 4: Sim deaths, simulated doses (real schedule), NO constraint
    df_case4 = df_age_sim_deaths.copy()
    df_case4 = assign_doses_real_curve_random(df_case4, df_age, constrained=False)
    out_case4 = save_case(df_case4, "AF) case4_sim_deaths_sim_real_doses_no_constraint.csv")
    print(f"CASE 4 saved: {out_case4}")

    # Case 5: Sim deaths, sim flat curve for doses - same number as real, WITH constraint (death_day > dose day)
    df_case5 = df_age_sim_deaths.copy()
    dose_curve = generate_flat_dose_curve(df_age, DOSE_DATE_COLS[0], DOSE_START_DAY, DOSE_END_DAY)
    df_case5 = assign_doses_curve_random_constrained(df_target=df_case5,daily_doses=dose_curve,start_day=DOSE_START_DAY, constrained=True )
    out_case5 = save_case(df_case5, "AF) case5_sim_deaths_sim_flat_random_doses_with_constraint.csv")
    print(f"CASE 5 saved: {out_case5}")
    

    # Case 6: Sim deaths, sim flat-random doses - same number as real, NO constraint
    df_case6 = df_age_sim_deaths.copy()
    dose_curve = generate_flat_dose_curve(df_age, DOSE_DATE_COLS[0], DOSE_START_DAY, DOSE_END_DAY)
    df_case6 = assign_doses_curve_random_constrained(df_target=df_case6,daily_doses=dose_curve,start_day=DOSE_START_DAY, constrained=False )
    out_case6 = save_case(df_case6, "AF) case6_sim_deaths_sim_flat_random_doses_no_constraint.csv")
    print(f"CASE 6 saved: {out_case6}")


    # Case 7: Sim deaths, sim bell curve for doses - same number as real, WITH constraint (death_day > dose day)
    df_case7 = df_age_sim_deaths.copy()
    dose_curve = generate_bell_dose_curve(df_age, DOSE_DATE_COLS[0], DOSE_START_DAY, DOSE_END_DAY)
    df_case7 = assign_doses_curve_random_constrained(df_target=df_case7,daily_doses=dose_curve,start_day=DOSE_START_DAY, constrained=True )
    out_case7 = save_case(df_case7, "AF) case7_sim_deaths_sim_beelcurve_random_doses_with_constraint.csv")
    print(f"CASE 7 saved: {out_case7}")


    # Case 8: Sim deaths, sim bell curve for doses - same number as real, NO  constraint
    df_case8 = df_age_sim_deaths.copy()
    dose_curve = generate_bell_dose_curve(df_age, DOSE_DATE_COLS[0], DOSE_START_DAY, DOSE_END_DAY)
    df_case8 = assign_doses_curve_random_constrained(df_target=df_case8,daily_doses=dose_curve,start_day=DOSE_START_DAY, constrained=False )
    out_case8 = save_case(df_case8, "AF) case8_sim_deaths_sim_beelcurve_random_doses_no_constraint.csv")
    print(f"CASE 8 saved: {out_case8}")

    # Case 9: Sim deaths, sim real curve for doses - same number as real, WITH constraint (death_day > dose day)
    df_case9 = df_age_sim_deaths.copy()
    dose_curve = generate_real_dose_curve(df_age, DOSE_DATE_COLS[0], DOSE_START_DAY, DOSE_END_DAY)
    df_case9 = assign_doses_curve_random_constrained(df_target=df_case9,daily_doses=dose_curve,start_day=DOSE_START_DAY, constrained=True )
    out_case9 = save_case(df_case9, "AF) case9_sim_deaths_sim_realcurve_random_doses_with_constraint.csv")
    print(f"CASE 9 saved: {out_case9}")

    # Case 10: Sim deaths, sim real curve for doses - same number as real, NO constraint
    df_case10 = df_age_sim_deaths.copy()
    dose_curve = generate_real_dose_curve(df_age, DOSE_DATE_COLS[0], DOSE_START_DAY, DOSE_END_DAY)
    df_case10 = assign_doses_curve_random_constrained(df_target=df_case10,daily_doses=dose_curve,start_day=DOSE_START_DAY, constrained=False )
    out_case10 = save_case(df_case10, "AF) case10_sim_deaths_sim_realcurve_random_doses_no_constraint.csv")
    print(f"CASE 10 saved: {out_case10}")

    
    print("✅ All cases processed and saved.")

if __name__ == "__main__":
    run_all_cases()
