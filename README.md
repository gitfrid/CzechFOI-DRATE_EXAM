# CzechFOI-DRATE_EXAM

## Investigation of the Non-Random Boundary Condition Bias

The **non-random boundary condition bias** arises when a homogeneous group is **divided into subgroups based on a non-random time point**—as often happens in real observational studies—and these subgroups are then compared.

This lead to **illusory differences between the groups**, even when **no real causal effect** exists.

Unlike more widely recognized biases such as: Immortal time bias, Survivorship Bias, Healthy user effect, Selection bias, Treatment delay bias  
...this bias is often **overlooked**, yet it has inevitable a **major impact** on the outcomme of observational data.

Only few scientists seem to be aware of this statistical artefact,
and widely used scientific methods such as Cox regression or Kaplan-Meier analysis seems not correct for it - which often leads to misleading results.

**Charles Sanders Peirce recognized over a century ago that improper randomization and selection conditions distort conclusions,
a warning still relevant in today**

[**CzechFOI-DRATE project for mor information**](https://github.com/gitfrid/CzechFOI-DRATE)
_________________________________________

### AA) simulate deaths doses.py

The simulation script demonstrates how applying the "alive at dose time" condition introduces a systematic bias — even when deaths are assigned completely at random, and doses are subsequently assigned only to individuals who meet the boundary condition (i.e., no causal effect of the vaccine is simulated — the null hypothesis).


<br><br>**Case 3 illustrates this clearly: the unvaccinated group appears to die at a much higher rate, despite no difference in the underlying, constant death risk between vaccinated and unvaccinated individuals. This is an example of bias, where the requirement to be alive at the time of dosing creates an illusion of vaccine effectiveness.**

<br>Phyton script [AA) simulate deaths doses.py](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Py%20Scripts/AA%29%20simulate%20deaths%20doses.py) 

_________________________________________

| Case | Deaths       | Doses        | ** Condition (only assign if alive at dose day)   | Expected Bias   |
|-------|--------------|--------------|------------------------------------------------|-----------------|
| 1     | Real Data&nbsp;&nbsp;&nbsp;   | Real Data&nbsp;&nbsp;&nbsp;   | Inherent applied (real doses can only happen if alive)   | ✅ Bias Present  |
| 2     | Simulated&nbsp;&nbsp;&nbsp;   | Real Data&nbsp;&nbsp;&nbsp;   | Not applied (assigned real doses regardless if alive)           | ❌ No Bias      |
| 3     | Simulated&nbsp;&nbsp; | Probabilistic from real curve&nbsp;&nbsp;&nbsp;   | ✅ Applied                                       | ✅ Bias Present  |
| 4     | Simulated&nbsp;&nbsp; | Probabilistic from real curve&nbsp;&nbsp;&nbsp;   | ❌ Not applied                                   | ❌ No Bias      |
| 5     | Simulated&nbsp;&nbsp;&nbsp;   | Simulate rectangular curve&nbsp;&nbsp; | ✅ Applied                                       | ✅ Bias Present  |
| 6     | Simulated&nbsp;&nbsp;&nbsp;   | Simulate rectangular curve&nbsp;&nbsp; | ❌ Not applied                                   | ❌ No Bias      |

****Boundary condition case3: death day > last dose day**
<br>****Boundary condition case5: death day > first  dose day**


# Simulation Cases Visualization

**Below are the result plots from six different simulation cases, for age group 70:**

- Kaplan-Meier Survival Curves comparing Total, Vaccinated, and Unvaccinated  
- Normalized Deaths, Population, and Doses for Total, Vaccinated, and Unvaccinated

---

**Case 3:**  
Deaths After Last Dose (Real-Life Constraint)

In this simulated case, deaths are only allowed to occur after the individual's last recorded vaccine dose, based on the 7 real dose columns.

This mimics real-life data behavior:  
If someone has a recorded 3rd dose, they must have been alive at least until that dose date.

---

**The Problem:**

This introduces immortal time bias, giving an unfair advantage to the vaccinated group.

---

**Effect:**

- Kaplan-Meier and Normalized Time-Series curves show fewer deaths in vaccinated  
- Cox regression shows a significant strong protective effect, but this effect is biased and misleading



| Case&nbsp;&nbsp;&nbsp; | Raw Time Series (CA-AA) | Normalized Time Series (ZI-AA) |
|-------|-------------------------|-------------------------------|
| 1&nbsp;&nbsp;&nbsp; | ![Case 1 Raw](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Plot%20Results/CA%29%20KM%20vx%20uvx/CA-AA%29%20case1_real_deaths_real_doses.png) | ![Case 1 Normalized](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Plot%20Results/ZI%29%20vx%20uvx%20norm/ZI-AA%29%20case1_real_deaths_real_doses.png) |
| 2&nbsp;&nbsp;&nbsp; | ![Case 2 Raw](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Plot%20Results/CA%29%20KM%20vx%20uvx/CA-AA%29%20case2_sim_deaths_real_doses_no_constraint.png) | ![Case 2 Normalized](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Plot%20Results/ZI%29%20vx%20uvx%20norm/ZI-AA%29%20case2_sim_deaths_real_doses_no_constraint.png) |
| 3&nbsp;&nbsp;&nbsp; | ![Case 3 Raw](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Plot%20Results/CA%29%20KM%20vx%20uvx/CA-AA%29%20case3_sim_deaths_sim_real_doses_with_constraint.png) | ![Case 3 Normalized](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Plot%20Results/ZI%29%20vx%20uvx%20norm/ZI-AA%29%20case3_sim_deaths_sim_real_doses_with_constraint.png) |
| 4&nbsp;&nbsp;&nbsp; | ![Case 4 Raw](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Plot%20Results/CA%29%20KM%20vx%20uvx/CA-AA%29%20case4_sim_deaths_sim_real_doses_no_constraint.png) | ![Case 4 Normalized](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Plot%20Results/ZI%29%20vx%20uvx%20norm/ZI-AA%29%20case4_sim_deaths_sim_real_doses_no_constraint.png) |
| 5&nbsp;&nbsp;&nbsp; | ![Case 5 Raw](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Plot%20Results/CA%29%20KM%20vx%20uvx/CA-AA%29%20case5_sim_deaths_sim_flat_random_doses_with_constraint.png) | ![Case 5 Normalized](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Plot%20Results/ZI%29%20vx%20uvx%20norm/ZI-AA%29%20case5_sim_deaths_sim_flat_random_doses_with_constraint.png) |
| 6&nbsp;&nbsp;&nbsp; | ![Case 6 Raw](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Plot%20Results/CA%29%20KM%20vx%20uvx/CA-AA%29%20case6_sim_deaths_sim_flat_random_doses_no_constraint.png) | ![Case 6 Normalized](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Plot%20Results/ZI%29%20vx%20uvx%20norm/ZI-AA%29%20case6_sim_deaths_sim_flat_random_doses_no_constraint.png) |


<br>[Download interactive htmls](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/tree/main/Plot%20Results)
<br>Phyton script [CA) KM vx uvx.py](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Py%20Scripts/CA%29%20KM%20vx%20uvx.py) 
<br>Phyton script [ZI) vx uvx norm.py](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/edit/main/Py%20Scripts/ZI%29%20vx%20uvx%20norm.py) 

_________________________________________

**Cox risk hazard vaxed/unvaxed results**

| Case | Description                                      | β (coef) | HR = exp(β) | Expected HR | Risk Reduction (%) | 95% CI (HR) | z      | p-value | −log₂(p) |
| ---- | ------------------------------------------------ | -------- | ----------- | ----------- | ------------------ | ----------- | ------ | ------- | -------- |
| 1    | Real deaths, real doses (age 70)                 | -0.29    | 0.75        |             | 25.0%              | 0.73 – 0.76 | -22.30 | <0.005  | 363.62   |
| 2    | Real deaths, real doses (no constraint)          | 0.00     | 1.00        | 1.00        | 0.0%               | 0.98 – 1.03 | 0.18   | 0.86    | 0.22     |
| 3    | Simulated deaths, real doses (with constraint)   | -0.47    | 0.62        | 1.00        | 38.0%              | 0.61 – 0.64 | -37.53 | <0.005  | 1021.66  |
| 4    | Simulated deaths, sim/real doses (no constraint) | -0.00    | 1.00        | 1.00        | 0.0%               | 0.97 – 1.03 | -0.10  | 0.92    | 0.12     |
| 5    | Sim deaths, flat random doses (with constraint)  | -0.01    | 0.99        | 1.00        | 1.0%               | 0.97 – 1.02 | -0.39  | 0.70    | 0.52     |
| 6    | Sim deaths, flat random doses (no constraint)    | -0.00    | 1.00        | 1.00        | 0.0%               | 0.97 – 1.02 | -0.25  | 0.80    | 0.32     |


<br>Phyton script [AE) Cox compare vx uvx.py](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Py%20Scripts/AE%29%20Cox%20compare%20vx%20uvx.py)  
_________________________________________
### Further simulation of ten differnet cases

<br>Phyton script [AF) simulate deaths doses curves.py](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Py%20Scripts/AF%29%20simulate%20deaths%20doses%20curves.py)  Cox Results: [Cox Results TXT](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/tree/main/Plot%20Results/AE%29%20Cox%20compare%20vx%20uvx/AE-AF%29)
_________________________________________

## Solution to correct the Non-Random Boundary Condition Bias

[**CzechFOI-DRATE-NOBIAS project for mor information**](https://github.com/gitfrid/CzechFOI-DRATE-NOBIAS)
_________________________________________

### Software Requirements:

These scripts don't require SQLite queries to aggregate the 11 million individual data rows.
Instead, the aggregation is handled directly by Python scripts, which can generate aggregated CSV files very quickly.
For coding questions or help, visit https://chatgpt.com.

- [Python 3.12.5](https://www.python.org/downloads/) to run the scripts.
- [Visual Studio Code 1.92.2](https://code.visualstudio.com/download) to edit and run scripts.


### Disclaimer:
**The results have not been checked for errors. Neither methodological nor technical checks or data cleansing have been performed.**
