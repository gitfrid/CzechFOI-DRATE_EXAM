# CzechFOI-DRATE_EXAM

## Investigation of the Non-Random Boundary Condition Bias

The **non-random boundary condition bias** arises when a homogeneous group is **divided into subgroups based on a non-random time point**—as often happens in real observational studies—and these subgroups are then compared.

This can lead to **illusory differences between groups**, even when **no real causal effect** exists.

Unlike more widely recognized biases such as:

- Immortal time bias
- Survivorship Bias  
- Healthy user effect  
- Selection bias  
- Treatment delay bias  

...this bias is often **overlooked**, yet it can have a **major impact** on the interpretation of observational data.

_________________________________________

### AA) simulate deaths doses.py

The simulation script shows how the application of "alive at dose time" condition creates a structural bias, even when deaths and doses are assigned completely at random (i.e., no causal effect of the vaccine is assumed or simulated).

<br>Phyton script [AA) simulate deaths doses.py](https://github.com/gitfrid/CzechFOI-DRATE_EXAM/blob/main/Py%20Scripts/AA%29%20simulate%20deaths%20doses.py) 

_________________________________________

| Case | Deaths       | Doses        | Condition (only assign if alive at dose day)   | Expected Bias   |
|-------|--------------|--------------|------------------------------------------------|-----------------|
| 1     | Real&nbsp;&nbsp;&nbsp;   | Real&nbsp;&nbsp;&nbsp;   | Inherent (real doses can only happen if alive)   | ✅ Bias Present  |
| 2     | Simulated&nbsp;&nbsp;&nbsp;   | Real&nbsp;&nbsp;&nbsp;   | Removed (assign regardless of survival)           | ❌ No Bias      |
| 3     | Simulated&nbsp;&nbsp; | Probabilistic from real curve&nbsp;&nbsp;&nbsp;   | ✅ Applied                                       | ✅ Bias Present  |
| 4     | Simulated&nbsp;&nbsp; | Probabilistic from real curve&nbsp;&nbsp;&nbsp;   | ❌ Not applied                                   | ❌ No Bias      |
| 5     | Simulated&nbsp;&nbsp;&nbsp;   | Simulate rectangular&nbsp;&nbsp; | ✅ Applied                                       | ✅ Bias Present  |
| 6     | Simulated&nbsp;&nbsp;&nbsp;   | Simulate rectangular&nbsp;&nbsp; | ❌ Not applied                                   | ❌ No Bias      |



# Simulation Cases Visualization

Below are the results of the 6 cases, shown as raw (CA-AA) and normalized (ZI-AA) time series side-by-side.

| Case | Raw Time Series (CA-AA) | Normalized Time Series (ZI-AA) |
|-------|-------------------------|-------------------------------|
| Case 1 | ![Case 1 Raw](path/to/CA-AA_case1_real_deaths_real_doses.png) | ![Case 1 Normalized](path/to/ZI-AA_case1_real_deaths_real_doses.png) |
| Case 2 | ![Case 2 Raw](path/to/CA-AA_case2_sim_deaths_real_doses_no_constraint.png) | ![Case 2 Normalized](path/to/ZI-AA_case2_sim_deaths_real_doses_no_constraint.png) |
| Case 3 | ![Case 3 Raw](path/to/CA-AA_case3_sim_deaths_sim_real_doses_with_constraint.png) | ![Case 3 Normalized](path/to/ZI-AA_case3_sim_deaths_sim_real_doses_with_constraint.png) |
| Case 4 | ![Case 4 Raw](path/to/CA-AA_case4_sim_deaths_sim_real_doses_no_constraint.png) | ![Case 4 Normalized](path/to/ZI-AA_case4_sim_deaths_sim_real_doses_no_constraint.png) |
| Case 5 | ![Case 5 Raw](path/to/CA-AA_case5_sim_deaths_sim_flat_random_doses_with_constraint.png) | ![Case 5 Normalized](path/to/ZI-AA_case5_sim_deaths_sim_flat_random_doses_with_constraint.png) |
| Case 6 | ![Case 6 Raw](path/to/CA-AA_case6_sim_deaths_sim_flat_random_doses_no_constraint.png) | ![Case 6 Normalized](path/to/ZI-AA_case6_sim_deaths_sim_flat_random_doses_no_constraint.png) |

