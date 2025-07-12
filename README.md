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
