# BS-DESPOT
An implementation of the BS-DESPOT (Better Selection DESPOT) online POMDP Solver. BS-DESPOT is a variation of DESPOT. It provides action branch selection based on upper and lower bounds, and multi-observation branches selection.
Original ARDESPOT please refer to https://github.com/JuliaPOMDP/ARDESPOT.jl.

## Installation
```bash
Pkg> registry add git@github.com:JuliaPOMDP/Registry.git
Pkg> add https://github.com/LAMDA-POMDP/BS-DESPOT # If a mature version is needed
Pkg> dev PATH/TO/BS-DESPOT # If a version in development is needed, please first clone the project to the local.
```

## Usage
### Action Branch Selection
BS-DESPOT provides three methods for selecting action branch based on upper and lower bounds: value-based, probability-based, ranking-based. The default is ranking-based method. Usage is as follows:
```julia
solver = BS_DESPOTSolver(..., impl=:rank, ...) # Ranking-based
solver = BS_DESPOTSolver(..., impl=:val, ...) # Value-based
solver = BS_DESPOTSolver(..., impl=:prob, ...) # Probability-based
```
$\beta$ is the coefficient for adjusting the engagement of the lower bound. The default is 0.
```julia
solver = BS_DESPOTSolver(..., beta=0.1, ...) # How to adjust beta
```

### Observation Branch Selection
$\zeta$ is the parameter to determine how close the branches are to the optimal ones will be selected. The default is 1.
```julia
solver = BS_DESPOTSolver(..., zeta=0.9, ...) # How to adjust zeta
```
If you need to dynamically adjust $\zeta$ during planning, please define a function related to d and k, i.e.
```julia
# Define a function to adjust zeta dynamically. d is the ratio of the current depth to the maximum depth, k is the ratio of the number of current scenarios to K.
function f_zeta(d, k)
  1 - 0.1*k - 0.1*(1-d)
end
# When initializing the solver, specify the function
solver = BS_DESPOTSolver(..., adjust_zeta=f_zeta, ...)
```
