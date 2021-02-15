# BSDESPOT
An implementation of the BSDESPOT (Better Selection DESPOT) online POMDP Solver. BSDESPOT is a variation of DESPOT. It provides action branch selection based on upper and lower bounds, and multi-observation branches selection.

## Installation
```bash
Pkg> registry add git@github.com:JuliaPOMDP/Registry.git
Pkg> add https://github.com/LAMDA-POMDP/BSDESPOT.jl # If a mature version is needed
Pkg> dev PATH/TO/BSDESPOT # If a version in development is needed, please first clone the project to the local.
```

## Usage
```julia
using POMDPs, POMDPModels, POMDPSimulators, BSDESPOT

pomdp = TigerPOMDP()

solver = BS_DESPOTSolver(bounds=IndependentBounds(-20.0, 0.0))
planner = solve(solver, pomdp)

for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
```

## Solver Options
For some detailed parameters of DESPOT, please refer to original ARDESPOT: https://github.com/JuliaPOMDP/ARDESPOT.jl.

### Action Branch Selection
BSDESPOT provides two methods for selecting action branch based on upper and lower bounds: value-based, ranking-based. The default is ranking-based method. Usage is as follows:
```julia
solver = BS_DESPOTSolver(..., impl=:rank, ...) # Ranking-based
solver = BS_DESPOTSolver(..., impl=:val, ...) # Value-based
```
$\beta$ is the coefficient for adjusting the engagement of the lower bound. The default is 0 (only use upper bound selection).
```julia
solver = BS_DESPOTSolver(..., beta=0.1, ...) # How to adjust beta
```

### Observation Branch Selection
$\zeta$ is the parameter to determine how close the branches are to the optimal ones will be selected. The default is 1 (only expand single observation branch). If you need to dynamically adjust $\zeta$ during planning, please define a function related to d and k, i.e.
```julia
# Define a function to adjust zeta dynamically. d is the ratio of the current depth to the maximum depth, k is the ratio of the number of current scenarios to K.
function f_zeta(d, k)
  1 - 0.1*k - 0.1*(1-d)
end
# When initializing the solver, specify the function
solver = BS_DESPOTSolver(..., adjust_zeta=f_zeta, ...)
```
