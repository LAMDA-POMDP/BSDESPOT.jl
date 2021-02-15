using POMDPs
using BS_DESPOT
using POMDPModelTools
using POMDPModels
using ProgressMeter
using Random
using POMDPSimulators

T = 50
N = 50

pomdp = BabyPOMDP()

bounds = IndependentBounds(DefaultPolicyLB(FeedWhenCrying()), 0.0)
# bounds = IndependentBounds(reward(pomdp, false, true)/(1-discount(pomdp)), 0.0)

solver = BS_DESPOTSolver(epsilon_0=0.001,
                      K=100,
                      D=50,
                      lambda=0.01,
                      bounds=bounds,
                      T_max=Inf,
                      max_trials=500,
                      rng=MersenneTwister(4),
                     )

@show solver.lambda

global rsum = 0.0
global fwc_rsum = 0.0
@showprogress for i in 1:N
    planner = solve(solver, pomdp)
    sim = RolloutSimulator(max_steps=T, rng=MersenneTwister(i))
    fwc_sim = deepcopy(sim)
    global rsum += simulate(sim, pomdp, planner)
    global fwc_rsum += simulate(fwc_sim, pomdp, FeedWhenCrying())
end

@show rsum/N
@show fwc_rsum/N
