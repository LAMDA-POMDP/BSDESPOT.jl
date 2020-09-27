push!(LOAD_PATH, "..")
using PL_DESPOT # PL_DESPOT pkg
using Test

using POMDPs
using POMDPModels
using POMDPSimulators
using Random
using POMDPModelTools
using ParticleFilters
using CPUTime

include("memorizing_rng.jl")
include("independent_bounds.jl")

pomdp = BabyPOMDP()
pomdp.discount = 1.0

K = 10
rng = MersenneTwister(14)
rs = MemorizingSource(K, 50, rng)
Random.seed!(rs, 10)
b_0 = initialstate(pomdp)
scenarios = [i=>rand(rng, b_0) for i in 1:K]
o = false
b = ScenarioBelief(scenarios, rs, 0, o)
pol = FeedWhenCrying()
r1 = PL_DESPOT.branching_sim(pomdp, pol, b, 10, (m,x)->0.0)
r2 = PL_DESPOT.branching_sim(pomdp, pol, b, 10, (m,x)->0.0)
@test r1 == r2
tval = 7.0
r3 = PL_DESPOT.branching_sim(pomdp, pol, b, 10, (m,x)->tval)
@test r3 == r2 + tval*length(b.scenarios)

scenarios = [1=>rand(rng, b_0)]
b = ScenarioBelief(scenarios, rs, 0, false)
pol = FeedWhenCrying()
r1 = PL_DESPOT.rollout(pomdp, pol, b, 10, (m,x)->0.0)
r2 = PL_DESPOT.rollout(pomdp, pol, b, 10, (m,x)->0.0)
@test r1 == r2
tval = 7.0
r3 = PL_DESPOT.rollout(pomdp, pol, b, 10, (m,x)->tval)
@test r3 == r2 + tval

# AbstractParticleBelief interface
@test n_particles(b) == 1
s = particle(b,1)
@test rand(rng, b) == s
@test pdf(b, rand(rng, b_0)) == 1
sup = support(b)
@test length(sup) == 1
@test first(sup) == s
@test mode(b) == s
@test mean(b) == s
@test first(particles(b)) == s
@test first(weights(b)) == 1.0
@test first(weighted_particles(b)) == (s => 1.0)
@test weight_sum(b) == 1.0
@test weight(b, 1) == 1.0
@test currentobs(b) == o
@test_deprecated previous_obs(b)
@test history(b)[end].o == o

pomdp = BabyPOMDP()

# constant bounds
bds = (reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
solver = PL_DESPOTSolver(bounds=bds, impl=:val, beta=0.3)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=2)
@time hist = simulate(hr, pomdp, planner)

# policy lower bound
bds = IndependentBounds(DefaultPolicyLB(FeedWhenCrying()), 0.0)
solver = PL_DESPOTSolver(bounds=bds, impl=:prob, beta=0.3)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=2)
@time hist = simulate(hr, pomdp, planner)

# policy lower bound with final value
fv(m::BabyPOMDP, x) = reward(m, true, false)/(1-discount(m))
bds = IndependentBounds(DefaultPolicyLB(FeedWhenCrying(), final_value=fv), 0.0)
solver = PL_DESPOTSolver(bounds=bds, impl=:rank, beta=0.3)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=2)
@time hist = simulate(hr, pomdp, planner)

# Type stability
pomdp = BabyPOMDP()
bds = IndependentBounds(reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
solver = PL_DESPOTSolver(epsilon_0=0.1,
                      bounds=bds,
                      rng=MersenneTwister(4)
                     )
p = solve(solver, pomdp)

b0 = initialstate(pomdp)
D = @inferred PL_DESPOT.build_despot(p, b0)
@inferred PL_DESPOT.explore!(D, 1, p, 1)
@inferred PL_DESPOT.expand!(D, length(D.children), p)
@inferred PL_DESPOT.prune!(D, 1, p)
@inferred PL_DESPOT.find_blocker(D, length(D.children), p)
@inferred PL_DESPOT.make_default!(D, length(D.children))
@inferred PL_DESPOT.backup!(D, 1, p)
@inferred PL_DESPOT.excess_uncertainty(D, 1, p)
@inferred PL_DESPOT.index_sort([1.0, 3.0, 2.0, 4.0], [1:4;])
@inferred PL_DESPOT.ind_rank([1.0, 3.0, 2.0, 4.0], [1:4;])
@inferred action(p, b0)


bds = IndependentBounds(reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
rng = MersenneTwister(4)
solver = PL_DESPOTSolver(epsilon_0=0.1,
                      bounds=bds,
                      rng=rng,
                      random_source=MemorizingSource(500, 90, rng),
                      tree_in_info=true
                     )
p = solve(solver, pomdp)
a = action(p, initialstate(pomdp))

include("random_2.jl")

# visualization
show(stdout, MIME("text/plain"), D)
a, info = action_info(p, initialstate(pomdp))
show(stdout, MIME("text/plain"), info[:tree])

# from README:
using POMDPs, POMDPModels, POMDPSimulators, PL_DESPOT

pomdp = TigerPOMDP()

solver = PL_DESPOTSolver(bounds=(-20.0, 0.0))
planner = solve(solver, pomdp)

for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
