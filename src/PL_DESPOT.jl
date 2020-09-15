module PL_DESPOT

using POMDPs
using BeliefUpdaters
using Parameters
using CPUTime
using ParticleFilters
using D3Trees
using Random
using Printf
using POMDPModelTools

using BasicPOMCP # for ExceptionRethrow and NoDecision
import BasicPOMCP.default_action

import Random.rand


export
    # PL_DESPOT.jl
    PL_DESPOTSolver,
    PL_DESPOTPlanner,

    # random_2.jl
    DESPOTRandomSource,
    MemorizingSource,
    MemorizingRNG,

    # scenario_belief.jl
    ScenarioBelief,
    previous_obs,

    # exceptions.jl
    default_action,
    NoGap,

    # bounds.jl
    IndependentBounds,
    FullyObservableValueUB,
    DefaultPolicyLB,
    bounds,
    init_bounds,
    lbound,
    ubound,
    init_bound,

    # MCTS.jl
    ReportWhenUsed

# include("random.jl")
include("random_2.jl")

"""
    PL_DESPOTSolver(<keyword arguments>)

Implementation of the PL_DESPOTSolver solver trying to closely match the pseudo code of:

http://bigbird.comp.nus.edu.sg/m2ap/wordpress/wp-content/uploads/2017/08/jair14.pdf

Each field may be set via keyword argument. The fields that correspond to algorithm
parameters match the definitions in the paper exactly.

# Fields
- `epsilon_0`
- `xi`
- `K`
- `D`
- `lambda`
- 'zeta'
- `T_max`
- `max_trials`
- `bounds`
- `default_action`
- `rng`
- `random_source`
- `bounds_warnings`
- `tree_in_info`

Further information can be found in the field docstrings (e.g.
`?PL_DESPOTSolver.xi`)
"""

@with_kw mutable struct PL_DESPOTSolver <: Solver
    "The target gap between the upper and the lower bound at the root of the partial DESPOT."
    epsilon_0::Float64                      = 0.0

    "The rate of target gap reduction."
    xi::Float64                             = 0.95

    "The number of sampled scenarios."
    K::Int                                  = 500

    "The maximum depth of the DESPOT."
    D::Int                                  = 90

    "Reguluarization constant."
    lambda::Float64                         = 0.01
    
    "The maximum online planning time per step."
    T_max::Float64                          = 1.0

    "The maximum number of trials of the planner."
    max_trials::Int                         = typemax(Int)

    "A representation for the upper and lower bound on the discounted value (e.g. `IndependentBounds`)."
    bounds::Any                             = IndependentBounds(-1e6, 1e6)

    """A default action to use if algorithm fails to provide an action because of an error.
    This can either be an action object, i.e. `default_action=1` if `actiontype(pomdp)==Int` or a function `f(pomdp, b, ex)` where b is the belief and ex is the exception that caused the planner to fail.
    """
    default_action::Any                     = ExceptionRethrow()

    "A random number generator for the internal sampling processes."
    rng::MersenneTwister                    = MersenneTwister(rand(UInt32))

    "A source for random numbers in scenario rollout"
    random_source::DESPOTRandomSource       = MemorizingSource(K, D, rng)

    "If true, sanity checks on the provided bounds are performed."
    bounds_warnings::Bool                   = true

    "If true, a reprenstation of the constructed DESPOT is returned by POMDPModelTools.action_info."
    tree_in_info::Bool                      = false

    "The fixed rate of choosing extra observation branches"
    zeta::Float64                           = 0.8
end

include("scenario_belief.jl")
include("default_policy_sim.jl")
include("bounds.jl")

struct PL_DESPOTPlanner{P<:POMDP, B, RS<:DESPOTRandomSource, RNG<:AbstractRNG} <: Policy
    sol::PL_DESPOTSolver
    pomdp::P
    bounds::B
    rs::RS
    rng::RNG
end

function PL_DESPOTPlanner(sol::PL_DESPOTSolver, pomdp::POMDP)
    bounds = init_bounds(sol.bounds, pomdp, sol)
    rng = deepcopy(sol.rng)
    rs = deepcopy(sol.random_source)
    Random.seed!(rs, rand(rng, UInt32))
    return PL_DESPOTPlanner(deepcopy(sol), pomdp, bounds, rs, rng)
end

include("tree.jl")
include("planner.jl")
include("pomdps_glue.jl")

include("visualization.jl")
include("exceptions.jl")

end # module