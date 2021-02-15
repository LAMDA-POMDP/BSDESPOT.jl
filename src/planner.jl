function build_despot(p::BS_DESPOTPlanner, b_0)
    D = DESPOT(p, b_0)
    b = 1
    trial = 1
    start = CPUtime_us()

    while D.mu[1]-D.l[1] > p.sol.epsilon_0 &&
          CPUtime_us()-start < p.sol.T_max*1e6 &&
          trial <= p.sol.max_trials
        p.bs_count = 0
        p.de_count = 0
        explore!(D, 1, p, true, false)
        trial += 1
    end
    return D
end

function explore!(D::DESPOT, b::Int, p::BS_DESPOTPlanner, opt_path::Bool, update_flag::Bool)
    p.bs_count += 1
    if opt_path
        p.de_count += 1
    end

    if excess_uncertainty(D, b, p) > 0 && D.Delta[b] <= p.sol.D && !prune!(D, b, p)
        if isempty(D.children[b]) # a leaf
            expand!(D, b, p)
        end

        start_ind = D.children[b][1]
        sorted_action, highest_ub = next_act(D, b, p)
        for i in length(sorted_action):-1:1
            @inbounds best_ba = start_ind + sorted_action[i] - 1
            obs_arr = next_obs(D, b, best_ba, p, highest_ub)
            if size(obs_arr)[1] != 0
                for i in 1:size(obs_arr)[1]
                    explore!(D, obs_arr[i], p, opt_path && (i==1), (i==size(obs_arr)[1]))
                end
            else
                continue
            end
            break
        end
    end

    if D.Delta[b] > p.sol.D
        make_default!(D, b)
    end

    if update_flag
        backup!(D, b, D.parent_b[b], p)
    end
    return nothing::Nothing
end

function next_act(D::DESPOT, b::Int, p::BS_DESPOTPlanner)
    start_ind = D.children[b][1]
    end_ind = start_ind + length(D.children[b]) - 1
    if p.sol.impl == :val || p.sol.beta == 0
        arr = D.ba_mu[start_ind:end_ind] + p.sol.beta .* D.ba_l[start_ind:end_ind]
    else
        mu_ranking = ind_rank(D.ba_mu, D.children[b])
        l_ranking = ind_rank(D.ba_l, D.children[b])
        arr = mu_ranking .+ p.sol.beta.*l_ranking
    end
    sorted_action = index_sort(arr, [1:length(arr);])
    highest_ub = maximum(D.ba_mu[start_ind:end_ind])
    return sorted_action, highest_ub
end

function next_obs(D::DESPOT, b::Int, ba::Int, p::BS_DESPOTPlanner, highest_ub::Float64)
    obs_arr = Int[]
    children_eu = [excess_uncertainty(D, bp, p) for bp in D.ba_children[ba]]
    max_eu, ind = findmax(children_eu)
    
    if max_eu > 0
        depth = D.Delta[b] / p.sol.D
        k = length(D.scenarios[b]) / p.sol.K
        zeta = 1.0

        push!(obs_arr, D.ba_children[ba][ind])
        if p.sol.adjust_zeta == null_adjust
            return obs_arr
        end

        if p.bs_count / p.de_count <= p.sol.C
            zeta = p.sol.adjust_zeta(depth, k)
            @assert(zeta<=1 && zeta>=0, "$depth, $k")
        end

        children_eu[ind] = -Inf

        while true
            eu, id = findmax(children_eu)
            if eu >= zeta * max_eu
                push!(obs_arr, D.ba_children[ba][id])
                children_eu[id] = -Inf
            else
                break
            end
        end
    elseif D.ba_mu[ba] == highest_ub
        push!(obs_arr, D.ba_children[ba][ind])
    end
    return obs_arr
end

function prune!(D::DESPOT, b::Int, p::BS_DESPOTPlanner)
    x = b
    blocked = false
    while x != 1
        n = find_blocker(D, x, p)
        if n > 0
            make_default!(D, x)
            backup!(D, x, 1, p)
            blocked = true
        else
            break
        end
        x = D.parent_b[x]
    end
    return blocked
end

function find_blocker(D::DESPOT, b::Int, p::BS_DESPOTPlanner)
    len = 1
    bp = D.parent_b[b]
    while bp != 1
        lsh = length(D.scenarios[bp])/p.sol.K*discount(p.pomdp)^D.Delta[bp]*D.U[bp] - D.l_0[bp]
        if lsh <= p.sol.lambda * len
            return bp
        else
            bp = D.parent_b[bp]
            len += 1
        end
    end
    return 0 # no blocker
end

function make_default!(D::DESPOT, b::Int)
    l_0 = D.l_0[b]
    D.mu[b] = l_0
    D.l[b] = l_0
end

function backup!(D::DESPOT, b::Int, target::Int, p::BS_DESPOTPlanner)
    while b != target
        ba = D.parent[b]
        b = D.parent_b[b]

        D.ba_mu[ba] = D.ba_rho[ba] + sum(D.mu[bp] for bp in D.ba_children[ba])
        D.ba_l[ba] = D.ba_rho[ba] + sum(D.l[bp] for bp in D.ba_children[ba])

        D.U[b] = maximum(D.ba_Rsum[ba] * discount(p.pomdp) * sum(length(D.scenarios[bp]) * D.U[bp] for bp in D.ba_children[ba]) for ba in D.children[b]) / length(D.scenarios[b])
        D.mu[b] = max(D.l_0[b], maximum(D.ba_mu[ba] for ba in D.children[b]))
        D.l[b] = max(D.l_0[b], maximum(D.ba_l[ba] for ba in D.children[b]))
    end
    return nothing
end

function excess_uncertainty(D::DESPOT, b::Int, p::BS_DESPOTPlanner)
    return D.mu[b]-D.l[b] - length(D.scenarios[b])/p.sol.K * p.sol.xi * (D.mu[1]-D.l[1])
end

function null_adjust(depth, k)
    # You may design a similar function and use it to construct DESPOT solver as adjust_zata filed in it
    1
end

function index_sort(arr::Vector{Float64}, inds::Vector{Int})
    # Sort the array by its index rather than the elements
    # Return an array of indexes of inds in ascending order of the real values
    sort_ind = Vector{Int64}(undef, length(inds))
    sort_ind[1] = 1
    for i in 2:length(inds)
        ind = 1
        for j in i-1:-1:1
            if arr[inds[i]] < arr[inds[sort_ind[j]]]
                sort_ind[j+1] = sort_ind[j]
            else
                ind = j + 1
                break
            end
        end
        sort_ind[ind] = i
    end
    return sort_ind::Array{Int64,1}
end

function ind_rank(arr::Vector{Float64}, inds::Vector{Int})
    sort_ind = index_sort(arr, inds)
    ind_ranking = Vector{Int64}(undef, length(inds))
    ind_ranking[sort_ind[1]] = 1
    rank = 1
    for i in 2:length(sort_ind)
        if arr[inds[sort_ind[i]]] == arr[inds[sort_ind[i-1]]]
            ind_ranking[sort_ind[i]] = rank
        else
            rank += 1
            ind_ranking[sort_ind[i]] = rank
        end
    end
    return ind_ranking::Array{Int64,1}
end
