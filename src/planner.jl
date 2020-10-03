function build_despot(p::PL_DESPOTPlanner, b_0)
    D = DESPOT(p, b_0)
    b = 1
    trial = 1
    start = CPUtime_us()

    while D.mu[1]-D.l[1] > p.sol.epsilon_0 &&
          CPUtime_us()-start < p.sol.T_max*1e6 &&
          trial <= p.sol.max_trials
        p.pl_count = 0
        p.de_count = 0
        explore!(D, 1, p, start, 1)
        trial += 1
    end
    return D
end

function explore!(D::DESPOT, b::Int, p::PL_DESPOTPlanner, start::UInt64, dist::Int)
    p.pl_count += 1
    if dist == 1
        p.de_count += 1
    end

    if excess_uncertainty(D, b, p) <= 0
        backup!(D, b, p)
        return nothing::Nothing
    end

    if D.Delta[b] > p.sol.D
        make_default!(D, b)
        backup!(D, b, p)
        return nothing::Nothing
    end

    if prune!(D, b, p)
        return nothing::Nothing
    end

    if isempty(D.children[b]) # a leaf
        expand!(D, b, p)
    end

    start_ind = D.children[b][1]
    end_ind = start_ind + length(D.children[b]) - 1
    if p.sol.impl == :prob
        arr =  rand(p.rng) > p.sol.beta ? D.ba_mu[start_ind:end_ind] : D.ba_l[start_ind:end_ind]
    elseif p.sol.impl == :val
        arr = D.ba_mu[start_ind:end_ind] + p.sol.beta .* D.ba_l[start_ind:end_ind]
    else
        mu_ranking = ind_rank(D.ba_mu, D.children[b])
        l_ranking = ind_rank(D.ba_l, D.children[b])
        arr = mu_ranking .+ p.sol.beta.*l_ranking
    end
    best_ba = 0
    max_eu = 0.0
    children_eu = Float64[]
    sorted_action = index_sort(arr, [1:length(arr);])
    highest_ub = maximum(D.ba_mu[start_ind:end_ind])

    for i in length(arr):-1:1
        best_ba = start_ind + sorted_action[i] - 1
        children_eu = [excess_uncertainty(D, bp, p) for bp in D.ba_children[best_ba]]
        max_eu, ind = findmax(children_eu)
        if max_eu > 0
            explore!(D, D.ba_children[best_ba][ind], p, start, dist)
            children_eu[ind] = -Inf

            depth = D.Delta[b]/p.sol.D
            k = length(D.scenarios[b])/p.sol.K
            left_time = min(0, 1 - (CPUtime_us() - start)/(p.sol.T_max*1e6))

            ratio::Float64 = p.pl_count / p.de_count
            zeta = p.sol.zeta
            if ratio <= p.sol.C
                zeta *= p.sol.adjust_zeta(p.sol.zeta_l, depth, k, left_time)
            end
            for i in 1:length(children_eu)
                eu = children_eu[i]
                if eu >= zeta * max_eu
                    explore!(D, D.ba_children[best_ba][i], p, start, dist+1)
                end
            end
            break
        elseif D.ba_mu[best_ba] == highest_ub
            explore!(D, D.ba_children[best_ba][ind], p, start, dist)
            break
        end
    end

    backup!(D, b, p)
    return nothing::Nothing
end

function prune!(D::DESPOT, b::Int, p::PL_DESPOTPlanner)
    x = b
    blocked = false
    while x != 1
        n = find_blocker(D, x, p)
        if n > 0
            make_default!(D, x)
            backup!(D, x, p)
            blocked = true
        else
            break
        end
        x = D.parent_b[x]
    end
    return blocked
end

function find_blocker(D::DESPOT, b::Int, p::PL_DESPOTPlanner)
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

function backup!(D::DESPOT, b::Int, p::PL_DESPOTPlanner)
    if b != 1
        ba = D.parent[b]
        b = D.parent_b[b]

        D.ba_mu[ba] = D.ba_rho[ba] + sum(D.mu[bp] for bp in D.ba_children[ba])
        D.ba_l[ba] = D.ba_rho[ba] + sum(D.l[bp] for bp in D.ba_children[ba])

        U = []
        mu = []
        l = []
        for ba in D.children[b]
            weighted_sum_U = 0.0
            for bp in D.ba_children[ba]
                weighted_sum_U += length(D.scenarios[bp]) * D.U[bp]
            end
            push!(U, D.ba_Rsum[ba] + discount(p.pomdp) * weighted_sum_U)/length(D.scenarios[b])
            push!(mu, D.ba_rho[ba] + D.ba_mu[ba])
            push!(l, D.ba_rho[ba] + D.ba_l[ba])
        end

        l_0 = D.l_0[b]
        D.U[b] = maximum(U)
        D.mu[b] = max(l_0, maximum(mu))
        D.l[b] = max(l_0, maximum(l))
    end
    return nothing::Nothing
end

function excess_uncertainty(D::DESPOT, b::Int, p::PL_DESPOTPlanner)
    return D.mu[b]-D.l[b] - length(D.scenarios[b])/p.sol.K * p.sol.xi * (D.mu[1]-D.l[1])
end

function null_adjust(l, depth, k, left_time)
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
