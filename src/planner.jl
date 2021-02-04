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

    if excess_uncertainty(D, b, p) <= 0
        if update_flag
            backup!(D, b, p)
        end
        return nothing::Nothing
    end

    if D.Delta[b] > p.sol.D
        make_default!(D, b)
        if update_flag
            backup!(D, b, p)
        end
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
    sorted_action = index_sort(arr, [1:length(arr);])
    highest_ub = maximum(D.ba_mu[start_ind:end_ind])

    for i in length(arr):-1:1
        best_ba = start_ind + sorted_action[i] - 1
        children_eu = [excess_uncertainty(D, bp, p) for bp in D.ba_children[best_ba]]
        max_eu, ind = findmax(children_eu)

        if max_eu > 0
            depth = D.Delta[b] / p.sol.D
            k = length(D.scenarios[b]) / p.sol.K
            zeta = p.sol.zeta

            if p.bs_count / p.de_count <= p.sol.C
                zeta *= p.sol.adjust_zeta(depth, k)
            end

            next_eu, next_id = max_eu, ind
            for i in 1:length(children_eu)
                eu, id = next_eu, next_id
                children_eu[id] = -Inf
                next_eu, next_id = findmax(children_eu)

                if id != ind
                    opt_path = false
                end

                if eu >= zeta * max_eu && next_eu < zeta * max_eu
                    explore!(D, D.ba_children[best_ba][id], p, opt_path, true)
                elseif eu >= zeta * max_eu && next_eu >= zeta * max_eu
                    explore!(D, D.ba_children[best_ba][id], p, opt_path, false)
                else
                    break
                end
            end
            break
        elseif D.ba_mu[best_ba] == highest_ub
            explore!(D, D.ba_children[best_ba][ind], p, opt_path, true)
            break
        end
    end

    if update_flag
        backup!(D, b, p)
    end
    return nothing::Nothing
end

function prune!(D::DESPOT, b::Int, p::BS_DESPOTPlanner)
    x = b
    blocked = false
    while x != 1
        n = find_blocker(D, x, p)
        if n > 0
            make_default!(D, x)
            y = x
            while y != 1
                backup!(D, y, p)
                y = D.parent_b[y]
            end
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

function backup!(D::DESPOT, b::Int, p::BS_DESPOTPlanner)
    if b != 1
        ba = D.parent[b]
        b = D.parent_b[b]

        # https://github.com/JuliaLang/julia/issues/19398
        #=
        D.ba_mu[ba] = D.ba_rho[ba] + sum(D.mu[bp] for bp in D.ba_children[ba])
        =#
        sum_mu = 0.0
        for bp in D.ba_children[ba]
            sum_mu += D.mu[bp]
        end
        D.ba_mu[ba] = D.ba_rho[ba] + sum_mu

        #=
        max_mu = maximum(D.ba_rho[ba] + sum(D.mu[bp] for bp in D.ba_children[ba]) for ba in D.children[b])
        max_l = maximum(D.ba_rho[ba] + sum(D.l[bp] for bp in D.ba_children[ba]) for ba in D.children[b])
        =#
        max_U = -Inf
        max_mu = -Inf
        max_l = -Inf
        for ba in D.children[b]
            weighted_sum_U = 0.0
            sum_mu = 0.0
            sum_l = 0.0
            for bp in D.ba_children[ba]
                weighted_sum_U += length(D.scenarios[bp]) * D.U[bp]
                sum_mu += D.mu[bp]
                sum_l += D.l[bp]
            end
            new_U = (D.ba_Rsum[ba] + discount(p.pomdp) * weighted_sum_U)/length(D.scenarios[b])
            new_mu = D.ba_rho[ba] + sum_mu
            new_l = D.ba_rho[ba] + sum_l
            max_U = max(max_U, new_U)
            max_mu = max(max_mu, new_mu)
            max_l = max(max_l, new_l)
        end

        l_0 = D.l_0[b]
        D.U[b] = max_U
        D.mu[b] = max(l_0, max_mu)
        D.l[b] = max(l_0, max_l)
    end
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
