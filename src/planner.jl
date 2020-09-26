pl_num = 0
de_num = 0

function build_despot(p::PL_DESPOTPlanner, b_0)
    D = DESPOT(p, b_0)
    b = 1
    trial = 1
    start = CPUtime_us()

    while D.mu[1]-D.l[1] > p.sol.epsilon_0 &&
          CPUtime_us()-start < p.sol.T_max*1e6 &&
          trial <= p.sol.max_trials
          global pl_num = 1
          global de_num = 1
        explore!(D, 1, p, start)
        trial += 1
    end
    return D
end

function explore!(D::DESPOT, b::Int, p::PL_DESPOTPlanner, start::UInt64)
    global de_num = max(de_num, D.Delta[b]+1)
    depth = D.Delta[b]/p.sol.D
    k = length(D.scenarios[b])/p.sol.K
    left_time = min(0, 1 - (CPUtime_us() - start)/(p.sol.T_max*1e6))
    if D.Delta[b] <= p.sol.D &&
        excess_uncertainty(D, b, p) > 0.0 &&
        !prune!(D, b, p)

        if isempty(D.children[b]) # a leaf
            expand!(D, b, p)
        end

        # select action branch
        if p.sol.impl == :prob
            _max = -Inf
            best_ba = first(D.children[b])
            if rand(p.rng) > p.sol.beta
                for ba in D.children[b]
                    val = D.ba_mu[ba]
                    if val > _max
                        _max = val
                        best_ba = ba
                    end
                end
            else
                for ba in D.children[b]
                    val = D.ba_l[ba]
                    if val > _max
                        _max = val
                        best_ba = ba
                    end
                end
            end
        elseif p.sol.impl == :val
            _max = -Inf
            best_ba = first(D.children[b])
            for ba in D.children[b]
                val = D.ba_mu[ba] + p.sol.beta * D.ba_l[ba]
                if val > _max
                    _max = val
                    best_ba = ba
                end
            end
        else
            if p.sol.beta != 0
                mu_ranking = ind_rank(D.ba_mu, D.children[b])
                l_ranking = ind_rank(D.ba_l, D.children[b]) .- 1
                for i in 1:length(l_ranking)
                    if l_ranking[i] == 0
                        l_ranking .+= 10000
                    end
                end
                ranking = mu_ranking .+ p.sol.beta.*l_ranking
                _, ind = findmin(ranking)
            else
                _, ind = findmax([D.ba_mu[ba] for ba in D.children[b]])
            end
            best_ba = D.children[b][ind]
        end

        # select observation branch
        children_eu = [excess_uncertainty(D, bp, p) for bp in D.ba_children[best_ba]]
        max_eu, ind = findmax(children_eu)
        if max_eu <= 0
            global pl_num += 1
            explore!(D, D.ba_children[best_ba][ind], p, start)
        else
            ratio::Float64 = pl_num / de_num
            if ratio <= p.sol.C
                zeta = p.sol.zeta*p.sol.adjust_zeta(depth, k, left_time)
            else
                zeta = p.sol.zeta
            end
            @assert(zeta<=1, "$depth, $k, $left_time")
            for i in 1:length(D.ba_children[best_ba])
                eu, id = findmax(children_eu)
                if eu >= zeta * max_eu
                    global pl_num += 1
                    explore!(D, D.ba_children[best_ba][id], p, start)
                else
                    break
                end
                children_eu[id] = -Inf
            end
        end
    end
    if D.Delta[b] > p.sol.D
        make_default!(D, b)
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

function null_adjust(depth, k, left_time)
    # You may design a similar function and use it to construct DESPOT solver as adjust_zata filed in it
    1
end

function ind_rank(arr::Vector{Float64}, inds::Vector{Int})
    sort_ind = Vector{Int}(undef, length(inds))
    ind_ranking = Vector{Int}(undef, length(inds))
    sort_ind[1] = 1
    for i in 2:length(inds)
        ind = 1
        for j in i-1:-1:1
            if arr[inds[i]] > arr[inds[sort_ind[j]]]
                sort_ind[j+1] = sort_ind[j]
            else
                ind = j + 1
                break
            end
        end
        sort_ind[ind] = i
    end

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
    return ind_ranking
end
