function build_despot(p::PL_DESPOTPlanner, b_0)
    D = DESPOT(p, b_0)
    b = 1
    trial = 1
    start = CPUtime_us()

    while D.mu[1]-D.l[1] > p.sol.epsilon_0 &&
          CPUtime_us()-start < p.sol.T_max*1e6 &&
          trial <= p.sol.max_trials
        explore!(D, 1, p, start)
        trial += 1
    end
    return D
end

function explore!(D::DESPOT, b::Int, p::PL_DESPOTPlanner, start)
    depth = D.Delta[b]/p.sol.D
    k = length(D.scenarios[b])/p.sol.K
    left_time = 1 - (CPUtime_us() - start)/p.sol.T_max
    if D.Delta[b] <= p.sol.D &&
        excess_uncertainty(D, b, p) > 0.0 &&
        !prune!(D, b, p)

        if isempty(D.children[b]) # a leaf
            expand!(D, b, p)
        end

        # select action branch
        max_mu = -Inf
        best_ba = first(D.children[b])
        for ba in D.children[b]
            mu = D.ba_mu[ba]
            if mu > max_mu
                max_mu = mu
                best_ba = ba
            end
        end
    
        # select observation branch
        children_eu = [excess_uncertainty(D, bp, p) for bp in D.ba_children[best_ba]]
        max_eu, ind = findmax(children_eu)
        if max_eu <= 0
            explore!(D, D.ba_children[best_ba][ind], p, start)
        else
            zeta = p.sol.zeta*p.sol.adjust_zata(depth, k, left_time)
            @assert(zeta<=1, "adjust function need to be redesigned")
            for i in 1:length(D.ba_children[best_ba])
                if children_eu[i] >= zeta*max_eu
                    explore!(D, D.ba_children[best_ba][i], p, start)
                end
            end
        end
    end
    if D.Delta[b] > p.sol.D
        make_default!(D, b)
    end
    backup!(D, b, p)
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

        U = []
        mu = []
        l = []
        for ba in D.children[b]
            weighted_sum_U = 0.0
            sum_mu = 0.0
            sum_l = 0.0
            for bp in D.ba_children[ba]
                weighted_sum_U += length(D.scenarios[bp]) * D.U[bp]
                sum_mu += D.mu[bp]
                sum_l += D.l[bp]
            end
            push!(U, D.ba_Rsum[ba] + discount(p.pomdp) * weighted_sum_U)/length(D.scenarios[b])
            push!(mu, D.ba_rho[ba] + sum_mu)
            push!(l, D.ba_rho[ba] + sum_l)
        end

        l_0 = D.l_0[b]
        D.U[b] = maximum(U)
        D.mu[b] = max(l_0, maximum(mu))
        D.l[b] = max(l_0, maximum(l))
    end
end

function excess_uncertainty(D::DESPOT, b::Int, p::PL_DESPOTPlanner)
    return D.mu[b]-D.l[b] - length(D.scenarios[b])/p.sol.K * p.sol.xi * (D.mu[1]-D.l[1])
end

function null_adjust(depth, k, left_time)
    # You may design a similar function and use it to construct DESPOT solver as adjust_zata filed in it
    1
end