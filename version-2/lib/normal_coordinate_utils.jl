using CairoMakie
using Einsum
using Flux
using LinearAlgebra

##

function first_indexpair_asymmetric(riemann::Array)
    dim = size(riemann)[1]

    should_be_zero = 0.0

    for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        @inbounds should_be_zero += (riemann[a,b,c,d] + riemann[b,a,c,d])^2
    end

    return should_be_zero
end

function last_indexpair_asymmetric(riemann::Array)
    dim = size(riemann)[1]

    should_be_zero = 0.0

    # For any choice of indices in the first two coordinates, the resulting d x d tensor
    # should be antisymmetric
    for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        @inbounds should_be_zero += (riemann[a,b,c,d] .+ riemann[a,b,d,c])^2
    end

    return should_be_zero  
end

function check_algebraic_bianchi(riemann::Array)
    dim = size(riemann)[1]

    should_be_zero = 0.0

    # For any choice of the first index, the permutation computed below should be zero
    for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        @inbounds should_be_zero += (riemann[a,b,c,d] + riemann[a,c,d,b] + riemann[a,d,b,c])^2
    end

    return should_be_zero 
end

function check_interchange_of_pairs(riemann::Array)
    dim = size(riemann)[1]

    should_be_zero = 0.0
    for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        @inbounds should_be_zero += (riemann[a,b,c,d] - riemann[c,d,a,b])^2
    end

    return should_be_zero 
end

function loss(riemann::Array)
    checks = [first_indexpair_asymmetric, 
              last_indexpair_asymmetric, 
              check_algebraic_bianchi]


    errs = [check(riemann) for check in checks]
    return norm(errs)
end

##

function random_riemann(dim::Int64, eval_at::Float64; 
                        tol::Float64=1e-10, max_iters::Float64=Inf, return_training_stats::Bool=false)
    
    eval_at = convert(Float32, eval_at)
    network = Chain(Dense(1 => dim^4), arr -> reshape(arr, (dim,dim,dim,dim)))

    local training_loss
    training_losses = [Inf]
    test_losses = Float64[]
    
    params = Flux.params(network)
    opt   = ADAM()

    iter_count = 0
    while training_losses[end] > tol && iter_count < max_iters
        gradients = gradient(params) do 
            training_loss = loss(network([eval_at]))

            return training_loss
        end

        push!(training_losses, training_loss)
        push!(test_losses, check_interchange_of_pairs(network([eval_at])))
        Flux.update!(opt, params, gradients)
    end

    riemann = convert.(Float64, network([eval_at]))

    if return_training_stats
        return riemann, training_losses[2:end], test_losses
    else
        return riemann
    end


end

# dim = 2
# @time riemann, train, test = random_riemann(4, 100.0; max_iters=1e6)

# f  = Figure()
# ax = Axis(f[1,1])
# plot!(ax, log.(10, train), color=:red)
# plot!(ax, log.(10, test),  color=:blue)
# f

##

function flat_metric(dim::Int64)
    η = Matrix(1.0I, (dim, dim))
    η[1,1] = -1

    return η
end

function normal_metric_perturbation(riemann::Array, x::Vector{Float64})
    @einsum second_order_behavior[a,b] := (1/3) * riemann[a,s,b,t] * x[s] * x[t]

    return second_order_behavior
end

function normal_metric(riemann::Array, x::Vector{Float64})
    dim = size(riemann)[1]
    g = flat_metric(dim)

    return g .- normal_metric_perturbation(riemann, x)
end

function random_normal_metric(dim::Int64, x::Vector{Float64}; eval_at::Float64=10.0)
    riemann = random_riemann(dim, eval_at)
    return normal_metric(riemann, x)

end

function normal_tetrad(riemann_at_center::Array, metric_at_center::Array, x::Vector{Float64})
    dim = size(riemann_at_center)[1]
    tetrad_at_center = flat_metric(dim)

    @einsum raised_riemann[a,b,c,d] := metric_at_center[a,s] * riemann_at_center[s,b,c,d]

    @einsum second_order_behavior[a,b] := (1/6) * raised_riemann[s,t,b,u] * tetrad_at_center[s,a] * x[t] * x[u]

    return tetrad_at_center - second_order_behavior
end

function normal_ricci()
end

# dim = 4
# riemann = random_riemann(dim, 1.0)

# test_point = ones(dim)
# metric  = normal_metric(riemann,  test_point)
# tetrad  = normal_tetrad(riemann, metric, test_point)

# η = flat_metric(4)
# @einsum test_metric[a,b] := tetrad[a,s] * η[s,t] * tetrad[t,b]

##

function make_causet(event_coordinates::Matrix{Float64}, tetrad::Function, metric::Function)
    num_events = size(event_coordinates)[2]

    # Initiate an empty directed graph's adjacency_matrix
    adjacency  = LightGraphs.adjacency_matrix(SimpleDiGraph(num_events))
    
    # Check each causal relation
    index_pairs = [(i,j) for i in 1:num_events for j in 1:num_events]
    
    lk = ReentrantLock()
    Threads.@threads for (i,j) in index_pairs
        # Don't bother going through the computation if we have already
        # determined the events are causally connected 
        if adjacency[i,j] != 1

            # We are considering the vector going from i to j
            vector_i2j  = event_coordinates[:, j] - event_coordinates[:, i]
            interval    = vector_i2j' * metric(event_coordinates[:, i]) * vector_i2j
            
            # If the separation vector is timelike and future-orientated at xi 
            if interval <= 0 && tetrad(event_coordinates[:,i])[1,:]' * vector_i2j < 0
                begin 
                    lock(lk)

                    try
                        # Add the connection from xi to xj
                        adjacency[i, j] = 1

                        # Then add connections from every event in the past of xi, 
                        # its inneighbors, to xj
                        adjacency[adjacency[:,i] .> 0, j] .= 1

                    finally
                        unlock(lk)
                    end
                end
            end
        end

    end 
            
    return event_coordinates, SimpleDiGraph(adjacency)
end