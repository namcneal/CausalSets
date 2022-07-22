using CairoMakie
using Einsum
using Flux
using LinearAlgebra

##


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

