using Distributions: Uniform, Normal
using ReverseDiff
using ReverseDiff: jacobian

using Einsum
using LinearAlgebra
using TensorCast
using AutoGrad

Input = Union{Int64,
              Float32, 
              Float64,
              ReverseDiff.TrackedArray, 
}

VectorInput = Union{Vector{Int64}, 
                    Vector{Float32},          
                    Vector{Float64},
                    Vector{ReverseDiff.TrackedReal},
                    ReverseDiff.TrackedArray,
}

##

function metric_jacobian(metric::Function, point::VectorInput)
    ∂g = jacobian(x->metric(x), point)
    
    dimension = length(point)
    shape = (dimension, dimension, dimension)
    return reshape(vec(∂g), shape)
end

function christoffel(metric::Function, point::VectorInput)
    g     = metric(point) 
    g_inv = inv(g)
    ∂g =  metric_jacobian(metric, point)

    @einsum first_kind[μ,ν,ρ] := ∂g[ν,ρ,μ] + ∂g[μ,ρ,ν] - ∂g[μ,ν,ρ]
    @einsum Γ[μ,ν,ρ] :=  g_inv[ρ,σ] * first_kind[μ,ν,σ] 

    return Γ / 2

end

function christoffel_jacobian(metric::Function, point::VectorInput)
    ∂Γ = jacobian(x->christoffel(metric, x), point)

    dimension = length(point)
    shape = (dimension, dimension, dimension, dimension)
    return reshape(vec(∂Γ), shape)
end

function riemannian(metric::Function, point::VectorInput)
    Γ  = christoffel(metric, point)
    ∂Γ = christoffel_jacobian(metric, point)
    @einsum R[μ,ν,σ,ρ] := ∂Γ[μ,ρ,σ,ν] - ∂Γ[ν,ρ,σ,μ] +
                          Γ[μ,ρ,α]*Γ[α,ν,σ] - Γ[ν,ρ,α]*Γ[α,μ,σ]

    return R
end
# R = riemannian(network, s)
# R[1,2,:,:]
# @einsum bianchi[a,b,c,d] := R[a,b,c,d] + R[a,c,d,b] + R[a,d,b,c]


function ricci(metric::Function, point::VectorInput)
    Γ  = christoffel(metric, point)
    ∂Γ = christoffel_jacobian(metric, point)

    @einsum R[μ,ρ] := ∂Γ[μ,ρ,ν,ν] - ∂Γ[ν,ρ,ν,μ] +
                      Γ[μ,ρ,α]*Γ[α,ν,ν] - Γ[ν,ρ,α]*Γ[α,μ,ν]

    return R
end

function scalar(metric::Function, point::VectorInput)
    g = metric(point)
    R = ricci(metric, point)
    g_inv = inv(g)

    @einsum S := g_inv[i,j] * R[i,j]
    return S
end
##

function schwartzchild(point::VectorInput;
                       c::Float64=1.0, G::Float64=1.0, M::Float64=0.5)
    rs = 2 * G * M / c^2

    return diagm([-(1 - rs / point[2]), 
                 1/(1 - rs / point[2]),
                 point[2]^2,
                 point[2]^2 * sin(point[3])^2
                 ])
end

##
# point = [0, 2000, π/2, 0]
# scalar(p->schwartzchild(p; M=0.01), point)



