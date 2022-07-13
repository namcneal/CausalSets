using CairoMakie
using Colors, ColorSchemes
using Distributions: Uniform
using Einsum
using FLoops
using GeometryBasics
using LinearAlgebra
using Random
using StatsBase
using VoronoiCells

using BSON: @save
include("lib/MetricML.jl")
include("lib/metropolis.jl")


##
       
α=1e-2
β=1e-2
activation(x) = leakysoftplus(x, α=α, β=α)

dim = 4
tetrad = create_network(dim, [1], leakysoftplus)
metric(x)  = tetrad2metric(tetrad, x, tetrad.parameters)
density(x) = metric(x) |> det |> (x -> -x) |> sqrt

##
function inside_nsphere(x::Vector{Float64};
                        radius::Float64=1.0,
                        center::Vector{Float64}=zeros(length(x)))

    return norm(x .- center) <= radius
end

sphere_radius = 1.0
sphere_characteristic(x::Vector{Float64}) = inside_nsphere(x; radius=sphere_radius)

num_MCMC_samples = 500000
metropolic_width = sphere_radius / 10
initial_coordinates = zeros(dim)

MCMC = metropolis(sphere_characteristic, density, 
                  num_MCMC_samples, metropolic_width,
                  initial_coordinates)

# ##
# num_events = 10000
# @assert num_events < num_MCMC_samples
# scattering = MCMC[:, sample(1:num_MCMC_samples, num_events)]


# ##
# f = Figure()
# ax  = Axis(f[1,1], aspect=1.0)

# lim = 1.1 * sphere_radius
# xlims!(-lim, lim)
# ylims!(-lim, lim)

# # xs = MCMC[1,:]
# # ys = MCMC[2,:]

# xs = scattering[1,:]
# ys = scattering[2,:]

# scatter!(xs, ys, markersize=3.0)
# f

## 
"""
Trying to lower the curvature by training the neural create_network
"""

function loss(tetrad::TetradNN, points::Vector{Vector{Float64}}, parameters::Vector)
    num_points = length(points)

    metric = x -> tetrad2metric(tetrad, x, parameters)

    loss = 0
    for i in 1:num_points
        riemann = riemannian(metric, points[i])

        Random.seed!(123)
        target_riemann = rand(Uniform(-1.0, 1.0), size(riemann))

        loss += norm(riemann .- target_riemann) / length(riemann)   
    end

    return loss
end

# function reduce_curvature()
# end

##

num_points = 1000
dist = Uniform(-1.0, 1.0)
Random.seed!(123)
points = [rand(dist, dim) for _ in 1:num_points]

num_steps = 100
@time trained_tetrad, losses = train(tetrad, points, loss, num_steps)

##
plot(losses)