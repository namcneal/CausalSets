using CairoMakie
using Colors, ColorSchemes
using Einsum
using GeometryBasics
using LinearAlgebra
using StatsBase
using VoronoiCells

using BSON: @save
include("../lib/MetricML.jl")
include("../lib/metropolis.jl")
include("../geodesic_utils.jl");

##
""" The activation function"""
α=0.9
β=0.1
γ=0.8
activation(x) = leakysoftplus(x; α=α, β=β, γ=γ) 
##

dim = 2
network    = create_network(dim, [25,25],  activation)
tetrad(x)  = forward(network, x)
metric(x)  = network2metric(network, x)
density(x) = metric(x) |> det |> x->-x |> sqrt 

##
function inside_nsphere(x::Vector{Float64};
        radius::Float64=1.0,
        center::Vector{Float64}=zeros(length(x)))

        return norm(x .- center) <= radius
end

characteristic(x::Vector{Float64}) = inside_nsphere(x; radius=30.0)

num_MCMC_samples = 1e7
num_MCMC_samples = convert(Int64, num_MCMC_samples)
MCMC_width = 5e-1

MCMC = metropolis(characteristic, density, 
                  num_MCMC_samples, MCMC_width, 
                  zeros(2))

##
num_events = 10000
@assert num_events < num_MCMC_samples
events = MCMC[:, sample(1:num_MCMC_samples, num_events)]
events[:, 1] *= 0

xs = events[1,:]
ys = events[2,:]
f  = Figure()
ax = Axis(f[1,1], aspect=1)
 
source_index = 1
source = events[:, source_index]
ϵ = 31
xlims!(ax, source[1].+[-ϵ,ϵ])
ylims!(ax, source[2].+[-ϵ,ϵ])
scatter!(xs, ys, markersize=2)
f

##
function shoot_geodesics_epsilon_ball(source::Vector{Float64}, )
end


radius = ϵ
neighbor_indices = mapslices(norm, events .- source; dims=1) .< radius
neighbors        = events[:, vec(neighbor_indices)]
num_neighbors    = size(neighbors)[2]

geodesic_step_size = 1e-2
s = 4.0
shot_toward_neighbors = [shoot_geodesic(tetrad, source, neighbors[:,i]; step=geodesic_step_size, boundary_radius_scaling=s) 
                         for i in 1:num_neighbors]

##

source_index = 1
source = events[:, source_index]
ϵ = 4.0

p = scatter(xs, ys)
plot!(xlims=source[1].+[-ϵ,ϵ], ylims=source[2].+[-ϵ,ϵ], aspect_ratio=1, markersize=1, legend=nothing)

for path in shot_toward_neighbors
        plot!(path2xy(path)...)
end

p

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




