using Colors, ColorSchemes
using Einsum
using GeometryBasics
using LinearAlgebra
using Plots
using StatsBase
using VoronoiCells

using BSON: @save
include("lib/MetricML.jl")
include("lib/metropolis.jl")
include("geodesic_utils.jl");

##

α=0.1
β=0.5
activation(x) = leakysoftplus(x; α=α, β=α)

dim = 2
network    = create_network(dim, [25,25],  activation)
tetrad(x)  = forward(network, x)
metric(x)  = network2metric(network, x)
density(x) = metric(x) |> det |> x->-x |> sqrt 

##
num_MCMC_samples = 1e7
num_MCMC_samples = convert(Int64, num_MCMC_samples)
MCMC_step = 5e-1
characteristic(x::Vector{Float64}) = norm(x) < 30

MCMC = metropolis(characteristic, density, num_MCMC_samples, MCMC_step, zeros(2))

##
num_events = 10000
@assert num_events < num_MCMC_samples
events = MCMC[:, sample(1:num_MCMC_samples, num_events)]
events[:, 1] *= 0

xs = events[1,:]
ys = events[2,:]
scatter(xs, ys, markersize=2)

source_index = 1
source = events[:, source_index]
ϵ = 30
plot!(xlims=source[1].+[-ϵ,ϵ], ylims=source[2].+[-ϵ,ϵ], aspect_ratio=1)

##
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




