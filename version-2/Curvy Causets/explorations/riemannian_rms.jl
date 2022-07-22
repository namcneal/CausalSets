using CairoMakie
using Colors, ColorSchemes
using Einsum
using GeometryBasics
using LinearAlgebra
using Profile 
using StatsBase
using VoronoiCells

using BSON: @save


include("../lib/MetricML.jl")
include("../lib/metropolis.jl")
include("../geodesic_utils.jl");
include("../lib/plotting_utils.jl");

##
""" The activation function"""
α=0.1
β=0.35
γ=0.2
activation(x) = leakysoftplus(x; α=α, β=β, γ=γ) 

f  = Figure()
figure_layout = Dict(
    :title => "Activation Function on ℝ",
    :titlesize => 30,
    :xlabel=>"Function Argument",
    :xlabelsize=>25,
    :ylabel=>"Activation  Value",
    :ylabelsize=>25,
    :aspect=>1
)
ax = Axis(f[1,1]; figure_layout...) 

xs = -100:1e-2:100
ys = activation.(xs)
lines!(xs, ys, linewidth=5, color=:red)
f

##
dim = 2
network    = create_network(dim, [25,25],  activation)

"""
Trying to increase the curvature by training the neural create_network
"""

function loss(network::Network, parameters::Vector, samples::Vector{Vector{Float64}})
#     num_events = length(events)

    metric = x -> network2metric(network, x, parameters)

    loss = 0
    for i in 1:length(samples)
        riemann = riemannian(metric, samples[i])

        rms   = riemann.^2 |> mean |> sqrt
        loss -= rms
    end

    return loss / length(samples)
end


r = 1e-1
samples = vec([[x, y] for x in -r:1e-2:r, y in -r:1e-2:r])

num_steps = 100
@time network, losses = train(network, samples, loss, num_steps;
                                     η=5e-1, μ=5e-2)

##
final_losses = Float64[]
for i in 1:length(samples)
    sample_event = samples[i]
    riemann = riemannian(metric, sample_event)

    rms   = riemann.^2 |> mean |> sqrt
    push!(final_losses, rms)
end

# maximum(final_losses)

vals = log.(final_losses)
vals = sort(vals)
plot(vals, cumsum(vals)/sum(vals))
##

tetrad(x)  = forward(network, x)
metric(x)  = network2metric(network, x)
density(x::Vector{Float64}) = metric(x) |> det |> x->-x |> sqrt 

function inside_nsphere(x::Vector{Float64};
        radius::Float64=1.0,
        center::Vector{Float64}=zeros(length(x)),
        norm_p::Float64)

        return norm(x .- center, norm_p) <= radius
end

characteristic(x::Vector{Float64}) = inside_nsphere(x; radius=10.0, norm_p = Inf64)

num_MCMC_samples = 1e6
num_MCMC_samples = convert(Int64, num_MCMC_samples)
MCMC_width = 1.0

MCMC = metropolis(characteristic, density, 
                  num_MCMC_samples, MCMC_width, 
                  zeros(2))

##

num_events = 10000
@assert num_events < num_MCMC_samples
points = MCMC[:, sample(1:num_MCMC_samples, num_events)]
points[:, 1] *= 0

xs = points[1,:]
ys = points[2,:]
f  = Figure()
ax = Axis(f[1,1], aspect=1)
 
source_index = 1
source = points[:, source_index]
ϵ = 10
xlims!(ax, source[1].+[-ϵ,ϵ])
ylims!(ax, source[2].+[-ϵ,ϵ])
scatter!(xs, ys, markersize=2)
f

##

rms_riemann_curvature = Float64[] 
for i in 1:num_events
    riemann = riemannian(metric, points[:, i])
    push!(rms_riemann_curvature, riemann.^2 |> mean |> sqrt)
end


color=colorant"rgb(0,109,91)"
colorscheme = ColorScheme(range(colorant"white", color))

upto = 500
f, ax = plot_field([points[:,i] for i in 1:upto], rms_riemann_curvature[1:upto], [-10,10,-10,10.];
                   color=color)
ax.aspect = 1
xlims!(-10,10)
ylims!(-10,10)

limits = (minimum(rms_riemann_curvature), maximum(rms_riemann_curvature))
Colorbar(f[1,2], colormap=colorscheme, limits=limits)
f

## 


# @save "./network_where_curvature_slices_horizontally.bson" network