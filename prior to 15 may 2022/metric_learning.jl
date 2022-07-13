using CairoMakie
using Colors, ColorSchemes
using GeometryBasics
using LinearAlgebra
using StatsBase
using VoronoiCells

using BSON: @save
include("MetricML.jl")

##

function softplus(x)
    return @. log(1 + exp(x))
end

function leakysoftplus(x, α::Float64=0.3)
    return @. α * x + (1 - α) / softplus(x)
end

##

d = 2
num_data_events = 500
events = [rand(d) .- 0.5  for _ in 1:num_data_events]
Ts     = [rand(d,d)  for _ in 1:num_data_events]
data = (events, Ts)

##
network = create(d, [15,20], leakysoftplus)
losses = Float64[]

##
num_epochs = 5
η = 5e-4
μ = 0.1
grad_clip = 1e-2
network, _losses = @time train(network, data, num_epochs, η, μ; grad_clip=grad_clip);
append!(losses, _losses);
##
plot(log.(losses[losses .< 1] ./ losses[1]))

##
g = p->metric(network, p, network.parameters)
num_test_events = 3000
test_events = [rand(d) .- 0.5 for _ in 1:num_test_events]
L = [loss(network, data[1][i], data[2][i], network.parameters) for i in 1:num_data_events]
ρ = [det(g(event)) for event in test_events]
R = [scalar(g, event) for event in test_events]
# Λ = [sum(T) for T in Ts[]
##

# display(L)

function plot_field(events::Vector{Vector{Float64}}, field_values::Vector{Float64}, 
                    limits::Vector{Float64}; color=colorant"rgb(25,50,118)")

    xmin, xmax, ymin, ymax = limits

    rectangle   = Rectangle(Point2(xmin, ymin), Point2(xmax, ymax))
    tessellation = voronoicells(Point2.(events), rectangle)

    field = field_values
    num_cells   = length(field)
    colorscheme = ColorScheme(range(colorant"white", color))
    colors      = field .- minimum(field)
    colors /= maximum(colors)

    f = Figure()
    Axis(f[1, 1])
    for (i, cell) in enumerate(tessellation.Cells)
        _color = get(colorscheme, colors[i])
        poly!(cell, color=_color, strokecolor=_color, strokewidth=3)
    end
    
    return f 
end


limits = [-0.5, 0.5, -0.5, 0.5]
f = plot_field(test_events, R, limits)
f

limits = [-0.5, 0.5, -0.5, 0.5]
f = plot_field(test_events, R, limits, color=colorant"green")
f
##


description = """
This is a network that's actually working fairly well. The loss decreased a good amount using these parameters.
I can probably tweak it to make it a bit better, but I wanted to save this as a reference:

        d = 2
        num_data_events = 500
        events = [rand(d) .- 0.5  for _ in 1:num_data_events]
        Ts     = [rand(d,d)  for _ in 1:num_data_events]
        data = (events, Ts)

        ##
        network = create(d, [15,20], leakysoftplus)
        losses = Float64[]

        ##
        num_epochs = 1600
        η = 5e-4
        μ = 0.1
        grad_clip = 1e-2
        network, _losses = @time train(network, data, num_epochs, η, μ; grad_clip=grad_clip);
        append!(losses, _losses);

The loss function I used was: 
        function loss(network::MetricNN, point::Vector, stressenergy::Matrix{Float64}, parameters::Vector)
            network2metric = point -> metric(network, point, parameters)
            
            Λ = sum(stressenergy)
            EFE = EFE_LHS(network2metric, point) + Λ * metric(network, point, parameters) #.- stressenergy

            return sum(EFE.^2)
        end

And the training is Nesterov accelerated gradient

"""

@save "Data/first_decent_network.bson" network losses data