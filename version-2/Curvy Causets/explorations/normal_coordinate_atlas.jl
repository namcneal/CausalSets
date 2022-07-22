using Flux
using LinearAlgebra

include("../lib/plotting_utils.jl")

##

struct CommonParams
    dim::Int64
    max_frequency::Int64
    network_inputs::Vector{Vector{Int64}}
    network_activation::Function
end

struct FourierData
    params::CommonParams
    network::Flux.Chain
    shifts::Vector{Float32}
end

function CommonParams(dim::Int64, max_frequency::Int64, activation::Function=tanh)
    frequencies = -max_frequency:max_frequency
    
    iterator_arg   = repeat([frequencies], dim)
    inputs = collect(Iterators.product(iterator_arg...))
    inputs = map(tup -> [x for x in tup], inputs) |> vec


    return CommonParams(dim, max_frequency, inputs, activation)
end

function FourierData(params::CommonParams, network_layer_structure::Vector{Int64}=Int64[],
                     maximum_shift::Float64=20.0)
    layer_sizes = [params.dim, network_layer_structure..., 1]

    layers = []
    for i in 1:(length(layer_sizes) - 1)
        layer = Dense(layer_sizes[i] => layer_sizes[i+1], params.network_activation)
        push!(layers, layer)
    end

    return FourierData(params, Chain(layers...), maximum_shift*rand(Float32, length(params.network_inputs)))
end

function (fourier::FourierData)(x::Vector{Float64})
    output = [convert(Float32, 0.0)]
    x = convert.(Float32, x)

    for (i,input) in enumerate(fourier.params.network_inputs)
        coefficient = fourier.network(vec(input))[1]
        θ = dot(x, input)
        output .+= coefficient * sin(θ - fourier.shifts[i]) / (norm(input) + 1)
    end
    
    return output
end

dim = 2
max_frequency = 5
params = CommonParams(dim, max_frequency, tanh)
fourier = FourierData(params, [1])


##

num_points = 2000
sidelength = 10.0
points = 2*sidelength*rand(Float64, (dim, num_points)) .- sidelength
points = [points[:, i] for i in 1:num_points]
@time field_values = [fourier(point)[1] for point in points]

limits = [-sidelength, sidelength, -sidelength, sidelength]
@time f, ax = plot_field(points, convert.(Float64, field_values), limits)
f