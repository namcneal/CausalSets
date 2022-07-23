using Flux
using Einsum
using LinearAlgebra

using CairoMakie

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
                     maximum_shift::Float64=20.0; is_zero_series::Bool=false)

    if is_zero_series
        network = Chain(x->0)
        return FourierData(params, network, zeros(Float32, length(params.network_inputs)))
    end

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

function test_fourier_heatmap()
    dim = 2
    max_frequency = 10
    params = CommonParams(dim, max_frequency, tanh)
    fourier = FourierData(params, [10])
    
    L = 10.0
    dL = 1e-1
    xs = collect(-L:dL:L)
    ys = xs
    num_points = length(xs)
    
    points = [[x; y] for x in xs, y in ys] |> vec
    @time field_values = [fourier(point)[1] for point in points]
    xs = [point[1] for point in points]
    ys = [point[2] for point in points]
    heatmap(xs, ys, field_values)
end

##

mutable struct Riemann
    dim::Int64
    tensor::Dict{Vector{Int64}, FourierData}
    keymap::Dict{Vector{Int64}, Vector{Int64}}
    parities::Dict{Vector{Int64}, Float64}
    mutation_rate::Float64
end

function Base.eachindex(riemann::Riemann) 
    return Iterators.product(repeat([1:riemann.dim], 4)...) 
end

function Riemann(fourier_common_params::CommonParams, default_maximum_fourier_shift::Float64, 
                 mutation_rate::Float64)

    tensor   = Dict()
    keymap   = Dict()
    parities = Dict()

    for index in Iterators.product(repeat([1:fourier_common_params.dim], 4)...) 
        index = [index...]

        # if the current set of indices hasn't been mapped to a tensor element, then 
        # that means none of its pair-wise asymmetric variants (in the same "eq clas" 
        # have not either. So insert that tensor element into the dictionary and map 
        # the whole class to this index in the keymap
        if !haskey(keymap, index)
            fourier_series = FourierData(fourier_common_params, fourier_network_layer_structure, default_maximum_fourier_shift)
            tensor[index]  = fourier_series
    
            first_pair_flipped  = index[[2,1,3,4]]
            second_pair_flipped = index[[1,2,4,3]]

            keymap[index]               = index
            keymap[ first_pair_flipped] = index
            keymap[second_pair_flipped] = index

            parities[index]               =  1.0
            parities[first_pair_flipped]  = -1.0
            parities[second_pair_flipped] = -1.0
        end
            
    end

    return Riemann(fourier_series_common_params.dim, tensor, keymap, parities, mutation_rate)
end


function Base.getindex(riemann::Riemann, a::Int64, b::Int64, c::Int64, d::Int64) 
    if (a==b)||(c==d)
        return x->0.0
    else
        index = riemann.keymap[[a,b,c,d]]
        x -> riemann.tensor[index](x) * riemann.parities[[a,b,c,d]]
    end
end

function (riemann::Riemann)(x::Vector{Float64})
    return map(f->f(x) .* riemann.parities, riemann)
end

dim = 4
max_fourier_frequency = 10
fourier_network_activation = tanh
fourier_network_layer_structure = [10,10]

fourier_series_common_params = CommonParams(dim, max_fourier_frequency, fourier_network_activation)
default_fourier_sin_shift    = 5.0
default_mutation_rate        = 0.5

riemann = Riemann(fourier_series_common_params, default_fourier_sin_shift, default_mutation_rate)

##
@time test_index = [1,2,3,4]
display(riemann[test_index...](ones(dim)))
##
include("../lib/normal_coordinate_utils.jl")
@time first_indexpair_asymmetric(riemann, ones(dim))
 



