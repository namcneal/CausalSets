using Flux
using LinearAlgebra

include("../lib/plotting_utils.jl")
include("../lib/normal_coordinate_utils.jl")

##

struct CommonParams
    dim::Int64
    possible_indices::Vector{Vector{Int}}
    onehot_possible_indices::Flux.OneHotMatrix
end

struct Riemann 
    params::CommonParams
    network::Flux.Chain
end

function CommonParams(dim::Int64)
    index_range= 1:dim
    iterator_arg   = repeat([index_range], dim)
    possible_indices = Iterators.product(iterator_arg...) |> collect |> vec

    possible_indices
    onehots = Flux.onehotbatch(possible_indices, possible_indices)

    return CommonParams(dim, possible_indices, onehots)
end

function Riemann(params::CommonParams,
                 network_layer_structure::Vector{Int64}=Int64[],
                 network_activation::Function=NNlib.mish)

    layer_sizes = [dim^4 + dim, network_layer_structure..., 1]

    layers = []
    for i in 1:(length(layer_sizes) - 1)
        layer = Dense(layer_sizes[i] => layer_sizes[i+1], network_activation)
        push!(layers, layer)
    end

    return Riemann(params, Chain(layers...))

end

# function indices2network_input()
# end

# function (riemann::Riemann)(a::Int64, b::Int64, c::Int64, d::Int64)
#     if (a==b) || (c==d)
#         return 0.0
#     end

# end

CommonParams(2)
# Riemann(4)