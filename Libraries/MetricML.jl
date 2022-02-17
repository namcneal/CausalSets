using Distributions: Normal
using LinearAlgebra
##

Input = Union{Float32, 
              Float64,
              Vector{Float32},          
              Vector{Float64},
              ReverseDiff.TrackedArray,
              Vector{ReverseDiff.TrackedReal}
             }

ParamInput = Union{Param{Matrix{Float32}}, 
                   Param{Vector{Float32}},
                   Vector{Param{Matrix{Float32}}}, 
                   Vector{Param{Vector{Float32}}},
}


function logistic(x)
    return 1 / (1 + exp(-x))
end

##

""" Defining the NN Architecture and Functions """ 
mutable struct MetricNN
    dimension::Int64
    layer_sizes::Vector{Int64}

    weights::Vector{Param{Matrix{Float32}}}
    biases::Vector{Param{Vector{Float32}}}

    activation::Function
end

# Constructors and initialisors
function init_weights(size::Tuple{Int64, Int64})
    var = 2 / size[1]
    d = Normal(0, var)
    return Param(convert.(Float32, rand(d, size)))
end

function init_bias(size::Int64)
    var = 2 / size
    d = Normal(0, var)
    return Param(convert.(Float32, rand(d, size)))
end

function create(dimension::Int64, intermediate_layer_sizes::Vector{Int64}, activation::Function)
    d = dimension
    input_dim  = d
    output_dim = d^2

    layers   = [input_dim, intermediate_layer_sizes..., output_dim]
    num_maps = length(layers) - 1

    all_dims = Tuple{Int64, Int64}[]
    for i in 1:num_maps
        input_dim  = layers[i]
        output_dim = layers[i+1]
        
        push!(all_dims, (input_dim, output_dim))

    end
    
    weights = [init_weights(reverse(dims)) for dims in all_dims]
    biases  = [init_bias(dims[2]) for dims in all_dims]
    
    return MetricNN(dimension, layers, weights, biases, activation)
        
end

function vector2symmetric(dimension::Int64, vector::Input)
    return [ i<=j ? vector[j*(j-1) ÷ 2 + i] : vector[i*(i-1) ÷ 2 + j] for i=1:dimension, j=1:dimension]
end
function vector2symmetric(vector::Input)
    n = convert(Int64, sqrt(length(vector)))
    return vector2symmetric(n, vector)
end
function vector2symmetric(vector)
    n = convert(Int64, sqrt(length(vector)))
    return vector2symmetric(n, vector)
end

function linear(input::Input,  weight::Matrix{Float32}, bias::Vector{Float32})
    return weight * input .+ bias
end

function forward(input::Input, weights::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, activation::Function)
    activated = input[:]

    num_transformations = length(weights)
    for i in 1:num_transformations - 1
        @inbounds output = linear(activated, weights[i], biases[i])
        activated = activation.(output)
    end
    @inbounds activated = linear(activated, weights[end], biases[end])

    dimension = length(input)
    return reshape(activated, (dimension, dimension))
end


# Make the network callable.
(network::MetricNN)(input) = forward(input, network.weights, network.biases, network.activation)

##
function metric(network::MetricNN, point::Input)
    d = network.dimension
    num_inner_layers = length(network.layer_sizes) - 1

    # Flat spacetime metric
    η = Matrix{Float64}(I, d, d)
    η[1,1] *= -1

    # Matrix whose columns give the eigenvectors of the metric
    Q = forward(point, network.weights, network.biases, network.activation)

    return Q * η * Q'
end

function LHS(metric::Function, point::Input)
    gij = metric(point)
    Rij = ricci(metric, point)
    R   = scalar(metric, point)

    LHS = @. Rij - R * gij
    
    return LHS
end


function loss(network::MetricNN, point::Input, stressenergy::Matrix{Float64})
    network2metric(point) = metric(network, point)
    
    Tij = stressenergy
    EFE = LHS(network2metric, point) .- Tij

    return sum(EFE.^2)

end

d = 4
point = [0, 2000, π/2, 0]
network = create(d, [12], logistic)
Tij = (A->A'*A)(rand(d,d))

