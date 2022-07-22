using CairoMakie
using Distributions: Normal
using LinearAlgebra

include("ForwardDiffGR.jl")
using ForwardDiff: gradient

##

""" Defining the NN Architecture and Functions """ 
mutable struct Network
    dimension::Int64
    layer_sizes::Vector{Int64}

    parameters::Vector{Float64}
    weights::Vector{UnitRange{Int64}}
     biases::Vector{UnitRange{Int64}}
    activation::Function
    
    weight_matrices::Vector{Array}
      bias_matrices::Vector{Array}
end

##
# Constructors and initialisors
function init_weights(size::Tuple{Int64, Int64})
    var = 2 / size[1]
    d = Normal(0, var)
    return convert.(Float32, rand(d, size))
end

function init_bias(size::Int64)
    var = 2 / size
    d = Normal(0, var)
    return convert.(Float32, rand(d, size))
end

function softplus(x)
    return @. log(1 + exp(x))
end

function leakysoftplus(x; α::Float64=1.0, β::Float64=1.0, γ::Float64=1.0)
    return @. α * β*x + (1 - α) * softplus(γ*x)
end

function create_network(dimension::Int64, intermediate_layer_sizes::Vector{Int64}, activation::Function)
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
    
    parameter_vector = Float64[]
    weight_indices   = UnitRange{Int64}[]
      bias_indices   = UnitRange{Int64}[]

    i = 0
    for param in [weights..., biases...]
        from = length(parameter_vector) + 1
        to   = from + length(param) - 1

        if i < length(weights)
            push!(weight_indices, from:to)
        else
            push!(bias_indices, from:to)
        end

        i += 1
        parameter_vector = cat(parameter_vector, vec(param), dims=1)
    end

    
    return Network(dimension, layers, 
                   parameter_vector, weight_indices, bias_indices, activation,
                   weights, biases)
        
end

function weight(network::Network, parameters::Vector, index::Int64)
    weight_vector = parameters[network.weights[index]]
    return reshape(weight_vector, size(network.weight_matrices[index]))
end

function bias(network::Network, parameters::Vector, index::Int64)
    return parameters[network.biases[index]]
end

function update_parameter_matrices(network::Network)
    num_transformations = length(network.layer_sizes) - 1
    for i in 1:num_transformations
        @inbounds network.weight_matrices[i] = weight(network, network.parameters, i)
        @inbounds network.bias_matrices[i]   =   bias(network, network.parameters, i)
    end
end

function linear(network::Network, input::VectorInput, parameters::Vector, index::Int64)
    shape = network.layer_sizes[index:index+1]
    shape = reverse(shape) # Reversed due to how linear maps are defined

    # Use the shape of the weight matrix to get the parameters for the iteration
    weight_vector_length = shape[1] * shape[2]
    column_height = shape[1]

    # Initialise the output vector
    # Needed for the forward pass to compute the metric derivatives 
    # with respect to the spacetime coordinates
    if !(input[1] isa Float64 || input[1] isa Vector{Float64})
        output = 0 .* repeat([input[1]], column_height)
    else
        output = parameters[network.biases[index]]
    end    

    input_index = 1
    for row_start in 1:column_height:weight_vector_length
        # This specitic means of adding to the output is necessary for 
        # ForwardDiff to work, due to how the arrays are being converted into 
        # the types needed for autodifferentiation
        output = output + input[input_index] * parameters[row_start:row_start + column_height - 1]
        input_index += 1
    end

    return output 
end

function forward(network::Network, input::VectorInput, parameters::Vector)
    activated = input[:]

    num_transformations = length(network.layer_sizes) - 1
    for i in 1:num_transformations - 1
        @inbounds output = linear(network, activated, parameters, i)
        activated = network.activation.(output)
    end

    @inbounds activated = linear(network,  activated, parameters, num_transformations)

    dimension = length(input)
    return reshape(activated, (dimension, dimension))
end

function forward(network::Network, input::VectorInput)
    activated = input[:]

    num_transformations = length(network.layer_sizes) - 1
    for i in 1:num_transformations - 1
        @inbounds output = network.weight_matrices[i] * activated .+ network.bias_matrices[i]
        activated = network.activation.(output)
    end

    @inbounds activated = network.weight_matrices[end] * activated .+ network.bias_matrices[end]

    dimension = length(input)
    return reshape(activated, (dimension, dimension))
end

# Make the network callable.
(network::Network)(input::VectorInput) = forward(network, input, network.parameters)

function network2metric(network::Network, point::VectorInput, parameters::Vector)
    Q = I + forward(network, point, parameters)
    
    # Flat spacetime metric
    η = Matrix{Float64}(I, size(Q))
    η[1,1] *= -1

    return Q' * η * Q

end

function network2metric(network::Network, point::VectorInput)
    Q = I + forward(network, point)
    
    # Flat spacetime metric
    η = Matrix{Float64}(I, size(Q))
    η[1,1] *= -1

    return Q' * η * Q

end

function train(tetrad::Network, inputs::Vector, loss::Function, 
               num_steps::Int64; η::Float64=1e-2, μ::Float64=0.1, grad_clip::Float64=Inf)

    ℓ  = parameters -> loss(tetrad, parameters, inputs)
    
    losses = Float64[]
    best_network = deepcopy(tetrad)

    velocity = zeros(Float64, size(tetrad.parameters))
    for e in 1:num_steps
        _loss = ℓ(tetrad.parameters)
        push!(losses, _loss)
        print("loss for this batch $(e) was ", _loss, "\n")


        if _loss < minimum(losses)
            best_network = tetrad

        end

        # Look ahead for the gradient
        grad = gradient(ℓ, tetrad.parameters .- μ*velocity)

        grad ./= norm(grad)
        # # Clip
        # grad[grad .> grad_clip] .= grad_clip
        # display(grad .> grad_clip)
        # display(grad)
        # print("\n")

        # Update the velocity
        velocity = -μ * velocity - η * grad
        
        # Update the network parameters themselves
        tetrad.parameters .+= velocity

        update_parameter_matrices(network)
    end

    return best_network, losses
end

##

# d = 2
# num_data_points = 1000
# points = [  rand(d) for _ in 1:num_data_points]
# Ts     = [zeros(d,d) for _ in 1:num_data_points]
# data = (points, Ts)

# ##
# network = create(d, [10,10], logistic)
# num_epochs = 100
# η = 1e-0
# trained, losses = @time train(network, data, num_epochs, η);
# plot(losses)

# ##
# g = p->metric(network, p, network.parameters)
# @time EFE_LHS(g, points[2])]