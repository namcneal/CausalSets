using CairoMakie
using Distributions: Normal
using LinearAlgebra

include("ForwardDiffGR.jl")
using ForwardDiff: gradient



##

""" Defining the NN Architecture and Functions """ 
mutable struct MetricNN
    dimension::Int64
    layer_sizes::Vector{Int64}

    parameters::Vector{Float64}
    weights::Vector{UnitRange{Int64}}
     biases::Vector{UnitRange{Int64}}
    activation::Function
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

    
    return MetricNN(dimension, layers, parameter_vector, weight_indices, bias_indices, activation)
        
end


function weight(network::MetricNN, parameters::Vector, index::Int64)


    weight_vector = parameters[network.weights[index]]
    return reshape(weight_vector, (shape...))
end

function bias(network::MetricNN, parameters::Vector, index::Int64)
    return parameters[network.biases[index]]
end


function linear(network::MetricNN, input::VectorInput, parameters::Vector, index::Int64)
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

function forward(network::MetricNN, input::VectorInput, parameters::Vector)
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

# Make the network callable.
(network::MetricNN)(input) = forward(network, input, network.parameters)

function metric(network::MetricNN, point::VectorInput, parameters::Vector)
    d = network.dimension

    # Flat spacetime metric
    η = Matrix{Float64}(I, d, d)
    η[1,1] *= -1

    # Matrix whose columns give the eigenvectors of the metric
    Q = forward(network, point, parameters)

    return Q * η * Q'
end

function loss(network::MetricNN, point::Vector, stressenergy::Matrix{Float64}, parameters::Vector)
    network2metric = point -> metric(network, point, parameters)
    
    Λ = sum(stressenergy)
    EFE = EFE_LHS(network2metric, point) + Λ * metric(network, point, parameters) #.- stressenergy

    return sum(EFE.^2)
end

function loss(network::MetricNN, data::Tuple{Vector{Vector{Float64}}, Vector{Matrix{Float64}}}, parameters::Vector)
   
    num_data_points = length(data)
    return sum([loss(network, data[1][i], data[2][i], parameters) for i in 1:num_data_points]) / num_data_points
end

function random_stressenergy(d::Int64)
    return (A->A'*A)(rand(d,d))
end

function train(network::MetricNN, data::Tuple{Vector{Vector{Float64}}, Vector{Matrix{Float64}}}, 
               num_epochs::Int64, η::Float64=1e-2, μ::Float64=0.1; grad_clip::Float64=Inf)

    ℓ  = params -> loss(network, data, params)
    
    losses = Float64[]
    best_network = deepcopy(network)

    velocity = zeros(Float64, size(network.parameters))
    for e in 1:num_epochs
        epoch_loss = ℓ(network.parameters)
        push!(losses, epoch_loss)

        if epoch_loss > maximum(losses)
            best_network = network
        end

        # Look ahead for the gradient
        grad = gradient(ℓ, network.parameters .- μ*velocity)

        grad ./= norm(grad)
        # # Clip
        # grad[grad .> grad_clip] .= grad_clip
        # display(grad .> grad_clip)
        # display(grad)
        # print("\n")

        # Update the velocity
        velocity = -μ * velocity - η * grad
        
        # Update the network parameters themselves
        network.parameters .+= velocity
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