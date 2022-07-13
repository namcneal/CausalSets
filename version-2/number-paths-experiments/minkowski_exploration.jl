using CairoMakie
using Distributions
using DelimitedFiles
using LightGraphs
using Statistics

include("../lib/DiamondBounds.jl")
include("../lib/metropolis.jl")

include("../lib/conformally_flat_utils.jl")
include("../lib/causet_utils.jl")


##

const min_time  = -1.0
const max_time  =  1.0
const num_spacial_dimensions = 1
diamond = DiamondBounds(min_time, max_time, num_spacial_dimensions)

flat_space_density(x)     = density(     diamond, 0.0, x)
diamond_characteristic(x) = is_in_diamond(diamond, x)
num_MCMC_samples = 10000
metropolic_width = 0.1
initial_coordinates = [0.0; 0.0]

MCMC = metropolis(diamond_characteristic, flat_space_density, 
                  num_MCMC_samples, metropolic_width,
                  initial_coordinates)

##
num_events = 3000
@assert num_events < num_MCMC_samples
events = MCMC[:, sample(1:num_MCMC_samples, num_events)]

lower_bound_coordinates = [-1.0; 0.0]
# upper_bound_coordinates = [ 1.0; 0.0]
events[:, 1]   = lower_bound_coordinates
# events[:, end] = upper_bound_coordinates

@time events, causet = make_causet(events, minkowski)
causet = transitivereduction(causet)

##

function count_paths_from_lower_bound(causet::SimpleDiGraph)
    adjacency = adjacency_matrix(causet)
    current_adjacency_power = adjacency
    counts                  = convert.(UInt128, adjacency)

    while sum(current_adjacency_power) > 0
        current_adjacency_power *= adjacency
        counts .+= current_adjacency_power
    end

    return counts[1, 2:end]
end

function geodesic_distance_from_lower_bound(event_coordinates::Matrix{Float64})
    num_events = size(event_coordinates)[2]

    distances  = Float64[]
    for i in 2:num_events
        vector_from_lower_bound = event_coordinates[:, i] - event_coordinates[:, 1]
        interval   = vector_from_lower_bound' * minkowski([0.0;0.0]) * vector_from_lower_bound

        push!(distances, interval)
    end

    return sqrt.(-distances)
end

num_paths = count_paths_from_lower_bound(causet)
data      = collect(log.(num_paths))
distances = geodesic_distance_from_lower_bound(events)

f = Figure()

ax = Axis(f[1, 1], xlabel="Geodesic Disance", ylabel="Log-Number of Paths")

plot!(distances, data, color="black", markersize=3.0)

f
     

##

""" Model fitting 

Assume the points have been drawn from a logistic function that
depends on three parameters. 

https://en.wikipedia.org/wiki/Bayesian_linear_regression

"""

function logistic(xs::Vector{Float64}, A::Float64, b::Float64, s::Float64, y0::Float64)
    arg = -b * (xs .- s)

    return y0 .+  A ./ (1 .+ exp.(arg))
end

function prob_data(data::Vector{Float64}, xs::Vector{Float64}, 
                   A::Float64, b::Float64, s::Float64, y0::Float64, 
                   σ::Float64)

    #  Predictions from the model
    predicted = logistic(xs, A, b, s, y0)
    
    # The difference b/w the model and the data
    diff = data .- predicted

    # Feed this difference into a normal distribution to add noise
    arg  = diff' * diff
    arg /= (2 * σ^2)
    num_data_points = length(data)
    coeff = 1 / (σ^2 * 2 * π)^(num_data_points / 2)
    
    return coeff * exp(-arg)
end

function log_prob_data(data::Vector{Float64}, xs::Vector{Float64}, 
                       A::Float64, b::Float64, s::Float64, y0::Float64, 
                       σ::Float64)
    #  Predictions from the model
    predicted = logistic(xs, A, b, s, y0)
    
    # The difference b/w the model and the data
    diff = data .- predicted

    # Feed this difference into a normal distribution to add noise
    arg  = diff' * diff
    variance = σ^2
    arg /= (2 * variance )
    num_data_points = length(data)
    coeff = 1 / (variance  * 2 * π)^(num_data_points / 2)
    
    return log(coeff) - arg
end

function normal(x::Float64, μ::Float64, σ::Float64)
    coeff = 1 / σ / sqrt(2*π)
    arg = x .- μ
    arg = (arg / σ).^2

    return coeff * exp.(-arg ./ 2)
end

function prior_A(A::Float64)
    return normal(A, 38.0, 2.0)
end

function prior_b(b::Float64)
    return normal(b, 5.0, 4.0)
end

function prior_s(s::Float64)
    return normal(s, 0.0, 1.0)
end

function prior_σ(σ::Float64)
    return normal(σ, 0.0, 1.0)^2
end

function prior_y0(y0::Float64)
    return normal(y0, 0.0, 4.0)
end

function log_posterior(A::Float64, b::Float64, s::Float64, y0::Float64, σ::Float64, 
                       xs::Vector{Float64}, data::Vector{Float64})
    log_post  = log_prob_data(data, xs, A, b, s, y0, σ)
    
    log_post += log(prior_A(A)) + log(prior_b(b)) + log(prior_s(s)) + log(prior_y0(y0))
    log_post += log(prior_σ(σ))

    return log_post
end

function log_posterior(point::Vector{Float64}, xs::Vector{Float64}, data::Vector{Float64})    
    A, b, s, y0, σ = point

    return log_posterior(A, b, s, y0, σ, xs, data)
end

num_MCMC_samples = 2000000
metropolis_width = 0.01

# Coordinates are of the form (A, b, s, y0, σ)
initial_coordinates = [44.0, 3.0, 0.5, -10, 0.5]
# initial_coordinates = all_samples[:, end]

characteristic(x) =  true
prob_density(point::Vector{Float64}) = log_posterior(point, distances, data)


MCMC_param_samples = metropolis(characteristic, prob_density, 
                                   num_MCMC_samples, metropolis_width,
                                   initial_coordinates, log_density=true)
##

probs = [prob_density(MCMC_param_samples[:, i]) for i in 1:num_MCMC_samples]

slopes = MCMC_param_samples[2,:]
lines(slopes)

##

mode_params = MCMC_param_samples[:, argmax(probs)] 
mean_params = mean(MCMC_param_samples, dims=2)

A, b, s, y0, σ = mean_params
xs = collect(0:0.01:2.0)
ys = logistic(xs, A, b, s, y0)

f = Figure()

ax = Axis(f[1, 1], xlabel="Geodesic Disance", ylabel="Log-Number of Paths")

plot!(distances, data, color="black", markersize=3.0)
lines!(xs, ys, color="orange", linewidth=5)
f

##


## 

all_samples = MCMC_param_samples
all_probs   = probs

mode_params = all_samples[:, argmax(all_probs)] 

print("Largest log-posterior: ", maximum(probs), "\n\n")
display(mode_params)


A, b, s, y0, σ = mean_params
xs = collect(0:0.01:2.0)
ys = logistic(xs, A, b, s, y0)

f = Figure()

ax = Axis(f[1, 1], xlabel="Geodesic Disance", ylabel="Log-Number of Paths")

plot!(distances, data, color="black", markersize=4.0)
lines!(xs, ys, color="red", linewidth=5)

f

## 
function write_vector_to_file(vector::Vector{Float64}, filename::String)
    io = open(filename, "w") do io
        for x in vector
          println(io, x)
        end
    end
end

write_vector_to_file(all_probs,   "probabilities.txt")
writedlm("MCMC_param_samples.txt", all_samples, ',')




