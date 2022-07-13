using CairoMakie
using Graphs

import LinearAlgebra
import Random
import StatsBase

include("../lib/Sprinkling.jl")

##

struct SpacetimeBall
    center::Vector{Float64}
    radius::Float64
    norm_p::Float64
end

function χ_ball(coordinates::Vector{Float64}, ball::SpacetimeBall)

    separation_vector = coordinates .- ball.center
    
    return LinearAlgebra.norm(separation_vector, ball.norm_p) <= ball.radius
end

function χ_ball(coordinates::Array{Float64},  ball::SpacetimeBall)
    
    return vec(mapslices(x -> χ_ball(x, ball), coordinates; dims=1))
end

##

mutable struct MCMC
    width::Float64
    last_accepted_coordinates::Vector{Float64}
end

##

struct RandomWalkParameters
    initial_vertex::Int64
    total_num_steps::Int64
    graph::SimpleDiGraph
    directedness::Symbol
end

function random_walk_step(current_vertex::Int64, graph::SimpleDiGraph; 
                            directedness::Symbol=:forward)

    rng = Random.MersenneTwister(45646)

    if directedness == :forward && !isempty(outneighbors(graph, current_vertex))
        return sample(rng, outneighbors(graph, current_vertex))
    end

    if !isempty(inneighbors(graph, current_vertex))
        return sample(rng, inneighbors(graph, current_vertex))
    end

end

function random_walk_step(current_vertex::Int64, params::RandomWalkParameters)
    return random_walk_step(current_vertex, params.graph; directedness=params.directedness)
end

function random_walk(params::RandomWalkParameters)

    walk = [params.initial_vertex]
    
    current_vertex = params.initial_vertex

    for _ in 1:params.total_num_steps
        next_vertex = random_walk_step(current_vertex, params.graph; 
                                         directedness=params.directedness)

        push!(walk, next_vertex)
        current_vertex = next_vertex
    end

    return walk
end

function repeated_random_walks(num_walks::Int64, params::RandomWalkParameters)
    vertex_frequencies = Dict{Int64}{Int64}()

    lk = ReentrantLock()
    @time for _ in 1:num_walks
        walk = random_walk(params)

        begin 
            lock(lk)
            try 
                mergewith!(+, vertex_frequencies, StatsBase.countmap(walk))
            finally
                unlock(lk)
            end
        end

    end

    return vertex_frequencies
end