using LinearAlgebra
using Random

# Graph theory
using LightGraphs
include("dag_utils.jl")

## 

function minkowski(point::Vector{Float64})
    return [-1 0;
             0 1.]
end

function deSitter(point::Vector{Float64})
    H = sqrt(1 / 12)
    g = [-1 / H^2 / point[2]^2 0; 
         0 1 / H^2 / point[2]^2]

    return g
end

function anti_deSitter(point::Vector{Float64})
    H = -sqrt(1 / 12)
    g = [-1 / H^2 / point[1]^2 0; 
         0 1 / H^2 / point[1]^2]

    return g
end

##
function density(region::DiamondBounds, R::Float64, point::Vector{Float64})
    if abs(R) < 1e-6
        g = det(minkowski(point))
    else
        # We need to shift the causal diamond so that the top point 
        # is just below t = 0 and the bottom point is at negative time
        t = point[2]
        x = point[1]

        distance_from_tmin = t - region.tmin
        distance_from_xmin = x - region.xmin

        new_tmin = region.tmin - region.tmax
        # de Sitter space has time and space like Minkowski
        if R > 0
            t = new_tmin + distance_from_tmin - 1e-2

            g = det(deSitter([x; t]))
        end

        # anti-de Sitter space has the volumetric role of space
        # and time reversed. We need to move each point 
        if R < 0
            x = new_tmin + distance_from_xmin - 1e-2
            t = region.tmin + distance_from_xmin
            
            g = det(anti_deSitter([x,t,x]))
        end
    end

    return sqrt(-g)
end

function make_causet(event_coordinates::Matrix{Float64}, metric::Function)
    num_events = size(event_coordinates)[2]

    # Sort the events by their time ordering in this coordinate frame
    time_ordering = sortperm(event_coordinates[1,:])
    event_coordinates = event_coordinates[:, time_ordering]
    
    # Initiate an empty directed graph's adjacency_matrix
    adjacency  = LightGraphs.adjacency_matrix(SimpleDiGraph(num_events))
    
    # Check each causal relation
    index_pairs = [(i,j) for i in 1:num_events for j in i+1:num_events]
    
    lk = ReentrantLock()
    Threads.@threads for (i,j) in index_pairs
        vector_i2j = event_coordinates[:, j] - event_coordinates[:, i]
        interval   = vector_i2j' * metric([0.0;0.0]) * vector_i2j

        if interval <= 0
            begin 
                lock(lk)
                try
                    adjacency[i, j] = 1
                finally
                    unlock(lk)
                end
            end
        end

    end 
            
    return event_coordinates, SimpleDiGraph(adjacency)
end










