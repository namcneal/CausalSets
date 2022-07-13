using Graphs, SimpleWeightedGraphs
using CairoMakie
using Random
using BSON

Real = Union{Float64, Int64}
##

struct CausalDiamond
    tmin::Float64
    tmax::Float64
    xmin::Float64
    xmax::Float64
end

function CausalDiamond(tmin::Real, tmax::Real)
    xmin = (tmin - tmax) / 2
    xmax = (tmax - tmin) / 2
    
    return CausalDiamond(tmin, tmax, xmin, xmax)
end

function in_diamond(diamond::CausalDiamond, event::Vector{Float64})
    t = event[2]
    x = event[1]
    
    return (t-diamond.tmin)^2 > x^2 && (t-diamond.tmax)^2 > x^2
end


##
function minkowski_event(diamond::CausalDiamond)
    r = rand(Float64)
    t = r * (diamond.tmax - diamond.tmin) + diamond.tmin
    
    r = rand(Float64)
    x = r * (diamond.xmax - diamond.xmin) + diamond.xmin
    
    return [x; t]
end

function desitter_event(diamond::CausalDiamond)
    r = rand(Float64)
    t = diamond.tmin*diamond.tmax 
    t /= r * (diamond.tmin - diamond.tmax) + diamond.tmax
    
    r = rand(Float64)
    x = r * (diamond.xmax - diamond.xmin) + diamond.xmin
    
    return [x; t]
end

function sprinkle(diamond::CausalDiamond, num_events::Int64,
                  event_generator::Function)
    events = Vector{Float64}[]
    
    for i = 1:num_events
        while length(events) < i
            event = event_generator(diamond)
            
            if in_diamond(diamond, event)
                push!(events, event)
            end
        end
    end
    
    return reduce(hcat, events)
end

##

##
diamond  = CausalDiamond(-10, -1e-2)
sprinkling = sprinkle(diamond, 1000,  desitter_event)

scatter(sprinkling[1, :], sprinkling[2,:],  color=:black,
        axis=(aspect=1,)
        )


##

function causal_distance(vector_A2B::Vector{Float64})
    η = [1 0; 0 -1.]
    return vector_A2B' * η * vector_A2B
end

function make_causal_set(diamond::CausalDiamond, sprinkling::Matrix{Float64})
    num_points = size(sprinkling)[2]
    
    # Initiate an empty directed graph 
    causal_set = SimpleWeightedDiGraph(num_points)
    
    # Check each causal relation
    for i = 1:num_points
        for j = 1:num_points
            vector_i2j = sprinkling[:, j] - sprinkling[:, i]
            interval = causal_distance(vector_i2j)
            
            if  interval >= 0 && vector_i2j[2] > 0
                add_edge!(causal_set, i, j, -1.0)
            end
        
        end
    end 
            
    return causal_set
end

##

causal_set = make_causal_set(diamond, sprinkling)

##

function is_direct_child(graph, source::Int64, candidate::Int64)
    for descendant in outneighbors(graph, source)
        for ancestor in inneighbors(graph, candidate)
            if descendant == ancestor
                return true
            end
        end
    end

    return false
end

function filter_indirect_descendants(graph, source::Int64)
    remaining_descendants = collect(outneighbors(graph, source))

    for candidate in remaining_descendants
        if !is_direct_child(graph, source, candidate)
            # Remove the indirect descendant that we tested
            deleteat!(remaining_descendants,     remaining_descendants .== candidate);

            # Remove all of that descendant's descendants, as they will also be indirect
            for indirect in outneighbors(graph, candidate)
                deleteat!(remaining_descendants, remaining_descendants .== indirect);
            end 
        end
    end

    return remaining_descendants
end

function remove_indirect_descendants(graph, source::Int64)
    pruned = deepcopy(graph)

    for candidate in outneighbors(pruned, source)
        if !is_direct_child(graph, source, candidate)
            # Remove the edge to the indirect descendant that we tested
            rem_edge!(pruned, source, candidate)

            # Remove all of that descendant's descendants, as they will also be indirect
            for indirect in outneighbors(graph, candidate)
                rem_edge!(pruned, source, indirect;
            end 
        end
    end

    return pruned
end

##

# outneighbors(causal_set, 1)

# inneighbors(causal_set, 5)

@time filter_indirect_descendants(causal_set, 1)


