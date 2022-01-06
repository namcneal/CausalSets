using Graphs, SimpleWeightedGraphs
using CairoMakie
using GraphPlot
using Random
using BSON: @save, @load

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

function causal_distance(vector_A2B::Vector{Float64})
    η = [-1 0; 0 1.]
    return vector_A2B' * η * vector_A2B
end

function make_causal_set(diamond::CausalDiamond, sprinkling::Matrix{Float64})
    num_points = size(sprinkling)[2]
    
    # Initiate an empty directed graph 
    causal_set = SimpleDiGraph(num_points)
    
    # Check each causal relation
    for i = 1:num_points
        for j = 1:num_points
            vector_i2j = sprinkling[:, j] - sprinkling[:, i]
            interval = causal_distance(vector_i2j)
            
            if  interval >= 0 && vector_i2j[2] > 0
                add_edge!(causal_set, i, j)
            end
        
        end
    end 
            
    return causal_set
end
##

function is_direct_child(graph::SimpleDiGraph, root::Int64, specific_descendant::Int64)
    for descendant in outneighbors(graph, root)
        for ancestor in inneighbors(graph, specific_descendant)
            if descendant == ancestor
                return false
            end
        end
    end

    return true
end

function remove_indirect_descendants(graph::SimpleDiGraph, root::Int64)
    pruned = deepcopy(graph)

    for descendant in outneighbors(graph, root)
        if !is_direct_child(graph, root, descendant)
            # Remove the edge to the indirect descendant that we tested
            rem_edge!(pruned, root, descendant)

            # Remove all of that descendant's descendants, as they will also be indirect
            for indirect in outneighbors(graph, descendant)
                rem_edge!(pruned, root, indirect);
            end 
        end
    end

    return pruned
end

function make_causet_irreducible(causet::SimpleDiGraph)
    
    for event in vertices(causet)
        causet = remove_indirect_descendants(causet, event)
    end

    return causet
end
##

function compute_volume_growth(irreducible_causet::SimpleDiGraph, max_distance::Int64=-1; root::Int64=1)
    if max_distance < 0
        max_distance = nv(irreducible_causet)
    end

    volumes = Int64[]
    for d = 1:max_distance
        push!(volumes, length(neighborhood(causet, root, d)))
    end

    
    return volumes
end

function causet2arrows(causet::SimpleDiGraph, sprinkling::Matrix{Float64})
    arrow_data = Vector{Float64}[]

    for edge in edges(causet)
        x,y = sprinkling[:, src(edge)]

        u,v = sprinkling[:, dst(edge)] - sprinkling[:, src(edge)] 

        push!(arrow_data, [x; y; u; v])
    end

    return [reduce(hcat, arrow_data)[i, :] for i in 1:4]
end

##
diamond  = CausalDiamond(0, 10)
sprinkling = sprinkle(diamond, 7500,  minkowski_event)

base_point = [0; 0]
sprinkling[:, 1] = base_point

scatter(sprinkling[1, :], sprinkling[2,:], color=:black, axis=(aspect=1,) 
        )

##
@time reducible = make_causal_set(diamond, sprinkling)

##
# @time causet = make_causet_irreducible(reducible)

description = "Description: This is a 7500-event causal set saved in two forms. The first is the causal set 
with all poset morphisms. It took 39.5 s to generate from the spacetime sprinkling. It was further
processed by removing all the reducible/decomposable morphisms, a process that took 1 hr and 23 min 
to compute. The number of morphisms went from 1,410,3547 to 55,806, leaving only 3.9%."

@save "causal_set_5_jan_2021.bson" description reducible causet

a
##

# f = Figure(resolution = (800, 800))
# Axis(f[1, 1], backgroundcolor = "white")
# scatter(sprinkling[1, :], sprinkling[2,:], color=:black)
# arrow_data = causet2arrows(causet, sprinkling)
# arrows(arrow_data..., color=:black, axis=(aspect=1,), arrowsize=15)

##
plot(compute_volume_growth(causet, 10))