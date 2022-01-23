using Graphs, SimpleWeightedGraphs
using LongestPaths
using CairoMakie
using GraphPlot
using LinearAlgebra
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

function sprinkle(diamond::CausalDiamond, num_eventss::Int64,
                  event_generator::Function)
    events = Vector{Float64}[]
    
    for i = 1:num_eventss
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

function make_causal_set(sprinkling::Matrix{Float64})
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
    all_descendants = outneighbors(graph, root)
    all_ancestors   = inneighbors(graph, specific_descendant)
    for descendant in all_descendants
        for ancestor in all_ancestors
            if descendant == ancestor
                return false
            end
        end
    end

    return true
end

function remove_indirect_descendants(graph::SimpleDiGraph, root::Int64, sprinkling::Matrix{Float64})
    # Store an initially identical copy of the causal set
    pruned = deepcopy(graph)

    # Get all the outgoing neighbors
    all_descendants = collect(outneighbors(graph, root))

    if length(all_descendants) == 0
        return pruned
    end

    # Sort descendants by distance from root
    descendants_coords = sprinkling[:, all_descendants]
    distances = mapslices(norm, descendants_coords .- sprinkling[:,root]; dims=1)
    all_descendants = all_descendants[sortperm(vec(distances))]

    for descendant in all_descendants
        if !is_direct_child(graph, root, descendant)
            # Remove the edge to the indirect descendant that we tested
            rem_edge!(pruned, root, descendant)
            # deleteat!(all_descendants, all_descendants .== root);

            # Remove all of that descendant's descendants, as they will also be indirect
            for indirect in outneighbors(graph, descendant)
                rem_edge!(pruned, root, indirect);
                # deleteat!(all_descendants, all_descendants .== indirect);
            end 
        end
    end

    return pruned
end

function make_causet_irreducible(causet::SimpleDiGraph, sprinkling::Matrix{Float64})
    
    for event in vertices(causet)
        causet = remove_indirect_descendants(causet, event, sprinkling)
    end

    return causet
end
##

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

function make_sprinkling(tmin::Float64, tmax::Float64, num_events::Int64,
                         sprinkler::Function)

    diamond  = CausalDiamond(tmin, tmax)
    sprinkling = sprinkle(diamond, num_events, sprinkler)

    return sprinkling
end

function make_sprinkling(tmin::Float64, tmax::Float64, num_events::Int64,
                         sprinkler::Function, base_point::Vector{Float64})

    sprinkling = make_sprinkling(tmin, tmax, num_events, sprinkler)
    sprinkling[:, 1] = base_point
    return sprinkling
end


function make_irreducible_causet(tmin::Float64, tmax::Float64, num_events::Int64,
                                 sprinkler::Function, base_point::Vector{Float64})

    sprinkling = make_sprinkling(tmin, tmax, num_events, sprinkler, base_point)

    @time reducible = make_causal_set(sprinkling)
    @time causet    = make_causet_irreducible(reducible, sprinkling)

    return causet
end

function make_irreducible_causet(sprinkling)
    @time reducible = make_causal_set(sprinkling)
    @time causet    = make_causet_irreducible(reducible, sprinkling)

    return causet
end

function hasse_diagram(irreducible_causet::SimpleDiGraph, sprinkling::Matrix{Float64},
                       colors::Vector{Symbol})

    f = Figure(resolution = (800, 800))
    Axis(f[1, 1], backgroundcolor = "white")

    arrow_data = causet2arrows(causet, sprinkling)
    arrows(arrow_data..., color=:black, arrowsize=15, axis=(aspect=1,))
    scatter!(sprinkling[1, :], sprinkling[2,:], color=colors)
    return f
end

##

tmin = -10.
tmax = -1e-2
num_events = 200
sprinkling = make_sprinkling(tmin, tmax, num_events, desitter_event)
sprinkling = sprinkling[[2,1], :]
sprinkling[:, 1] = [-5; -5]
causet = make_irreducible_causet(sprinkling)
geodesic_distances = [find_longest_path(causet, 1, i; log_level=0).upper_bound 
                      for i in 1:num_events]; 
##
colors = repeat([:black], num_events)
colors[geodesic_distances .<= 4] .= :blue 

hasse_diagram(causet, sprinkling, colors)
current_figure()
##

function compute_volume_growth(geodesic_distances::Vector{Core.Real})
    volumes = Int64[]
    distances = collect(1:maximum(geodesic_distances))
    for d in distances
        push!(volumes, sum(geodesic_distances .<= d))
    end

    return reduce(hcat, [distances, volumes])
end

function compute_volume_growth(irreducible_causet::SimpleDiGraph; root::Int64=1)
    
    num_events = nv(irreducible_causet)
    geodesic_distances = [find_longest_path(irreducible_causet, root, i; log_level=0).upper_bound 
                          for i in 1:num_events]; 

    return compute_volume_growth(geodesic_distances)
end

function compute_volume_growth(tmin::Float64, tmax::Float64, num_events::Int64,
    sprinkler::Function, root::Vector{Float64})

    causet =  make_irreducible_causet(tmin, tmax, num_events, sprinkler, root)

    return compute_volume_growth(causet)
end

##
tmin = 0.
tmax = 10.
num_events = 1500
num_trials = 50
minkowski_data = Matrix{Float64}[]
for j in 1:num_trials
    @time data = compute_volume_growth(tmin, tmax, num_events, minkowski_event, [0.; 0])
    push!(minkowski_data, data)
end

##
tmin = -10.
tmax = -1e-2
num_events = 1500
num_trials = 50
desitter_data = Matrix{Float64}[]
for j in 1:num_trials
    @time data = compute_volume_growth(tmin, tmax, num_events, desitter_event, [0.; -10])
    push!(desitter_data, data)
end

##
tmin = -10.
tmax = -1e-2
num_events = 1500
num_trials = 50
antidesitter_data = Matrix{Float64}[]

for j in 1:num_trials
    sprinkling = make_sprinkling(tmin, tmax, num_events, desitter_event)
    sprinkling = sprinkling[[2,1], :]
    sprinkling[:, 1] = [tmin, tmin]
    causet = make_irreducible_causet(sprinkling)
    geodesic_distances = [find_longest_path(causet, 1, i; log_level=0).upper_bound 
                      for i in 1:num_events]; 

    @time data = compute_volume_growth(geodesic_distances)
    push!(antidesitter_data, data)
end
##
plotting_data_minkowski = reduce(vcat, minkowski_data)
plotting_data_desitter  = reduce(vcat, desitter_data)
plotting_data_antidesitter  = reduce(vcat, antidesitter_data)

scatter( plotting_data_minkowski[:,1],     plotting_data_minkowski[:,2],    color=:blue)
scatter!(plotting_data_desitter[:,1],      plotting_data_desitter[:,2],     color=:green)
scatter!(plotting_data_antidesitter[:,1],  plotting_data_antidesitter[:,2], color=:red)

current_figure()

## 

description = """
These three data sets were computed to determine how the volume of a 2D geodesic ball 
varies with its radius in causal sets derived from Minkowski, de Sitter, and anti-de Sitter
spaces. In the continuum, the scalar curvature is related to the rate of change of this
volume with respect to the radius. Using causal sets spinkled into these three spaces
according to the procedure outlined in Dhital's undergraduate honors thesis, I found that 
the curvature of these spaces can be seen in the volumetric growth of the geodesic ball
within the causal diamonds that I use for the finite spacetime region. This gives a promising 
result.

Each dataset contains the volume data for 50 causal sets with 1500 events. The Minkowski time
bounds are [0,10] and the de Sitter are [-10, -1e-2]. The anti-de Sitter space was created
by rotating the de Sitter spacetime and placing a root event at the appropriate corner of the 
causal diamond. The statistics for the three spacetimes were

Creation of the causal set
==========================
* Minkowski      : 1-2 s
* de Sitter      : 1-1.5 s
* anti-de Sitter : 1-1.5 s

Reduction of decomposable morphisms
==========================
* Minkowski      : 15-20 s
* de Sitter      : 35-40 s
* anti-de Sitter : 5-5.5 s

Total time to compute the volumetric growth rate
==========================
* Minkowski      : 300-400 s
* de Sitter      : 500-550 s
* anti-de Sitter : 0.003-0.006 s -> This function being timed takes the pre-computed geodesic distances, 
                                    so its much faster. I should have timed the function that computed 
                                    these distances
"""

@save  "geodesic_ball_volume_growth_three_spacetimes_1500_events_13_jan_2022.bson" description minkowski_data desitter_data antidesitter_data
