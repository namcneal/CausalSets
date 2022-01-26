using Graphs, SimpleWeightedGraphs
using Graphs.LinAlg
using LongestPaths
using DataStructures
include("utils/GraphAlgs.jl")

using CairoMakie
using GraphPlot
using LinearAlgebra
using Random
using BSON: @save, @load
import Plots

using Turing
using Statistics: mean, std

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

function get_adjacency_powers(causet::SimpleDiGraph, max_k::Float64=Inf)
    adj_matrix  = convert.(UInt128, adjacency_matrix(causet))
    power_k = adj_matrix[:,:]
    all_powers = Matrix{UInt128}[]

    k = 1
    push!(all_powers, power_k)
    while norm(power_k) > 0.5 && k < max_k
        power_k     *= adj_matrix
        push!(all_powers, power_k)
    end

    return cat(all_powers...; dims=3)
end

function get_numberof_paths(causet::SimpleDiGraph, max_k::Float64=Inf)
    adj_matrix  = convert.(UInt128, adjacency_matrix(causet))
    path_matrix = adj_matrix[:,:]

    k = 1
    power_k     = adj_matrix[:,:]
    while norm(power_k) > 0.5 && k < max_k
        power_k     *= adj_matrix
        path_matrix += power_k
        k+=1 
    end

    return path_matrix
end

function get_numberof_paths(adjacency_powers::Array{UInt128}, max_k::Float64=Inf)
    return sum(adjacency_powers; dims=3)[1,:,1]
end

function get_geodesic_distances(causet::SimpleDiGraph, root::Int64=1)

    num_events = nv(causet)
    adjacency_powers = get_adjacency_powers(causet)
    num_layers = size(adjacency_powers)[3]
    
    geodesic_distances = zeros(Int64, num_events)
    for event in 1:num_events
        paths_to_root = adjacency_powers[root, event, :]
        for i in num_layers:-1:1
            if paths_to_root[i] > 0
                geodesic_distances[event] = i
                break
            end
        end
    end

    return geodesic_distances
end

end

##

tmin = 0.
tmax = 10.
num_events = 500
sprinkling = make_sprinkling(tmin, tmax, num_events, minkowski_event)
# sprinkling = sprinkling[[2,1], :]
sprinkling[:, 1] = [0; 0]
causet = make_irreducible_causet(sprinkling)

##


 

##
function geodesic_distances_from_adjacency(causet::SimpleDiGraph, root::Int64=1)
    

@time geodesic_distances_from_adjacency(causet);

@time geodesic_distances = [find_longest_path(causet, 1, i; log_level=0).upper_bound 
                      for i in 1:num_events]; 
# geodesic_distances = convert.(Float64, geodesic_distances)





##
# y = convert.(Float64, log10.(num_paths))
# plot(geodesic_distances, y);

# y = log10.(num_paths .+ 1)
# y = reshape(y, length(y), 1)

@model linreg(X, y; predictors=size(X, 2)) = begin
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))
    β ~ filldist(TDist(3), predictors)
    σ ~ Exponential(1)

    #likelihood
    y ~ MvNormal(α .+ X .* β, σ)
end;


model = linreg(geodesic_distances, y);
chain = sample(model, NUTS(), MCMCThreads(), 2_000, 4)
slope     = summarystats(chain)[2,:][1]
intercept = summarystats(chain)[1,:][1]
xs = collect(1:26)
ys = intercept .+ xs * slope
plot(geodesic_distances, y);
lines!(xs, ys, linewidth=5, color=:green)
current_figure()