function topologically_sort_dag(dag::SimpleDiGraph)
    A = adjacency_matrix(dag)
    sorted_order  = topological_sort_by_dfs(dag)
    
    B = zeros(Int64, size(A))
    for i in 1:nv(dag), j in 1:nv(dag)
        B[i,j] = A[sorted_order[i],sorted_order[j]]
    end
    
    return SimpleDiGraph(B)
end

# function get_adjacency_powers(causet::SimpleDiGraph, max_k::Float64=Inf)

#     adj_matrix  = convert.(UInt128, adjacency_matrix(causet))
#     all_powers = Matrix{UInt128}[adj_matrix]
    
#     power_k = adj_matrix[:,:]
#     while norm(power_k) > 0.5 && k < max_k
#         power_k     *= adj_matrix
#         push!(all_powers, power_k)
#     end

#     return cat(all_powers...; dims=3)
# end

# function compute_number_of_paths(causet::SimpleDiGraph, max_k::Float64=Inf)
#     adj_matrix  = convert.(UInt128, adjacency_matrix(causet))
#     path_matrix = adj_matrix[:,:]

#     k = 1
#     power_k     = adj_matrix[:,:]
#     while norm(power_k) > 0.5 && k < max_k
#         power_k     *= adj_matrix
#         path_matrix += power_k
#         k+=1 
#     end

#     return path_matrix
# end


# function compute_numberof_paths(adjacency_powers::Array{UInt128}, max_k::Float64=Inf)
#     return sum(adjacency_powers; dims=3)[1,:,1]
# end

# function compute_geodesic_distances(causet::SimpleDiGraph, root::Int64=1)

#     num_events = nv(causet)
#     adjacency_powers = get_adjacency_powers(causet)
#     num_layers = size(adjacency_powers)[3]
    
#     geodesic_distances = zeros(Int64, num_events)
#     for event in 1:num_events
#         paths_to_root = adjacency_powers[root, event, :]
#         for i in num_layers:-1:1
#             if paths_to_root[i] > 0
#                 geodesic_distances[event] = i
#                 break
#             end
#         end
#     end

#     return geodesic_distances
# end

# function compute_geodesic_distances(adjacency_powers::Array{UInt128}, root::Int64=1)
#     num_events = size(adjacency_powers)[1]
#     num_layers = size(adjacency_powers)[3]
    
#     geodesic_distances = zeros(Int64, num_events)
#     for event in 1:num_events
#         paths_to_root = adjacency_powers[root, event, :]
#         for i in num_layers:-1:1
#             if paths_to_root[i] > 0
#                 geodesic_distances[event] = i
#                 break
#             end
#         end
#     end

#     return geodesic_distances
# end
