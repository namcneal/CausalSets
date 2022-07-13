using LightGraphs
using CairoMakie

import SparseArrays
import LinearAlgebra

const SparseMatrix = SparseArrays.SparseMatrixCSC{Int64, Int64}
##

function all_binary_numbers_up_to(n; include_0::Bool=false)
    start = convert(Int64, !include_0)
    
    numbers = String[]
    for i = start:2^n-1
       s = bitstring(i)
       s = s[end-n+1:end]
       
       push!(numbers, s)
    end

    return numbers
end

function binary_num_to_vector(num::String)
    return [convert(Int64, char=='1') for char in num]
end

function embed_sparse_matrix(smaller::SparseMatrix, larger::SparseMatrix)
    rows, cols, _ = SparseArrays.findnz(smaller)
    num_entries = length(rows)

    for i in 1:num_entries
        larger[rows[i], cols[i]] = 1
    end

    return larger
end

function sparse_matrix_characteristic_polynomial(matrix::SparseMatrix, x::Float64)
    singular_at_eigenvalues = convert.(Float64, matrix[:,:])
    
    for i in 1:size(matrix)[1]
        singular_at_eigenvalues[i,i] = -x
    end

    return LinearAlgebra.det(singular_at_eigenvalues)
end

##
function generate_all_dag_adjs(max_num_nodes::Int64)
    root_graph = SimpleDiGraph(1)

    current_num_nodes = 1
    current_layer     = [adjacency_matrix(root_graph)]

    all_dag_adjs  = SparseMatrix[]
    append!(all_dag_adjs, current_layer)


    while current_num_nodes <= max_num_nodes
        next_num_nodes       = current_num_nodes += 1
        next_layer_binary_nums = all_binary_numbers_up_to(next_num_nodes -1)

        empty_next_layer_adj = adjacency_matrix(SimpleDiGraph(next_num_nodes))
        next_layer = SparseArrays.SparseMatrixCSC{Int64, Int64}[]
        for adj in current_layer
            base_adj_for_this_layer = embed_sparse_matrix(adj, empty_next_layer_adj)
            
            for num in next_layer_binary_nums
                new_col = binary_num_to_vector(num)
                new_adj = base_adj_for_this_layer[:,:]
                
                for i in 1:next_num_nodes-1
                    new_adj[i, end] = new_col[i]
                end
            
                push!(next_layer, new_adj)
            end

            append!(all_dag_adjs, next_layer)
            current_layer = next_layer
        end
    end
    
    return all_dag_adjs
end


##

@time all_dag_adjs = generate_all_dag_adjs(3)
print(length(all_dag_adjs))
# xs = collect(-1.0:0.1:1)

# fig = Figure()
# ax = Axis(fig[1, 1])

# for i in 1:length(all_dag_adjs)
#     adj  = all_dag_adjs[i]
#     charpoly = x -> sparse_matrix_characteristic_polynomial(adj, x)
#     ys = charpoly.(xs)
#     lines!(xs, ys)
# end

# fig
