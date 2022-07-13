using ComplexPortraits
using ImageShow
using LightGraphs
using LinearAlgebra

using CairoMakie

import SparseArrays
@ComplexPortraits.import_normal



function characteristic_polynomial(t::ComplexF64, matrix::SparseArrays.SparseMatrixCSC)
    return det(t*I - matrix)
end

##
graph = SimpleDiGraph(4)
add_edge!(graph, 1,2)
add_edge!(graph, 2,3)
add_edge!(graph, 3,1)


A = adjacency_matrix(graph)
f(z) = characteristic_polynomial(z, A)

xs = collect(-1:0.1:1)
ys = f.(xs)

lines(xs, ys)






# display(A)

# upper_left  = 1im - 1
# lower_right = 1 - 1im

# scale_bounds = 0.01
# bounds       = scale_bounds .* [upper_left, lower_right]
# num_pixels  = 1200

# portrait(bounds..., f; no_pixels=num_pixels, point_color=cs_j())