using CairoMakie
using Distributions
using DelimitedFiles
using LightGraphs
using Statistics

import FiniteDiff
include("../lib/normal_coordinate_utils.jl")

##
const num_spacial_dimensions = 1
dim = num_spacial_dimensions + 1

riemann_at_center   = random_riemann(dim, 30.0)
metric_perturbation(x) = normal_metric_perturbation(riemann_at_center, x) 
metric(x::Vector{Float64})  = flat_metric(dim) + metric_perturbation(x)


sidelength = 10.0
ds = 1e-2
ncube_uniform = [[x; y] for x in -sidelength:ds:sidelength, y in -sidelength:ds:sidelength] |> vec
norms = norm.(ncube_uniform)
ncube_uniform = ncube_uniform[sortperm(norms)]

metric_singular_here = Float64[]
for point in ncube_uniform 
    g = det(metric(point))

    # indicates the metric has switched signs
    if g > 0
        push!(metric_singular_here, 1.0)
    else
        push!(metric_singular_here, 0.0)
    end
end

##
f = Figure()
ax = Axis(f[1,1])
ys = cumsum(metric_singular_here)
lines!(norm.(ncube_uniform)[1:end-1], ys[2:end].- ys[1:end-1])
f

## 
guaranteed_boundary = 