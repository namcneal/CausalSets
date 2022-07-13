using LinearAlgebra
using Random

# Graph theory
using LightGraphs

## 

function causet2arrows(causet::SimpleDiGraph, sprinkling::Matrix{Float64}, align::Symbol=:center)
    arrow_data = Vector{Float64}[]

    for edge in edges(causet)
        x,y = sprinkling[:, src(edge)]

        u,v = sprinkling[:, dst(edge)] - sprinkling[:, src(edge)] 

        # Position the head at the midpoint
        if align == :center
            x = (u .+ 2*x) / 2
            y = (v .+ 2*y) / 2
        end

        push!(arrow_data, [x; y; u; v])
    end

    return [reduce(hcat, arrow_data)[i, :] for i in 1:4]
end

function plot_events(sprinkling::Matrix{Float64};
                     colors::Vector{Symbol}=repeat([:black], size(sprinkling)[2]), backgroundcolor=:white, ms::Float64=10.0)

    sprinkling = sprinkling[[2,1], :]

    f  = Figure(resolution = (800, 800), backgroundcolor = backgroundcolor)
    ax=  Axis(f[1, 1], backgroundcolor = backgroundcolor)

    x = sprinkling[1, :]
    y = sprinkling[2,:]
    scatter!(x,y, align=:center, color=:black, markersize=ms)
    return ax, f
end


function hasse_diagram(sprinkling::Matrix{Float64},causet::SimpleDiGraph,
                       colors::Vector{Symbol}=repeat([:black], ne(causet)), backgroundcolor=:white)

    sprinkling = sprinkling[[2,1], :]

    f  = Figure(resolution = (800, 800), backgroundcolor = backgroundcolor)
    ax=  Axis(f[1, 1], backgroundcolor = backgroundcolor)

    for edge in edges(causet)
        positions = sprinkling[:,[src(edge), dst(edge)]]
        scatterlines!(positions, markersize=25, linewidth=5, color=:black)
    end

    arrow_data = causet2arrows(causet, sprinkling)
    
    arrows!(arrow_data..., lengthscale=0.01, width=10, align=:center, arrowsize=30, color=:black)
    return ax, f
end

nothing;







