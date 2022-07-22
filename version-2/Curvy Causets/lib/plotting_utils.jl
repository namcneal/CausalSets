using CairoMakie
using Colors
using ColorSchemes
using GeometryBasics
using VoronoiCells


function plot_field(events::Vector{Vector{Float64}}, field_values::Vector{Base.Float64}, 
    limits::Vector{Float64}; color=colorant"rgb(25,50,118)")

    xmin, xmax, ymin, ymax = limits

    rectangle   = Rectangle(Point2(xmin, ymin), Point2(xmax, ymax))
    tessellation = voronoicells(Point2.(events), rectangle)

    field = field_values
    num_cells   = length(field)
    colorscheme = ColorScheme(range(colorant"white", color))
    colors      = field .- minimum(field)
    colors /= maximum(colors)

    f = Figure()
    ax = Axis(f[1, 1])
    for (i, cell) in enumerate(tessellation.Cells)
        _color = get(colorscheme, colors[i])
        poly!(cell, color=_color, strokecolor=_color, strokewidth=3)
    end

    return f, ax
end
