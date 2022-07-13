"""
    DiamondBounds

Holds the bounds of a diamond-shaped region of spacetime. 
The top  and bottom corners are at x = 0 and t = tmin, tmax respectively. 
The left and right  corners are at t = 0 and x = xmin, xmax respectively.
"""
struct DiamondBounds
    min_time::Float64
    max_time::Float64
    
    min_space::Float64
    max_space::Float64

    num_spacial_dimensions::Int
end

"""
The parameters for a patch of spacetime
"""
function DiamondBounds(min_time::Real, max_time::Real, 
                       num_spacial_dimensions::Int64)

    Δtime = max_time - min_time
    
    min_space = -Δtime / 2
    max_space =  Δtime / 2
    
    return DiamondBounds(min_time, max_time, 
                         min_space, max_space,
                         num_spacial_dimensions)
end

"""
    in_diamond(diamond, event)

Determines whether an event at the given `event coordinates` is contained within the
causal diamond whose bounds are held in `diamond`. 
"""
function is_in_diamond(diamond::DiamondBounds, event_coordinates::Vector{Float64})
    # The time of the event
    t = event_coordinates[1]

    # The length of the spacial component of the vector
    x = norm(event_coordinates[2:end])
    
    # Not in the diamond because it's out of bounds
    if t < diamond.min_time || t > diamond.max_time || x < diamond.min_space || x > diamond.max_space
        return false
    end

    # Within the lightcones of the top and bottom elements
    return (t-diamond.min_time)^2 > x^2 && (t-diamond.max_time)^2 > x^2
end









