using Einsum 

##
function tetrad2metric(tetrad::Matrix{Float64})
    # Flat spacetime metric
    η = Matrix{Float64}(I, size(tetrad))
    η[1,1] *= -1

    return tetrad' * η * tetrad
end

##
"""
    Runge-Kutta 4
"""
function get_k1(current_y::Vector{Float64}, f_derivative::Function)
    return f_derivative(current_y)
end

function get_k2(current_y::Vector{Float64}, f_derivative::Function;
                k1::Vector{Float64}, step::Float64)
    return f_derivative(current_y + k1*step/2)
end

function get_k3(current_y::Vector{Float64}, f_derivative::Function;
                k2::Vector{Float64}, step::Float64)

    return f_derivative(current_y + k2*step/2)
end

function get_k4(current_y::Vector{Float64}, f_derivative::Function;
                k3::Vector{Float64}, step::Float64)

    return f_derivative(current_y + k3*step)
end

function get_ks(current_y::Vector{Float64}, f_derivative::Function; step::Float64)
    k1 = get_k1(current_y, f_derivative)
    k2 = get_k2(current_y, f_derivative; k1=k1, step=step)
    k3 = get_k3(current_y, f_derivative; k2=k2, step=step)
    k4 = get_k4(current_y, f_derivative; k3=k3, step=step)

    return k1, k2, k3, k4
end

function rk4_step(current_y::Vector{Float64}, f_derivative::Function; step::Float64)
    k1, k2, k3, k4 = get_ks(current_y, f_derivative; step=step)

    next_y = current_y + step * (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    return next_y
end

"""
    Solving the geodesic equation
"""
##

function geodesic_acceleration(metric::Function, position::Vector{Float64}, velocity::Vector{Float64})
    Γ = christoffel(metric, position)
    
    @einsum acceleration[k] := Γ[k,i,j] * velocity[i] * velocity[j]
end

function geodesic_step(metric::Function, position::Vector{Float64}, velocity::Vector{Float64}; step::Float64)
    next_position = rk4_step(position, v->velocity; step=step)
    next_velocity = rk4_step(velocity, v->geodesic_acceleration(metric, position, v);step=step)
    
    return next_position, next_velocity
end

function geodesic_path(initial_position::Vector{Float64}, initial_velocity::Vector{Float64}, num_steps::Int64; step::Float64)
    positions  = [initial_position]
    velocities = [initial_velocity]

    for _ in 1:num_steps
        next_position, next_velocity = geodesic_step(metric, positions[end], velocities[end]; step=step)
        push!(positions,  next_position)
        push!(velocities, next_velocity)
    end

    return positions, velocities
end

function is_timelike_future_oriented(tetrad::Function, point::Vector{Float64}, direction::Vector{Float64}; tol::Float64=1e-2)
    # Filter spacelike
    if direction' * (tetrad2metric∘tetrad)(point) * direction > tol
        return false
    else
        return tetrad(point)[1,:]' * direction > 0
    end
end

function shoot_geodesic(tetrad::Function, source::Vector{Float64}, target::Vector{Float64}; 
                        step::Float64, success_tol::Float64=1e-10, 
                        boundary_radius_scaling::Float64=2.0)

    path = [source]

    # We assume the target point is sufficiently close to the source. 
    # Then, we assume as well that Euclidean path from the source to 
    # the target is a good approximation of the geodesic path. Hence, 
    # we assume that this path must start from a future-oriented timelike
    # vector at the source. 
    diff = target .- source
    distance = norm(diff)

    if is_timelike_future_oriented(tetrad, source, diff/distance)
        # Shoot a geodesic toward the target until it leaves a circle surrounding the 
        # two points, centered at the midpoint with a diameter α-times the distance between
        midpoint = source .+ 0.5 * diff/2.0

        position = source
        velocity = diff / distance
        while norm(path[end] .- midpoint) <= boundary_radius_scaling*distance
            position, veloctity = geodesic_step(metric, position, velocity; step=step)
            push!(path, position)

            # End the geodesic if the target point is actually reached
            if norm(path[end] .- target) <= success_tol
                break
            end
        end
    end

    return path
end

function closest_approach(path::Vector{Vector{Float64}}, target::Vector{Float64})
    distances = [norm(target .- point) for point in path]

    return minimum(distances)
end

##
function iteration_x2y(metric::Function, x::Vector{Float64}, x_before::Vector{Float64}, x_after::Vector{Float64})
    midpoint = (1/2) * (x_before .- x_after)
    diff = x_after .- x_before
    Γ    = christoffel(metric, x)

    return midpoint + (1/4) * [diff' * Γ[k,:,:] * diff for k in 1:size(Γ)[1]]
end

function iteration_step(metric::Function, current_path::Vector{Vector{Float64}})
    new_path = current_path[1:1]

    for k in 2:length(current_path)-1
        push!(new_path, iteration_x2y(metric, current_path[k-1:k+1]...))
    end

    push!(new_path, current_path[end])

    return new_path
end


##

function path2xy(path::Vector{Vector{Float64}})
    return [point[1] for point in path], [point[2] for point in path]
end