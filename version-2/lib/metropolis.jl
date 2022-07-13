using Distributions
using Random

##
"""
https://storopoli.io/Bayesian-Julia/pages/5_MCMC/

    width: Each coordinate direction can jump within this distance, the next point in th MC will be in this square
"""
function metropolis(χ_region::Function  , density::Function  ,
                    num_samples::Int64  , width::Float64     ,
                    initial_coordinates::Vector{Float64};
                    seed::Int64=123, using_log_density=false)

    rng      = MersenneTwister(seed)
    dim      = length(initial_coordinates)
    draws    = Matrix{Float64}(undef, dim, num_samples)

    current_coordinates   = initial_coordinates[:]
    @inbounds draws[:, 1] = initial_coordinates[:]

    accepted = 1
    while accepted < num_samples
        dim = length(initial_coordinates)

        proposal_coordinates = current_coordinates .+ rand(rng, Uniform(-width, width), dim)

        # We want the Metropolis acceptance ratio to be zero if the density at
        # our current location is zero
        if !χ_region(proposal_coordinates)
            r = 1e-16
        else
            current_density  = density( current_coordinates)
            proposal_density = density(proposal_coordinates)

            if using_log_density
                r = exp(proposal_density - current_density)
            else
                r = exp(log(proposal_density) - log(current_density))
            end

            if r > rand(rng, Uniform())
                current_coordinates .= proposal_coordinates
                @inbounds draws[:, accepted] .= current_coordinates

                accepted += 1
            end

        end
    end

    # println("Acceptance rate is: $(accepted / S)")
    return draws
end







