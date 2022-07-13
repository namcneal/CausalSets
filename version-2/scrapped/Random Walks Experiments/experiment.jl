include("lib.jl")
include("../lib/Causets.jl")

import Colors
import ColorSchemes

##

MCMC_WIDTH          = 0.4

initial_coordinates = [0.0; 0.0]
sprinkler           = MCMC(MCMC_WIDTH, initial_coordinates)

center = [0.0; 0.0]
radius = 1.0
norm_p = Inf
ball   = SpacetimeBall(center, radius, norm_p)

##
num_events        = 20
total_spacetime = metropolis(x->χ_ball(x, ball) , x->1, 
                             num_events     , sprinkler.width,
                             sprinkler.last_accepted_coordinates)


sprinkler.last_accepted_coordinates = total_spacetime[:, end]

small_ball_center = center
small_radius      = 0.5
small_norm        = 1
small_ball        = SpacetimeBall(small_ball_center, small_radius, small_norm)

in_small_ball              = χ_ball(total_spacetime, small_ball)
events_in_small_ball       = total_spacetime[:, in_small_ball]
events_in_small_ball[:, 1] = small_ball_center

print(sum(in_small_ball))

##
metric(x::Vector{Float64}) = x' * [-1.0 0; 0 1] * x

causet = make_irreducible_causet(events_in_small_ball, metric)

hasse_diagram(events_in_small_ball, causet)
current_figure()

##

const INITIAL_VERTEX  = 1
const TOTAL_NUM_STEPS = 100
const NUM_WALKS       = 100

random_walk_trial_params = RandomWalkParameters(INITIAL_VERTEX, TOTAL_NUM_STEPS, 
                                                causet, :forward)

vertex_frequencies     = repeated_random_walks(NUM_WALKS, random_walk_trial_params)
maximum_frequency      = maximum(values(vertex_frequencies))
      
black_RGB = Colors.RGB(1.0, 1.0, 1.0)
colors = repeat([black_RGB], num_events)
for key in keys(vertex_frequencies)
    colors[key] = get(ColorSchemes.amp, vertex_frequencies[key] / maximum_frequency)
end

scatter(total_spacetime, color=colors, aspect_ratio=1.0)
current_figure()


## 

causet 