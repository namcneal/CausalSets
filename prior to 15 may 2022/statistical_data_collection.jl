using BSON: @save, @load
using CSV, DataFrames
using Distributions: Uniform
using Graphs

include("Libraries/CausalDiamonds2.jl")

# Useful reminder for function: categorical!(df, :A)
##

function init_data_containers()
    big_zero = convert(UInt128, 0)
    causet_level_data = DataFrame(scalar_curvature = Float64[0.],
                                  spacetime_type   = String["thisisafillerrow"],
                                  reduced          = Bool[false],
                                  cardinality      = Int64[-1],
                                  number_2paths    = UInt128[big_zero],
                                  number_3paths    = UInt128[big_zero],
                                  number_4paths    = UInt128[big_zero],
                                  exp_grand_sum    = UInt128[big_zero], 
                                )

    event_level_data = DataFrame(scalar_curvature = Float64[0.],
                                  spacetime_type   = String["thisisafillerrow"],
                                  reduced          = Bool[false],
                                  num_out_1_step   = Int64[-1],   # Children
                                  num_out_2_steps  = Int64[-1],   # Grandchildren
                                  num_out_3_steps  = Int64[-1],   # Great ...
                                  num_out_4_steps  = Int64[-1],   # ...
                                  num_in_1_step    = Int64[-1],   # Parents
                                  num_in_2_steps   = Int64[-1],   # Grandparents
                                  num_in_3_steps   = Int64[-1],   # Great-grandparents
                                  num_in_4_steps   = Int64[-1],   # ...
                                )

    return causet_level_data, event_level_data
end

##

function generate_full_causal_set(tmin::Float64, tmax::Float64, scalar_curvature::Float64, num_events::Int64, 
                                mcmc_num_to_sample::Int64, mcmc_start_collecting_after::Int64,
                                mcmc_jump_width::Float64, mcmc_start_time::Float64)

    region = CausalDiamond(tmin, tmax)
    sprinkling = sprinkle(region, num_events, scalar_curvature,
                          mcmc_num_to_sample, mcmc_start_collecting_after, 
                          mcmc_jump_width, mcmc_start_time)

    causet = make_causet(sprinkling)

    return causet
end
       
function determine_spacetime_type(scalar_curvature::Float64)
    if abs(scalar_curvature) < 1e-7
        return "Minkowski"
    elseif scalar_curvature > 0
        return "de Sitter"
    else 
        return "anti-de Sitter"
    end
end

function get_descendants(causet::SimpleDiGraph, event::Int64, depth::Int64)
    
    current_layer  = Int64[event]
    all_decendants = Int64[]
    for i in 1:depth
        next_layer = Int64[]
        for e in current_layer 
            append!(next_layer, outgoing(causet, e))
        end

        append!(all_decendants, next_layer)
        current_layer = deepcopy(next_layer)
    end

    return all_decendants
end

##

function process_causal_set(causet::SimpleDiGraph, scalar_curvature::Float64, reduced::Bool)

    # Collect the large, causet-level data
    spacetime_type = determine_spacetime_type(scalar_curvature)
    cardinality    = nv(causet)
    causet_level = [scalar_curvature, spacetime_type, reduced, cardinality]

    # Collect event-level details
    event_level  = Vector{Any}[]

    # Gather the causal structure in adjacency matrix form
    adj_powers = get_adjacency_powers(causet)

    # Extract the data for each adjancency power
    num_powers = size(adj_powers)[end]
    
    # Causet level power-related data
    number_2paths = sum(adj_powers[:,:,2])
    number_3paths = sum(adj_powers[:,:,3])
    number_4paths = sum(adj_powers[:,:,4])
    exponential   = sum([adj_powers[:,:,k] for k in 1:num_powers])
    exp_grand_sum = sum(exponential)
    append!(causet_level, [number_2paths, number_3paths, number_4paths, exp_grand_sum])

    # Now loop to get the event-level details
    for e in 1:nv(causet)
        event_data = [scalar_curvature, spacetime_type, reduced]

        for depth_out in 1:4
            push!(event_data, sum(adj_powers[e, :, depth_out]))
        end
        for depth_in  in 1:4
            push!(event_data, sum(adj_powers[:, e, depth_in ]))
        end


        push!(event_level, event_data)
    end

    return causet_level, event_level
end

##

# Empty containers for the datasets 
causet_level_data_path = "Data/causet_level_data_collected_21feb2022.bson"
 event_level_data_path =  "Data/event_level_data_collected_21feb2022.bson"

causet_level_data, event_level_data = init_data_containers()
# CSV.write(causet_level_data_path, causet_level_data)
# CSV.write(event_level_data_path,  event_level_data)

@save causet_level_data_path causet_level_data
@save  event_level_data_path  event_level_data

##

# Causet specifications
region = CausalDiamond(0, 1)
tmin = 0. 
tmax = 1.

# MCMC
start_collecting_after = 5000
num_mcmc_samples = start_collecting_after * 10
jump_width = 5e-2
start_time = 0.5 # Start in the middle of the causal diamond
mcmc_parameters = (num_mcmc_samples, start_collecting_after, jump_width, start_time)

# The data generating and saving loop
for R in [0., 1., -1.]
    for num_events in [250, 500, 1000, 2000, 4000]
        # Read the current DataFrame from file
        @load causet_level_data_path causet_level_data
        @load  event_level_data_path  event_level_data

        # Generate the causet
        @time _, causet = generate_full_causal_set(tmin, tmax, R, num_events, mcmc_parameters...)

        # Process and save the data for the raw causal set
        reduced = false
        causet_level, event_level =  process_causal_set(causet, R, reduced)

        push!(causet_level_data, causet_level)
        for event in 1:length(event_level)
            push!(event_level_data, event_level[event])
        end

        causet  = transitivereduction(causet)
        reduced = true
        causet_level, event_level =  process_causal_set(causet, R, reduced)

        push!(causet_level_data, causet_level)
        for event in 1:length(event_level)
            push!(event_level_data, event_level[event])
        end

        # The causet level data is finished, so save it and move onto the next one
        @save causet_level_data_path causet_level_data
        @save  event_level_data_path  event_level_data
    end
end