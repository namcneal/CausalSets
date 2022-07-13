using LightGraphs

##
struct Messenger
    id::Int64
    nodes_visited::Vector{Int64}
    other_messengers_seen::Vector{Int64}
end

function Messenger()
    return Messenger(-1, [], [])
end

function Messenger(id::Int64)
    return Messenger(id, [], [])
end

##
    
function send_messengers(graph::SimpleDiGraph, source::Int64, 
                         inboxes::Vector{Vector{Messenger}}, outboxes::Vector{Vector{Messenger}})

    for messenger in outboxes[source]
        for node in outneighbors(graph, source)
            push!(messenger.nodes_visited, node)
            push!(inboxes[node], deepcopy(messenger))
        end
    end

    outboxes[source] = Messenger[];
end

function inbox_to_outbox(inboxes::Vector{Vector{Messenger}}, outboxes::Vector{Vector{Messenger}})
    for i in 1:length(inboxes)
        outboxes[i] = inboxes[i][:]
        inboxes[i] = Messenger[]
    end
end

function run_messenger_iteration(graph::SimpleDiGraph, 
                                inboxes::Vector{Vector{Messenger}}, outboxes::Vector{Vector{Messenger}})

    num_nodes = nv(graph)
    for i_send in 1:num_nodes
        send_messengers(graph, i_send, inboxes, outboxes)
    end

    inbox_to_outbox(inboxes, outboxes)
    
    return inboxes, outboxes
end
##
num_nodes = 3
g = SimpleDiGraph(num_nodes)

outboxes = [Messenger[] for i in 1:num_nodes]
inboxes  = [Messenger[] for i in 1:num_nodes]

##

add_edge!(g, 1, 2)
add_edge!(g, 1, 3)
 
push!(outboxes[1], Messenger(1))
inboxes, outboxes = run_messenger_iteration(g, inboxes, outboxes)

##
# inboxes, outboxes = run_messenger_iteration(g, inboxes, outboxes)

outboxes[3]