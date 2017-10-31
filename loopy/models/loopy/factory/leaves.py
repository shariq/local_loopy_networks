import numpy as np
import math


# leaves are the elements of an ExpressionTree which cannot have children
# these functions almost all require some context to evaluate
# context includes whatever was passed to the update/initialize rule, number of edges, edge_memory_size, node_memory_size,...

# the context for the leaves is figured out at *render* time; and same with all other args for these leaves!

###############
# vector leaves requiring context
###############

def edge_memory(index, context):
    edge_memory_size = context['edge_memory_size']
    edge_memory = context['edge_memory']
    # edge_memory is derived from node_read_buffer; it's node_read_buffer[node_memory_size:]
    return edge_memory[index::edge_memory_size]

# by default all vectors are number of edges long
# getting vectors to be smaller is done through filters
def gaussian(context):
    slot_type = context['slot_type']
    if slot_type != 'vector':
        return np.random.normal()
    number_edges = context['edges']
    return np.random.normal(size=number_edges)

def uniform(context):
    slot_type = context['slot_type']
    if slot_type != 'vector':
        return np.random.uniform(-1, 1)
    number_edges = context['edges']
    return np.random.uniform(-1, 1, size=number_edges)

###############
# float leaves requiring context
###############

def node_memory(index, context):
    node_memory = context['node_memory']
    return node_memory[index]

###############
# float/vector leaves which don't take arguments/context
###############

one = 1.0
zero = 0.0
minus_one = -1.0
two = 2.0
minus_two = -2.0
point_one = 0.1
point_zero_one = 0.01
point_zero_zero_one = 0.001
