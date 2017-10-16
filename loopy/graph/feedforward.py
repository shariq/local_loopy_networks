from collections import defaultdict

def make_feedforward_adjacency_dict(input_size=2, hidden_size=8, output_size=1, secondary_output_layer=False, layers=1):
    if layers != 1:
        raise Exception("make_feedforward_adjacency_dict does not support layers != 1 yet")
    input_nodes = list(range(input_size))
    hidden_nodes = list(range(input_size, input_size+hidden_size))
    output_nodes = list(range(input_size+hidden_size, input_size+hidden_size+output_size))
    adjacency_dict = defaultdict(lambda: set())
    for input_node in input_nodes:
        for hidden_node in hidden_nodes:
            adjacency_dict[input_node].add(hidden_node)
    for hidden_node in hidden_nodes:
        for output_node in output_nodes:
            adjacency_dict[hidden_node].add(output_node)
    if secondary_output_layer:
        # this lets us easily assign errors to edges instead of nodes
        # bunch of edges near the end which we can just assign errors as if secondary_output_layer sent those errors back
        non_secondary_output_size = input_size + hidden_size + output_size
        secondary_output_nodes = list(range(non_secondary_output_size, non_secondary_output_size + output_size))
        for output_node, secondary_output_node in zip(output_nodes, secondary_output_nodes):
            adjacency_dict[output_node].add(secondary_output_node)
    return dict(adjacency_dict)
