def make_feedforward_adjacency_dict(input_size=2, hidden_size=8, output_size=1, layers=1):
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
    return output_node
