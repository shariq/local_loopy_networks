def make_loop_adjacency_dict(size=10):
    adjacency_dict = {}
    for i in range(size):
        adjacency_dict[i] = set([(i+1) % size])
    return adjacency_dict
