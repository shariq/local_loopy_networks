from collections import defaultdict

import logging
logger = logging.getLogger()

def undirect_adjacency_dict(adjacency_dict):
    undirected_adjacency_dict = defaultdict(lambda: set())
    for key, value in adjacency_dict.items():
        for element in value:
            undirected_adjacency_dict[element].add(key)
            undirected_adjacency_dict[key].add(element)
            if element == key:
                logger.debug(adjacency_dict)
                raise Exception("networks with self connected nodes unsupported")
    for key, value in adjacency_dict.items():
        adjacency_dict[key] = sorted(list(value))
        # make it easier to lay out memory correctly
    return dict(undirected_adjacency_dict)
