import random
from collections import defaultdict
import numpy as np
import math

import loopy
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def reverse_adjacency_dict(adjacency_dict):
    reversed_adjacency_dict = defaultdict(lambda: set())
    for key, value in adjacency_dict.items():
        for element in value:
            reversed_adjacency_dict[element].add(key)
            if element == key:
                raise Exception("networks with self connected nodes unsupported")
    return dict(reversed_adjacency_dict)


def undirect_adjacency_dict(adjacency_dict):
    undirected_adjacency_dict = defaultdict(lambda: set())
    for key, value in adjacency_dict.items():
        for element in value:
            undirected_adjacency_dict[element].add(key)
            undirected_adjacency_dict[key].add(element)
            if element == key:
                raise Exception("networks with self connected nodes unsupported")
    return dict(undirected_adjacency_dict)


class Network:
    def __init__(self, in_adjacency_dict, node_memory_size=1, edge_memory_size=2):
        # in_adjacency_dict nodes have no self loops and not more than one edge per pair of nodes
        # and make sure to include an empty set if that node has no edges
        self.node_memory_size = node_memory_size
        self.edge_memory_size = edge_memory_size

        self.in_adjacency_dict = in_adjacency_dict
        self.out_adjacency_dict = reverse_adjacency_dict(in_adjacency_dict)
        self.undirected_adjacency_dict = undirect_adjacency_dict(in_adjacency_dict)

        self.in_adjacency_size = dict([(key, len(value)) for key, value in in_adjacency_dict.items()])
        self.out_adjacency_size = dict([key, len(value) for key, value in out_adjacency_dict.items()])
        self.undirected_adjacency_size = dict([key, len(value) for key, value in undirected_adjacency_dict.items()])

        self.nodes = sorted(list(self.undirected_adjacency_dict.keys()))

        '''
        # in_adjacency_dict is a dict like this
        {
            node_1: set([
                adjacent_1,
                adjacent_2,
                ...
            ]),
            node_2: ...
        ...
        }

        # out_adjacency_dict is the same, but in the reverse direction
        # since the network is directed

        # undirected_adjacency_dict contains neighbors for all nodes (in + out)
        '''

        def read_buffer_size(node):
            in_edges = self.in_adjacency_size[node]
            out_edges = self.out_adjacency_size[node]
            total_edges = in_edges + out_edges

            local_node_memory_size = node_memory_size * (total_edges)
            local_edge_memory_size = edge_memory_size * (in_edges)
            return local_edge_memory_size + local_node_memory_size

        def write_buffer_size(node):
            in_edges = self.in_adjacency_size[node]
            out_edges = self.out_adjacency_size[node]
            total_edges = in_edges + out_edges

            local_node_memory_size = node_memory_size * 1
            local_edge_memory_size = edge_memory_size * (out_edges)
            return local_edge_memory_size + local_node_memory_size

        self.read_buffers = [
            [np.zeros(read_buffer_size(node) for node in self.nodes],
            [np.zeros(read_buffer_size(node) for node in self.nodes],
        ]

        self.write_buffers = [
            [np.zeros(write_buffer_size(node) for node in self.nodes],
            [np.zeros(write_buffer_size(node) for node in self.nodes],
        ]
        # like a 2 element ring buffer

        self.writing_to_first_buffer = True
        # True / False to switch between buffers

    def initialize(self, initialize_rule):
        for node in self.nodes:
            out_edges = self.out_adjacency_size[node]
            in_edges = self.in_adjacency_size[node]
            self.local_buffers[0][node][:] = initialize_rule(in_edges=in_edges, out_edges=out_edges)
            self.local_buffers[1][node][:] = self.local_buffers[0][node][:]

    def update(self, update_rule):
        
        for node in self.nodes:
