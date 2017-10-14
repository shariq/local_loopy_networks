import random
from collections import defaultdict
import numpy as np
import math

import loopy
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def undirect_adjacency_dict(adjacency_dict):
    undirected_adjacency_dict = defaultdict(lambda: set())
    for key, value in adjacency_dict.items():
        for element in value:
            undirected_adjacency_dict[element].add(key)
            undirected_adjacency_dict[key].add(element)
            if element == key:
                raise Exception("networks with self connected nodes unsupported")
    for key, value in adjacency_dict.items():
        adjacency_dict[key] = sorted(list(value))
        # make it easier to lay out memory correctly
    return dict(undirected_adjacency_dict)


class Network:
    def __init__(self, in_adjacency_dict, node_memory_size=1, edge_memory_size=1):
        # in_adjacency_dict may or may not be directed; we'll fix it right up to be undirected :]
        self.node_memory_size = node_memory_size
        self.edge_memory_size = edge_memory_size

        # node => set(neighbors)
        self.adjacency_dict = undirect_adjacency_dict(in_adjacency_dict)

        # node => sorted_list(neighbors)
        self.sorted_adjacency_dict = dict([(key, sorted(list(value))) for key, value in self.adjacency_dict.items()])

        # meaning a mapping from node to number of edges
        self.adjacency_size = dict([(key, len(value)) for key, value in self.adjacency_dict.items()])

        self.nodes = sorted(list(self.adjacency_dict.keys()))

        assert set(self.nodes) == set(range(len(self.nodes)))
        # for now nodes are just ints which go from 0 to len(graph) - 1

        buffer_size = lambda node: node_memory_size + edge_memory_size * self.adjacency_size[node]

        self.read_buffer = [np.zeros(buffer_size(node)) for node in self.nodes]
        self.write_buffer = [np.zeros(buffer_size(node)) for node in self.nodes]

        # buffer LAYOUT:
        # [node_memory, edge_memory]

        # edge_memory LAYOUT:
        # for adjacent node in smallest to biggest;
        # [edge_memory_node, edge_memory_node, ...]

        # when we call it a read buffer or a write buffer, that refers to
        # whether the update rule is reading/writing to it. code here
        # still has to move data around between these buffers


    def initialize(self, initialize_rule):

        read_buffer = self.read_buffer

        for node in self.nodes:
            edges = self.adjacency_size[node]
            read_buffer[node][:] = initialize_rule(node_memory_size=self.node_memory_size, edge_memory_size=self.edge_memory_size, edges=edges)


    def _resolve_write_to_read(self):
        # in this method, we are moving data from the write buffer to the read buffer
        # which is the opposite of what is done in the update rule: where write = f(read)
        # because the edges can have conflicting values, we resolve conflicts by adding values together
        # it maybe makes more sense to multiply the values together instead

        read_buffer = self.read_buffer
        write_buffer = self.write_buffer

        for node in self.nodes:
            read_buffer[node][:] = write_buffer[node]
            # first let's set this node's own memory to the right thing
            # and initialize the edge memory to something reasonable

        for node in self.nodes:
            for i, adjacent_node in enumerate(self.sorted_adjacency_dict[node]):
                read_start = self.node_memory_size + i * self.edge_memory_size
                write_start = self.node_memory_size + self.sorted_adjacency_dict[adjacent_node].index(node) * self.edge_memory_size
                read_buffer[node][read_start:read_start + self.edge_memory_size] += write_buffer[adjacent_node][write_start:write_start + self.edge_memory_size]


    def update(self, update_rule):
        read_buffer = self.read_buffer
        write_buffer = self.write_buffer

        for node in self.nodes:
            edges = self.adjacency_size[node]
            update_rule(node_read_buffer=read_buffer[node], node_write_buffer=write_buffer[node], node_memory_size=self.node_memory_size, edge_memory_size=self.edge_memory_size, edges=edges)

        self._resolve_write_to_read()
