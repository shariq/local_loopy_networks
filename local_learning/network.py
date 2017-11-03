import random
from collections import defaultdict
import numpy as np
import math

import local_learning
import logging
logger = logging.getLogger()

from local_learning.graph.tools import undirect_adjacency_dict

class Network:
    def __init__(self, in_adjacency_dict, node_memory_size=1, edge_memory_size=1, step_rule=None):
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

        # NOTE: this class can be reimplemented as using a single buffer
        # biggest changes are update/initialize rules get applied in place, and when resolving conflicting writes we do something smart about only updating edges which haven't been updated before, e.g, only update edges with neighbor number > current node number

        self.read_buffer = [np.zeros(buffer_size(node)) for node in self.nodes]
        self.write_buffer = [np.zeros(buffer_size(node)) for node in self.nodes]

        # buffer LAYOUT:
        # [node_memory, edge_memory]

        # edge_memory LAYOUT:
        # for adjacent node in smallest to biggest;
        # [edge_memory_node, edge_memory_node, ...]

        # when we call it a read buffer or a write buffer, that refers to
        # whether the step rule is reading/writing to it. code here
        # still has to move data around between these buffers

        self.step_rule = step_rule


    def initialize(self, initialize_rule):

        write_buffer = self.write_buffer

        for node in self.nodes:
            edges = self.adjacency_size[node]
            write_buffer[node][:] = initialize_rule(node_memory_size=self.node_memory_size, edge_memory_size=self.edge_memory_size, edges=edges)

        self._resolve_write_to_read()


    def _resolve_write_to_read(self):
        # in this method, we are moving data from the write buffer to the read buffer
        # which is the opposite of what is done in the step rule: where write = f(read)
        # because the edges can have conflicting values, we resolve conflicts by adding values together
        # it maybe makes more sense to multiply the values together instead

        read_buffer = self.read_buffer
        write_buffer = self.write_buffer

        for node in self.nodes:
            read_buffer[node][:self.node_memory_size] = write_buffer[node][:self.node_memory_size]
            # first let's set this node's own memory to the right thing
            read_buffer[node][self.node_memory_size:] = write_buffer[node][self.node_memory_size:] / 2.0
            # and initialize the edge memory to half of previous value (conflicts resolved by taking average)

        for node in self.nodes:
            for i, adjacent_node in enumerate(self.sorted_adjacency_dict[node]):
                read_start = self.node_memory_size + i * self.edge_memory_size
                write_start = self.node_memory_size + self.sorted_adjacency_dict[adjacent_node].index(node) * self.edge_memory_size
                read_buffer[node][read_start:read_start + self.edge_memory_size] += write_buffer[adjacent_node][write_start:write_start + self.edge_memory_size] / 2.0
                # Divide 2.0 since write conflicts are resolved by taking the average, and every edge has two writers


    def step(self, step_rule=None):
        if step_rule is None:
            if self.step_rule is None:
                raise Exception("you need to pass in a step_rule now or in the past!")
            else:
                step_rule = self.step_rule
        else:
            self.step_rule = step_rule

        read_buffer = self.read_buffer
        write_buffer = self.write_buffer

        for node in self.nodes:
            edges = self.adjacency_size[node]
            # IMPORTANT: node_write_buffer is a reference to be written to by the step_rule!
            step_rule(node_read_buffer=read_buffer[node], node_write_buffer=write_buffer[node], node_memory_size=self.node_memory_size, edge_memory_size=self.edge_memory_size, edges=edges)

        self._resolve_write_to_read()


    def set_edge_memory(self, edge, edge_memory):
        node_a, node_b = edge

        node_a_start_index = self.node_memory_size + self.sorted_adjacency_dict[node_a].index(node_b) * self.edge_memory_size
        node_a_end_index = node_a_start_index + self.edge_memory_size
        self.read_buffer[node_a][node_a_start_index:node_a_end_index] = edge_memory

        node_b_start_index = self.node_memory_size + self.sorted_adjacency_dict[node_b].index(node_a) * self.edge_memory_size
        node_b_end_index = node_b_start_index + self.edge_memory_size
        self.read_buffer[node_b][node_b_start_index:node_b_end_index] = edge_memory


    def get_edge_memory(self, edge):
        node_a, node_b = edge
        node_a_start_index = self.node_memory_size + self.sorted_adjacency_dict[node_a].index(node_b) * self.edge_memory_size
        node_a_end_index = node_a_start_index + self.edge_memory_size
        return self.read_buffer[node_a][node_a_start_index:node_a_end_index]


    def set_node_memory(self, node, node_memory):
        self.read_buffer[node][:self.node_memory_size] = node_memory[:]


    def get_node_memory(self, node):
        return self.read_buffer[node][:self.node_memory_size]


    def debug_log_buffer(self, buffer):
        print_options = np.get_printoptions()
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        for node, node_buffer in enumerate(buffer):
            logger.debug('{}={}'.format(node, node_buffer))
        np.set_printoptions(**print_options)


    def debug_log_buffers(self, message):
        logger.debug(message)
        logger.debug('read_buffer=>')
        self.debug_log_buffer(self.read_buffer)
        logger.debug('write_buffer=>')
        self.debug_log_buffer(self.write_buffer)
        logger.debug('\n')
