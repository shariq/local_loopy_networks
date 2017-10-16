import unittest
import numpy as np
import random
import argparse
import time

from collections import defaultdict

import logging
logger = logging.getLogger()

from loopy.network import Network
# Network(in_adjacency_dict, node_memory_size=1, edge_memory_size=2)
from loopy.graph.waxman import make_waxman_adjacency_dict
from loopy.graph.loop import make_loop_adjacency_dict


def basic_initialize_rule(node_memory_size, edge_memory_size, edges):
    return np.zeros(node_memory_size + edge_memory_size * edges)


def basic_step_rule(node_read_buffer, node_write_buffer, node_memory_size, edge_memory_size, edges):
    node_write_buffer[:] = node_read_buffer[:]
    node_write_buffer[node_memory_size:] = 0

    if node_read_buffer[0] > 0.1:
        node_write_buffer[0] /= 2.0
        return

    edge_read_memory = node_read_buffer[node_memory_size:]
    edge_write_memory = node_write_buffer[node_memory_size:]

    output = np.zeros(edge_memory_size)
    for neighbor in range(edges):
        output += edge_read_memory[neighbor * edge_memory_size: (neighbor + 1) * edge_memory_size]

    for neighbor in range(edges):
        neighbor_edge_read_memory = edge_read_memory[neighbor * edge_memory_size: (neighbor + 1) * edge_memory_size]
        neighbor_edge_write_memory = edge_write_memory[neighbor * edge_memory_size: (neighbor + 1) * edge_memory_size]
        neighbor_edge_write_memory[:] = output * (abs(neighbor_edge_read_memory) <= 0.01)

    if any(node_write_buffer[node_memory_size:]):
        node_write_buffer[0] = 1.0


class TestNetwork(unittest.TestCase):
    def test_init(self):
        Network(in_adjacency_dict=make_loop_adjacency_dict())

    def test_initialize(self):
        network = Network(in_adjacency_dict=make_loop_adjacency_dict(), node_memory_size=1, edge_memory_size=1)
        network.initialize(initialize_rule=basic_initialize_rule)

    def test_step(self):
        network = Network(in_adjacency_dict=make_loop_adjacency_dict(), node_memory_size=1, edge_memory_size=1, step_rule=basic_step_rule)
        network.initialize(initialize_rule=basic_initialize_rule)
        logger.debug('\n=> test_step <=')
        network.debug_log_buffers('initialized...')
        network.write_buffer[0] = np.array([1.0, 1.0, -1.0])
        network.debug_log_buffers('manual overwrite partial')
        network._resolve_write_to_read()
        network.debug_log_buffers('resolve_write_to_read')
        network.step()
        network.debug_log_buffers('network.step')

    def test_multiple_steps(self):
        network = Network(in_adjacency_dict=make_loop_adjacency_dict(), node_memory_size=1, edge_memory_size=1, step_rule=basic_step_rule)
        network.initialize(initialize_rule=basic_initialize_rule)
        logger.debug('\n=> test_multiple_steps <=')
        network.write_buffer[0] = np.array([1.0, 1.0, -1.0])
        network._resolve_write_to_read()
        network.debug_log_buffers('initialization')
        for i in range(10):
            network.step()
            network.debug_log_buffers('network.step {}'.format(i+1))

    def test_basic_step_rule(self):
        # basic_step_rule(node_read_buffer, node_write_buffer, node_memory_size, edge_memory_size, edges)
        node_read_buffer = np.array([ 1.00, 1.0, -1.0])
        node_write_buffer = np.array([3.145, 3.145, 3.145])
        expected_write_buffer = np.array([0.5, 0.0, 0.0])
        node_memory_size = 1
        edge_memory_size = 1
        edges = 2
        basic_step_rule(node_read_buffer=node_read_buffer, node_write_buffer=node_write_buffer, node_memory_size=node_memory_size, edge_memory_size=edge_memory_size, edges=edges)
        self.assertSequenceEqual(node_write_buffer.tolist(), expected_write_buffer.tolist())

    def test_large_loop_multiple_steps(self):
        network = Network(in_adjacency_dict=make_loop_adjacency_dict(10000), node_memory_size=5, edge_memory_size=5, step_rule=basic_step_rule)
        logger.debug('\n=> test_large_loop_multiple_steps <=')
        start = time.time()
        network.initialize(initialize_rule=basic_initialize_rule)
        logger.debug('network.initialize took {}'.format(time.time() - start))
        for i in range(10):
            start = time.time()
            network.step()
            logger.debug('network.step took {}'.format(time.time() - start))

    def test_large_waxman_multiple_steps(self):
        logger.debug('\n=> test_large_waxman_multiple_steps <=')
        start = time.time()
        network = Network(in_adjacency_dict=make_waxman_adjacency_dict(1000), node_memory_size=5, edge_memory_size=5, step_rule=basic_step_rule)
        logger.debug('Network() took {}'.format(time.time() - start))
        start = time.time()
        network.initialize(initialize_rule=basic_initialize_rule)
        logger.debug('network.initialize took {}'.format(time.time() - start))
        for i in range(10):
            start = time.time()
            network.step()
            logger.debug('network.step took {}'.format(time.time() - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unit tests.')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    args = parser.parse_args()

    if args.verbose == 0:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose > 1:
        logger.setLevel(logging.DEBUG)

    unittest.main()
