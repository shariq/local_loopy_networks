import loopy
import unittest
import numpy as np
import random
import argparse

import logging
logger = logging.getLogger()

Network = loopy.Network
# Network(in_adjacency_dict, node_memory_size=1, edge_memory_size=2)

loop_adjacency_dict = {}
for i in range(10):
    loop_adjacency_dict[i] = set([(i+1) % 10])

def basic_initialize_rule(node_memory_size, edge_memory_size, edges):
    return np.zeros(node_memory_size + edge_memory_size * edges)

# node_write_buffer is a reference to be written to
def basic_update_rule(node_read_buffer, node_write_buffer, node_memory_size, edge_memory_size, edges):
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


def log_buffer(buffer):
    print_options = np.get_printoptions()
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    for node, node_buffer in enumerate(buffer):
        logger.debug('{}={}'.format(node, node_buffer))
    np.set_printoptions(**print_options)


def log_buffers(message, network):
    logger.debug(message)
    logger.debug('read_buffer=>')
    log_buffer(network.read_buffer)
    logger.debug('write_buffer=>')
    log_buffer(network.write_buffer)
    logger.debug('\n')


class TestNetwork(unittest.TestCase):
    def test_init(self):
        Network(in_adjacency_dict=loop_adjacency_dict)

    def test_initialize(self):
        network = Network(in_adjacency_dict=loop_adjacency_dict)
        network.initialize(initialize_rule=basic_initialize_rule)

    def test_update(self):
        network = Network(in_adjacency_dict=loop_adjacency_dict)
        network.initialize(initialize_rule=basic_initialize_rule)
        logger.debug('\n=> test_update <=')
        log_buffers('initialized...', network)
        network.write_buffer[0] = np.array([1.0, 1.0, -1.0])
        log_buffers('manual overwrite partial', network)
        network._resolve_write_to_read()
        log_buffers('resolve_write_to_read', network)
        network.update(update_rule=basic_update_rule)
        log_buffers('network.update', network)

    def test_multiple_updates(self):
        network = Network(in_adjacency_dict=loop_adjacency_dict)
        network.initialize(initialize_rule=basic_initialize_rule)
        logger.debug('\n=> test_multiple_updates <=')
        network.write_buffer[0] = np.array([1.0, 1.0, -1.0])
        network._resolve_write_to_read()
        log_buffers('initialization', network)
        network.update(update_rule=basic_update_rule)
        log_buffers('network.update 1', network)
        network.update(update_rule=basic_update_rule)
        log_buffers('network.update 2', network)
        network.update(update_rule=basic_update_rule)
        log_buffers('network.update 3', network)
        network.update(update_rule=basic_update_rule)
        log_buffers('network.update 4', network)
        network.update(update_rule=basic_update_rule)
        log_buffers('network.update 5', network)
        network.update(update_rule=basic_update_rule)
        log_buffers('network.update 6', network)

    def test_basic_update_rule(self):
        # basic_update_rule(node_read_buffer, node_write_buffer, node_memory_size, edge_memory_size, edges)
        node_read_buffer = np.array([ 1.00, 1.0, -1.0])
        node_write_buffer = np.array([3.145, 3.145, 3.145])
        expected_write_buffer = np.array([0.5, 0.0, 0.0])
        node_memory_size = 1
        edge_memory_size = 1
        edges = 2
        basic_update_rule(node_read_buffer=node_read_buffer, node_write_buffer=node_write_buffer, node_memory_size=node_memory_size, edge_memory_size=edge_memory_size, edges=edges)
        self.assertSequenceEqual(node_write_buffer.tolist(), expected_write_buffer.tolist())


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
