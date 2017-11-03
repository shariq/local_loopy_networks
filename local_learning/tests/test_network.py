#!/usr/bin/env python3

import unittest
import numpy as np
import random
import argparse
import time

from collections import defaultdict

import logging
logger = logging.getLogger()

from local_learning.network import Network
# Network(in_adjacency_dict, node_memory_size=1, edge_memory_size=2)
from local_learning.graph.waxman import make_waxman_adjacency_dict
from local_learning.graph.loop import make_loop_adjacency_dict

unittest.run_fast = False


def basic_initialize_rule(node_memory_size, edge_memory_size, edges):
    return np.zeros(node_memory_size + edge_memory_size * edges)


def basic_step_rule(node_read_buffer, node_write_buffer, node_memory_size, edge_memory_size, edges):
    # this step_rule just propagates multidimensional signals around on edges
    # it averages them together

    # it ignores node_memory[1:] and edge_memory[2:] - so does not test for different node_memory_size / edge_memory_size

    NODE_SIGNAL_JUST_SENT_INDEX = 0

    EDGE_HAS_SIGNAL_INDEX = 0
    EDGE_SIGNAL_INDEX = 1

    node_write_buffer[:] = node_read_buffer[:]

    edge_read_memory = node_read_buffer[node_memory_size:]
    edge_write_memory = node_write_buffer[node_memory_size:]

    if node_read_buffer[NODE_SIGNAL_JUST_SENT_INDEX] != 0:
        # we just sent a signal last step; don't do anything this step
        node_write_buffer[NODE_SIGNAL_JUST_SENT_INDEX] = 0
        edge_write_memory[EDGE_SIGNAL_INDEX::edge_memory_size] = 0
        edge_write_memory[EDGE_HAS_SIGNAL_INDEX::edge_memory_size] = 0
        return


    if (edge_read_memory[EDGE_HAS_SIGNAL_INDEX::edge_memory_size] == 0).all() and node_read_buffer[NODE_SIGNAL_JUST_SENT_INDEX] == 0:
        # we have no incoming signals and we didn't sent any signals last step
        node_write_buffer[NODE_SIGNAL_JUST_SENT_INDEX] = 0
        edge_write_memory[EDGE_SIGNAL_INDEX::edge_memory_size] = 0
        edge_write_memory[EDGE_HAS_SIGNAL_INDEX::edge_memory_size] = 0
        return

    # if we reach here that means we have an incoming signal and need to propagate it!
    total_input = 0.0
    number_inputs = 0

    node_write_buffer[NODE_SIGNAL_JUST_SENT_INDEX] = 1.0

    for neighbor in range(edges):
        if edge_read_memory[neighbor*edge_memory_size + EDGE_HAS_SIGNAL_INDEX] != 0:
            number_inputs += 1
            total_input += edge_read_memory[neighbor*edge_memory_size + EDGE_SIGNAL_INDEX]

    output = total_input*2.0 / number_inputs
    # times 2.0 is a really weird requirement here: it's because what we assign an edge gets averaged with what the other node connected to it assigns it; so we multiply by 2 to indicate this is the actual value we want
    # this is a place where a geometric average would be nicer; we're able to indicate if we want to turn it off (set edge memory to 0) or just don't care what the output is (set edge memory to 1)
    # but life becomes complicated; square roots :'X
    # and you still need to set edges to 2 to indicate presence of a signal... since 1*1=1 ...

    for neighbor in range(edges):
        if edge_read_memory[neighbor*edge_memory_size + EDGE_HAS_SIGNAL_INDEX] == 0:
            edge_write_memory[neighbor*edge_memory_size + EDGE_HAS_SIGNAL_INDEX] = 2
            edge_write_memory[neighbor*edge_memory_size + EDGE_SIGNAL_INDEX] = output
        else:
            edge_write_memory[neighbor*edge_memory_size + EDGE_HAS_SIGNAL_INDEX] = 0
            edge_write_memory[neighbor*edge_memory_size + EDGE_SIGNAL_INDEX] = 0
            # make it have no signal


class TestNetwork(unittest.TestCase):
    def test_init(self):
        Network(in_adjacency_dict=make_loop_adjacency_dict())


    def test_initialize(self):
        network = Network(in_adjacency_dict=make_loop_adjacency_dict(), node_memory_size=1, edge_memory_size=2)
        network.initialize(initialize_rule=basic_initialize_rule)


    def test_step(self):
        network = Network(in_adjacency_dict=make_loop_adjacency_dict(), node_memory_size=1, edge_memory_size=2, step_rule=basic_step_rule)
        network.initialize(initialize_rule=basic_initialize_rule)
        logger.debug('\n=> test_step <=')
        network.debug_log_buffers('initialized...')
        network.set_edge_memory((0, 1), np.array([1.0, 1.0]))
        network.set_node_memory(0, np.array([1.0]))
        network.debug_log_buffers('set edge')
        network.step()
        network.debug_log_buffers('network.step')
        self.assertSequenceEqual(network.get_edge_memory((1, 2)).tolist(), [1.0, 1.0])
        self.assertEqual(network.get_node_memory(1)[0], 1.0)
        self.assertEqual(network.get_node_memory(2)[0], 0.0)


    def test_multiple_steps(self):
        network = Network(in_adjacency_dict=make_loop_adjacency_dict(size=20), node_memory_size=1, edge_memory_size=2, step_rule=basic_step_rule)
        network.initialize(initialize_rule=basic_initialize_rule)
        logger.debug('\n=> test_multiple_steps <=')
        network.set_edge_memory((0, 1), np.array([1.0, 1.0]))
        network.set_node_memory(0, np.array([1.0]))
        network.debug_log_buffers('initialization')
        for i in range(10):
            self.assertSequenceEqual(network.get_edge_memory((i, i+1)).tolist(), [1.0, 1.0])
            self.assertEqual(network.get_node_memory(i)[0], 1.0)
            self.assertEqual(network.get_node_memory(i+1)[0], 0.0)
            network.step()
            network.debug_log_buffers('network.step {}'.format(i))


    def test_basic_step_rule(self):
        # basic_step_rule(node_read_buffer, node_write_buffer, node_memory_size, edge_memory_size, edges)
        node_read_buffer = np.array([0., 1., 1., 0., 0.])
        node_write_buffer = 1. * node_read_buffer
        expected_write_buffer = np.array([1., 0., 0., 2., 2.])
        node_memory_size = 1
        edge_memory_size = 2
        edges = 2
        basic_step_rule(node_read_buffer=node_read_buffer, node_write_buffer=node_write_buffer, node_memory_size=node_memory_size, edge_memory_size=edge_memory_size, edges=edges)
        self.assertSequenceEqual(node_write_buffer.tolist(), expected_write_buffer.tolist())


    def test_perf_large_loop_multiple_steps(self):
        if unittest.run_fast:
            self.skipTest('skipping slow tests')
        logger.debug('\n=> test_large_loop_multiple_steps <=')
        start = time.time()
        network = Network(in_adjacency_dict=make_loop_adjacency_dict(10000), node_memory_size=5, edge_memory_size=5, step_rule=basic_step_rule)
        network.initialize(initialize_rule=basic_initialize_rule)
        logger.debug('Network()+network.initialize took {}'.format(time.time() - start))
        for i in range(5):
            start = time.time()
            network.step()
            logger.debug('network.step took {}'.format(time.time() - start))


    def test_perf_large_waxman_multiple_steps(self):
        if unittest.run_fast:
            self.skipTest('skipping slow tests')
        logger.debug('\n=> test_large_waxman_multiple_steps <=')
        start = time.time()
        network = Network(in_adjacency_dict=make_waxman_adjacency_dict(1000), node_memory_size=5, edge_memory_size=5, step_rule=basic_step_rule)
        network.initialize(initialize_rule=basic_initialize_rule)
        logger.debug('Network()+network.initialize took {}'.format(time.time() - start))
        for i in range(5):
            start = time.time()
            network.step()
            logger.debug('network.step took {}'.format(time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unit tests.')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--fast', '-f', action='store_true')
    args = parser.parse_args()

    if args.verbose == 0:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose > 1:
        logger.setLevel(logging.DEBUG)

    unittest.run_fast = args.fast

    unittest.main()
