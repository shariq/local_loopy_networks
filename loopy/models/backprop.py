import numpy as np
import random
import loopy

import logging
logger = logging.getLogger()

node_memory_size = 3
edge_memory_size = 5
# node_memory: signal_memory, signal_just_sent, error_just_sent
# edge_memory: signal (not multiplied by weight), has_signal, error, has_error, weight

# some help for thinking about error...
# error = (expected_signal - actual_signal) * gradient

NODE_SIGNAL_MEMORY_INDEX = 0
NODE_SIGNAL_JUST_SENT_INDEX = 1
NODE_ERROR_JUST_SENT_INDEX = 2

EDGE_SIGNAL_INDEX = 0
EDGE_HAS_SIGNAL_INDEX = 1
EDGE_ERROR_INDEX = 2
EDGE_HAS_ERROR_INDEX = 3
EDGE_WEIGHT_INDEX = 4


from loopy.graph.feedforward import make_feedforward_adjacency_dict
from loopy.network import Network


def backprop_initialize_rule(node_memory_size, edge_memory_size, edges):
    node_buffer = np.zeros(node_memory_size + edge_memory_size * edges)
    node_buffer[NODE_SIGNAL_MEMORY_INDEX] = 0
    node_buffer[NODE_SIGNAL_JUST_SENT_INDEX] = 0
    node_buffer[NODE_ERROR_JUST_SENT_INDEX] = 0
    node_buffer[node_memory_size + EDGE_SIGNAL_INDEX::edge_memory_size] = 0.0
    node_buffer[node_memory_size + EDGE_HAS_SIGNAL_INDEX::edge_memory_size] = 0
    node_buffer[node_memory_size + EDGE_ERROR_INDEX::edge_memory_size] = 0.0
    node_buffer[node_memory_size + EDGE_HAS_ERROR_INDEX::edge_memory_size] = 0
    node_buffer[node_memory_size + EDGE_WEIGHT_INDEX::edge_memory_size] = random.random() - 0.5
    return node_buffer


LEAKY_RELU_LEFT_SIDE_COEFFICIENT = 0.1
LEARNING_RATE = 1e-3
def backprop_step_rule(node_read_buffer, node_write_buffer, node_memory_size, edge_memory_size, edges):
    node_write_buffer[:] = node_read_buffer

    if node_read_buffer[NODE_SIGNAL_JUST_SENT_INDEX] == 0 and (node_read_buffer[node_memory_size + EDGE_HAS_SIGNAL_INDEX::edge_memory_size] != 0).any():
        # trigger on any incoming signals
        # part of forward pass
        node_signal = 0.0

        for neighbor in range(edges):
            edge_memory = node_read_buffer[node_memory_size + neighbor * edge_memory_size:node_memory_size + (neighbor + 1) * edge_memory_size]
            if edge_memory[EDGE_HAS_SIGNAL_INDEX] != 0:
                # this edge has a signal
                node_signal += edge_memory[EDGE_SIGNAL_INDEX] * edge_memory[EDGE_WEIGHT_INDEX]
            else:
                # this edge does not have a signal
                pass

        # leaky ReLU
        if node_signal < 0.0:
            node_signal = LEAKY_RELU_LEFT_SIDE_COEFFICIENT * node_signal

        for neighbor in range(edges):
            edge_memory = node_read_buffer[node_memory_size + neighbor * edge_memory_size:node_memory_size + (neighbor + 1) * edge_memory_size]
            if edge_memory[EDGE_HAS_SIGNAL_INDEX] != 0:
                # this edge has a signal
                edge_memory[EDGE_HAS_SIGNAL_INDEX] = 0
            else:
                # this edge does not have a signal
                edge_memory[EDGE_SIGNAL_INDEX] = node_signal
                edge_memory[EDGE_HAS_SIGNAL_INDEX] = 1

        node_write_buffer[NODE_SIGNAL_MEMORY_INDEX] = node_signal
        node_write_buffer[NODE_SIGNAL_JUST_SENT_INDEX] = 1
        return

    if node_read_buffer[NODE_SIGNAL_JUST_SENT_INDEX] == 1:
        # trigger if we sent a signal last step
        # part of forward pass
        node_write_buffer[NODE_SIGNAL_JUST_SENT_INDEX] = 0
        node_write_buffer[node_memory_size + EDGE_SIGNAL_INDEX::edge_memory_size] = 0
        node_write_buffer[node_memory_size + EDGE_HAS_SIGNAL_INDEX::edge_memory_size] = 0
        return

    if (node_read_buffer[node_memory_size + EDGE_HAS_ERROR_INDEX::edge_memory_size] != 0).any():
        # trigger on any incoming errors
        # part of backward pass

        actual_signal = node_read_buffer[NODE_SIGNAL_MEMORY_INDEX]
        if actual_signal < 0.0:
            gradient = LEAKY_RELU_LEFT_SIDE_COEFFICIENT
        else:
            gradient = 1.0

        pass



class BackpropModel:
    def __init__(self, input_size=2, hidden_size=1, output_size=1, layers=1):
        if layers != 1:
            raise Exception('BackpropModel does not support layers != 1 just yet')
        self.network = Network(in_adjacency_dict=make_feedforward_adjacency_dict(input_size=input_size, hidden_size=hidden_size, output_size=output_size, secondary_output_layer=True, layers=layers), node_memory_size=node_memory_size, edge_memory_size=edge_memory_size, step_rule=backprop_step_rule)
        self.network.initialize(initialize_rule=backprop_initialize_rule)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers

    def train(self, dataset, iterations=100):
        for iteration in range(iterations):
            input_data, output_data = random.choice(dataset)
            self.async_train(input_data, output_data)

    def clean(self):
        # clear everything except weights
        for node_buffer in self.network.read_buffer:
            node_memory = node_buffer[:node_memory_size]
            edge_memory = node_buffer[node_memory_size:]
            node_memory[NODE_SIGNAL_MEMORY_INDEX] = 0
            node_memory[NODE_SIGNAL_JUST_SENT_INDEX] = 0
            node_memory[NODE_ERROR_JUST_SENT_INDEX] = 0
            edge_memory[EDGE_SIGNAL_INDEX] = 0.0
            edge_memory[EDGE_HAS_SIGNAL_INDEX] = 0
            edge_memory[EDGE_ERROR_INDEX] = 0.0
            edge_memory[EDGE_HAS_ERROR_INDEX] = 0
            # don't reset weights

    def forward(self, input_data):
        # set input data
        for node, signal in enumerate(input_data):
            self.network.read_buffer[node][NODE_SIGNAL_MEMORY_INDEX] = signal
            self.network.read_buffer[node][NODE_SIGNAL_JUST_SENT_INDEX] = signal
            for neighbor in self.network.adjacency_dict[node]:
                edge_memory = np.zeros(edge_memory_size)
                edge_memory[EDGE_SIGNAL_INDEX] = signal
                self.network.set_edge_memory(edge=(node, neighbor), edge_memory=edge_memory)

        # propagate input data forward
        # note that right now multiple input signals coming in out of sync would make the network send output data along input edges
        # so don't try doing anything fancy to handle out of sync signals, since there's no way around this unfortunate fact

        # wait for all secondary_output_nodes to have NODE_SIGNAL_JUST_SENT_INDEX

        secondary_output_nodes = list(range(len(self.network.nodes) - self.output_size, len(self.network.nodes)))

        steps = 0
        while True:
            self.network.debug_log_buffers('step={}'.format(steps))
            self.network.step()
            ready = all(self.network.get_node_memory(output_node)[NODE_SIGNAL_JUST_SENT_INDEX] == 1 for output_node in secondary_output_nodes)
            if ready:
                break
            steps += 1
            if steps > 1000:
                logger.error('ERROR: ran > 1000 steps in a forward pass without getting an output; probably a bug!')
                raise Exception('ran > 1000 steps in forward pass without getting an output')

        output = []

        for output_node in secondary_output_nodes:
            output.append(self.network.read_buffer[output_node][NODE_SIGNAL_MEMORY_INDEX])

        return output

    def backward(self, output_data):
        # apply error to the last nodes/edges
        # then run step a few times
        return NotImplemented

    def initialize_weights(self, weights):
        # weights = {(node_a, node_b): weight, ...}
        self.clean()
        for edge, weight in weights.items():
            edge_memory = np.zeros(edge_memory_size)
            edge_memory[EDGE_WEIGHT_INDEX] = weight
            self.network.set_edge_memory(edge, edge_memory)

    def async_train(self, input_data, output_data):
        self.clean()
        self.forward(input_data)
        self.backward(output_data)
