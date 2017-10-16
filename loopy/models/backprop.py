import numpy as np
import random
import loopy

node_memory_size = 3
edge_memory_size = 3
# node_memory: forward_signal, forward_signal_sent, error_signal_sent
# edge_memory: signal (not multiplied by weight), error, weight

NODE_FORWARD_SIGNAL_INDEX = 0
NODE_FORWARD_SIGNAL_SENT_INDEX = 1
NODE_ERROR_SIGNAL_SENT_INDEX = 2
EDGE_SIGNAL_INDEX = 0
EDGE_ERROR_INDEX = 1
EDGE_WEIGHT_INDEX = 2

from loopy.graph.feedforward import make_feedforward_adjacency_dict
from loopy.network import Network


def backprop_initialize_rule(node_memory_size, edge_memory_size, edges):
    node_buffer = np.zeros(node_memory_size + edge_memory_size * edges)
    node_buffer[NODE_FORWARD_SIGNAL_INDEX] = 0.0
    node_buffer[NODE_FORWARD_SIGNAL_SENT_INDEX] = 0.0
    node_buffer[NODE_ERROR_SIGNAL_SENT_INDEX] = 0.0
    node_buffer[node_memory_size + EDGE_SIGNAL_INDEX::edge_memory_size] = 0.0
    node_buffer[node_memory_size + EDGE_ERROR_INDEX::edge_memory_size] = 0.0
    node_buffer[node_memory_size + EDGE_WEIGHT_INDEX::edge_memory_size] = random.random() - 0.5
    return node_buffer


def backprop_step_rule(node_read_buffer, node_write_buffer, node_memory_size, edge_memory_size, edges):
    return NotImplemented


class BackpropModel:
    LEARNING_RATE = 1e-3
    
    def __init__(self, input_size=2, hidden_size=1, output_size=1, layers=1):
        if layers != 1:
            raise Exception('BackpropModel does not support layers != 1 just yet')
        self.network = Network(in_adjacency_dict=make_feedforward_adjacency_dict(input_size=input_size, hidden_size=hidden_size, output_size=output_size, layers=layers), node_memory_size=3, edge_memory_size=3)
        network.initialize(initialize_rule=backprop_initialize_rule)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers

    def train(self, dataset, iterations=100):
        for iteration in range(iterations):
            input_data, output_data = random.choice(train_dataset)
            self.async_train(input_data, output_data)

    def clean(self):
        # clear everything except weights
        for node_buffer in self.network.read_buffer:
            node_memory = node_buffer[:node_memory_size]
            edge_memory = node_buffer[node_memory_size:]
            node_memory[NODE_FORWARD_SIGNAL_INDEX] = 0.0
            node_memory[NODE_FORWARD_SIGNAL_SENT_INDEX] = 0.0
            node_memory[NODE_ERROR_SIGNAL_SENT_INDEX] = 0.0
            edge_memory[EDGE_SIGNAL_INDEX] = 0.0
            edge_memory[EDGE_ERROR_INDEX] = 0.0
            # dont reset weights

    def forward(self, input_data):
        # set input data
        for node, signal in enumerate(input_data):
            self.network.read_buffer[node][NODE_FORWARD_SIGNAL_INDEX] = signal
            for neighbor in self.network.adjacency_dict[node]:
                edge_memory = np.zeros(edge_memory_size)
                edge_memory[EDGE_SIGNAL_INDEX] = signal
                self.network.set_edge_memory(edge=(node, neighbor), edge_memory=edge_memory)

        # propagate input data forward
        # note that right now multiple input signals coming in out of sync would make the network send output data along input edges
        # so don't try doing anything fancy to handle out of sync signals, since there's no way around this unfortunate fact

        for layer in range(self.layers + 2):
            self.network.step()

        output = []
        output_nodes = list(range(len(self.network.nodes) - self.output_size, len(self.network_nodes)))

        for output_node in output_nodes:
            output.append(self.network.read_buffer[output_node][NODE_FORWARD_SIGNAL_INDEX])

        return output

    def backward(self, output_data):
        # apply error to the last nodes/edges
        # then run step a few times
        return NotImplemented

    def async_train(self, input_data, output_data):
        self.clean()
        self.forward(input_data)
        self.backward(output_data)
