def initialize_rule(node_memory_size, edge_memory_size, edges):
    node_write_buffer = np.zeros(node_memory_size + edge_memory_size * edges)
    try:
        slot_edge_0=node_write_buffer[6+0::5]
        slot_edge_1=node_write_buffer[6+1::5]
        slot_edge_2=node_write_buffer[6+2::5]
        slot_edge_3=node_write_buffer[6+3::5]
        slot_edge_4=node_write_buffer[6+4::5]
        node_write_buffer[0]=operators.sum_(leaves.gaussian({'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}, leaf_type='vector'))
        node_write_buffer[1]=operators.sum_(leaves.gaussian({'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}, leaf_type='vector'))
        node_write_buffer[2]=operators.std(leaves.gaussian({'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}, leaf_type='vector'))
        node_write_buffer[3]=leaves.gaussian({'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}, leaf_type='float')
        node_write_buffer[4]=operators.sum_(operators.abs_(leaves.point_zero_zero_one({'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}, leaf_type='float')))
        node_write_buffer[5]=operators.mean_two(leaves.uniform({'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}, leaf_type='vector'), leaves.one({'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}, leaf_type='vector'))
        slot_edge_0[:]=operators.ensure_vector(operators.multiply(leaves.uniform({'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}, leaf_type='vector'), leaves.uniform({'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}, leaf_type='vector')), edges, None)
        slot_edge_1[:]=operators.ensure_vector(leaves.zero({'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}, leaf_type='vector'), edges, None)
        slot_edge_2[:]=operators.ensure_vector(operators.std(leaves.gaussian({'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}, leaf_type='vector')), edges, None)
        slot_edge_3[:]=operators.ensure_vector(operators.relu(operators.sum_(leaves.gaussian({'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}, leaf_type='vector'))), edges, None)
        slot_edge_4[:]=operators.ensure_vector(operators.sign(leaves.uniform({'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}, leaf_type='float')), edges, None)
    except Exception as e:

        raise
    return node_write_buffer


def step_rule(node_read_buffer, node_write_buffer, node_memory_size, edge_memory_size, edges):
    node_write_buffer[:] = node_read_buffer
    try:
        slot_edge_0=node_write_buffer[6+0::5]
        slot_edge_1=node_write_buffer[6+1::5]
        slot_edge_2=node_write_buffer[6+2::5]
        slot_edge_3=node_write_buffer[6+3::5]
        slot_edge_4=node_write_buffer[6+4::5]
        slot_filter_0=operators.ensure_vector(operators.equal(operators.sign(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=5)), operators.sum_(operators.abs_(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=2)))), edges, None)>0.0
        slot_filter_1=operators.ensure_vector(operators.negative(operators.max_two(operators.sum_(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=3)), operators.sign(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=3)))), edges, None)>0.0
        slot_filter_2=operators.ensure_vector(leaves.gaussian({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'), edges, None)>0.0
        slot_filter_3=operators.ensure_vector(operators.clip(operators.sum_(operators.std(leaves.point_zero_zero_one({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector')))), edges, None)>0.0
        slot_conditional_0=bool(operators.sum_(operators.min_(operators.min_(operators.std(operators.std(operators.clip(operators.sum_(operators.subtract(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=2), leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=2)))))))))>0.0)
        slot_conditional_1=bool(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=5)>0.0)
        slot_conditional_2=bool(operators.max_(operators.sum_(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=2)))>0.0)
        slot_conditional_3=bool(leaves.gaussian({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='float')>0.0)
        slot_conditional_4=bool(operators.min_(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=0))>0.0)
        node_write_buffer[0]=operators.sum_(operators.undo_filter(operators.apply_filter(leaves.one({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'), slot_filter_2), slot_filter_2))
        if slot_conditional_3:node_write_buffer[1]=operators.mean_two(operators.undo_filter(operators.apply_filter(leaves.point_one({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'), slot_filter_2), slot_filter_2), operators.sum_(operators.mean(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=2))))
        if slot_conditional_4:node_write_buffer[2]=operators.mean_two(operators.undo_filter(operators.max_two(operators.apply_filter(operators.sign(leaves.point_zero_zero_one({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='float')), slot_filter_2), operators.abs_(operators.apply_filter(leaves.gaussian({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'), slot_filter_2))), slot_filter_2), operators.equal(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=0), leaves.zero({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector')))
        node_write_buffer[3]=operators.min_two(operators.undo_filter(operators.sum_two(operators.apply_filter(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=1), slot_filter_3), operators.apply_filter(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=4), slot_filter_3)), slot_filter_3), operators.undo_filter(operators.apply_filter(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=0), slot_filter_1), slot_filter_1))
        node_write_buffer[4]=operators.max_two(leaves.one({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'), operators.undo_filter(operators.apply_filter(leaves.gaussian({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'), slot_filter_0), slot_filter_0))
        node_write_buffer[5]=operators.sum_two(operators.undo_filter(operators.apply_filter(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=0), slot_filter_1), slot_filter_1), operators.undo_filter(operators.max_(operators.apply_filter(leaves.minus_one({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'), slot_filter_0)), slot_filter_0))
        if slot_conditional_0:slot_edge_0[:]=operators.ensure_vector(operators.add(operators.sum_(operators.multiply(operators.std(operators.undo_filter(operators.apply_filter(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=4), slot_filter_2), slot_filter_2)), operators.max_(leaves.minus_one({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector')))), operators.std(operators.undo_filter(operators.sum_(operators.apply_filter(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=4), slot_filter_0)), slot_filter_0))), edges, None)
        if slot_conditional_1:slot_edge_0[:]=operators.ensure_vector(operators.multiply(operators.undo_filter(operators.add(operators.apply_filter(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=0), slot_filter_3), operators.apply_filter(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=0), slot_filter_3)), slot_filter_3), operators.less_than_or_equal(operators.sum_(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=4)), operators.std(operators.undo_filter(operators.apply_filter(leaves.point_zero_zero_one({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'), slot_filter_1), slot_filter_1)))), edges, None)
        slot_edge_0[:]=operators.ensure_vector(operators.mean(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=2)), edges, None)
        if slot_conditional_4:slot_edge_0[:]=operators.ensure_vector(operators.sum_(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=0)), edges, None)
        if slot_conditional_2:slot_edge_1[:]=operators.ensure_vector(operators.multiply(operators.multiply(operators.sum_(leaves.gaussian({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector')), operators.max_(operators.undo_filter(operators.apply_filter(leaves.point_zero_zero_one({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'), slot_filter_1), slot_filter_1))), operators.std(leaves.gaussian({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'))), edges, None)
        slot_edge_1[:]=operators.ensure_vector(operators.sum_two(operators.min_two(operators.max_(operators.std(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=3))), leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=1)), operators.tanh(operators.undo_filter(operators.apply_filter(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=1), slot_filter_1), slot_filter_1))), edges, None)
        slot_edge_1[:]=operators.ensure_vector(operators.subtract(operators.sum_(operators.negative(operators.mean_two(operators.undo_filter(operators.mean(operators.apply_filter(leaves.uniform({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'), slot_filter_0)), slot_filter_0), operators.undo_filter(operators.sum_(operators.apply_filter(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=1), slot_filter_2)), slot_filter_2)))), operators.sum_(operators.undo_filter(operators.sum_(operators.apply_filter(leaves.one({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'), slot_filter_3)), slot_filter_3))), edges, None)
        slot_edge_2[:]=operators.ensure_vector(operators.sum_(operators.max_two(operators.min_two(operators.undo_filter(operators.sum_(operators.apply_filter(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=3), slot_filter_0)), slot_filter_0), operators.sum_(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=3))), leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=5))), edges, None)
        slot_edge_2[:]=operators.ensure_vector(operators.sum_two(operators.add(operators.undo_filter(operators.greater_than_or_equal(operators.clip(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=4)), operators.max_(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=2))), slot_filter_0), operators.undo_filter(operators.max_(operators.apply_filter(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=2), slot_filter_0)), slot_filter_0)), operators.sum_(operators.sign(operators.undo_filter(operators.apply_filter(leaves.gaussian({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'), slot_filter_1), slot_filter_1)))), edges, None)
        slot_edge_2[:]=operators.ensure_vector(operators.std(leaves.uniform({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector')), edges, None)
        if slot_conditional_1:slot_edge_3[:]=operators.ensure_vector(operators.mean(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=4)), edges, None)
        slot_edge_3[:]=operators.ensure_vector(operators.relu(operators.relu(operators.less_than_or_equal(operators.max_(operators.undo_filter(operators.apply_filter(leaves.edge_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, edge_index=4), slot_filter_2), slot_filter_2)), operators.min_(leaves.uniform({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'))))), edges, None)
        slot_edge_3[:]=operators.ensure_vector(operators.less_than_or_equal(operators.max_two(operators.undo_filter(operators.apply_filter(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=3), slot_filter_2), slot_filter_2), operators.undo_filter(operators.apply_filter(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=5), slot_filter_3), slot_filter_3)), operators.undo_filter(operators.apply_filter(leaves.node_memory({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, node_index=2), slot_filter_1), slot_filter_1)), edges, None)
        if slot_conditional_2:slot_edge_4[:]=operators.ensure_vector(operators.tanh(operators.clip(leaves.uniform({'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}, leaf_type='vector'))), edges, None)
    except Exception as e:
        logger.error(e, exc_info=True)

        raise



import numpy as np
import random

import local_learning
import local_learning.network
import local_learning.graph.tools
import local_learning.models.loopy.factory.operators as operators
import local_learning.models.loopy.factory.leaves as leaves
import local_learning.models.loopy
import local_learning.graph.feedforward

import logging
logger = logging.getLogger()

class Model:

    def __init__(self, input_size=2, output_size=1):
        # modify adjacency_dict to add single edge output nodes, so it's easier to apply error
        adjacency_dict = local_learning.graph.feedforward.make_feedforward_adjacency_dict(input_size=input_size, hidden_size=4, output_size=output_size, secondary_output_layer=False, layers=1)
        undirected_adjacency_dict = local_learning.graph.tools.undirect_adjacency_dict(adjacency_dict)
        regular_output_nodes = list(range(len(undirected_adjacency_dict) - output_size, len(undirected_adjacency_dict)))
        for regular_output_node in regular_output_nodes:
            new_output_node = regular_output_node + output_size
            for node_a, node_b in [(regular_output_node, new_output_node), (new_output_node, regular_output_node)]:
                # keep the graph undirected
                if not undirected_adjacency_dict.get(node_a, False):
                    undirected_adjacency_dict[node_a] = set()
                undirected_adjacency_dict[node_a].add(node_b)

        self.network = local_learning.network.Network(in_adjacency_dict=undirected_adjacency_dict, node_memory_size=6, edge_memory_size=5, step_rule=step_rule)
        self.network.initialize(initialize_rule=initialize_rule)
        self.input_size = input_size
        self.output_size = output_size
        self.output_nodes = list(range(len(self.network.nodes) - self.output_size, len(self.network.nodes)))

    def forward(self, input_data, max_steps=300):
        # set input data
        for node, signal in enumerate(input_data):
            self.network.read_buffer[node][0] = signal
            self.network.read_buffer[node][1] = 1
            for neighbor in self.network.adjacency_dict[node]:
                edge_memory = self.network.get_edge_memory((node, neighbor))[:]
                edge_memory[0] = signal
                edge_memory[1] = 1
                self.network.set_edge_memory(edge=(node, neighbor), edge_memory=edge_memory)

        # propagate input data forward
        # note that right now multiple input signals coming in out of sync would make the network send output data along input edges
        # so don't try doing anything fancy to handle out of sync signals, since there's no way around this unfortunate fact

        # wait for all output_nodes to have NODE_SIGNAL_JUST_SENT_INDEX

        steps = 0
        while True:
            self.network.step()
            ready = all(self.network.get_node_memory(output_node)[1] > 0.5 for output_node in self.output_nodes)
            if ready:
                break
            steps += 1
            if steps > max_steps:
                break

        output = []
        info = {}
        info['steps'] = steps

        for output_node in self.output_nodes:
            output.append(self.network.read_buffer[output_node][0])

        return output, info

    def backward(self, output_data, max_steps=300):
        # apply error to the secondary output edges
        # then run step until there's no error left
        for expected_signal, output_node in zip(output_data, self.output_nodes):
            actual_signal = self.network.read_buffer[output_node][0]

            error = expected_signal - actual_signal

            for neighbor in self.network.adjacency_dict[output_node]:
                edge_memory = self.network.get_edge_memory((output_node, neighbor))
                edge_memory[3] = 1
                edge_memory[2] = error
                self.network.set_edge_memory((output_node, neighbor), edge_memory)

            self.network.read_buffer[output_node][2] = 1

        steps = 0
        info = {}

        while True:
            self.network.step()
            ready = all(node_buffer[2] <= 0.5 for node_buffer in self.network.read_buffer)
            steps += 1
            if ready or steps > max_steps:
                break
        info['steps'] = steps

        return None, info

    def train(self, dataset, iterations=100):
        for iteration in range(iterations):
            print(iteration, 'iteration')
            input_data, output_data = random.choice(dataset)
            self.async_train(input_data, output_data)

    def async_train(self, input_data, output_data):
        print('input_data', input_data, 'output_data', output_data)
        output, info = self.forward(input_data)
        print('forward:', 'output', output, 'info', info)
        output, info = self.backward(output_data)
        print('backward:', 'output', output, 'info', info)
