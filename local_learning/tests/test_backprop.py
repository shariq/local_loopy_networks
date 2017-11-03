import unittest
import numpy as np
import random
import argparse
import time
# import tensorflow as tf

from collections import defaultdict

import logging
logger = logging.getLogger()

import local_learning.models.backprop as backprop
BackpropModel = backprop.BackpropModel

class TestBackprop(unittest.TestCase):
    def test_forward_add(self):
        # implement an add network
        # works with positive numbers ; ReLU messes up negative numbers
        model = BackpropModel(input_size=2, hidden_size=1, output_size=1)
        # input_nodes = [0, 1]
        # hidden_nodes = [2]
        # output_nodes = [3]
        # secondary_output_nodes = [4]
        model.set_weights({
            (0, 2): 1.0,
            (1, 2): 1.0,
            (2, 3): 1.0,
            (3, 4): 1.0
        })
        for a, b, c in [(1, 2, 3), (0, 1, 1), (1, 1, 2), (0, 0, 0)]:
            expected_c = model.forward([a, b])[0]
            self.assertEqual(c, expected_c)


    def test_train_add(self):
        train_dataset = [
            ([0, 0] , [0]),
            ([0, 1] , [1]),
            ([1, 0] , [1]),
            ([1, 1] , [2]),
        ]
        iterations = 100
        backprop.LEARNING_RATE = 0.3
        model = BackpropModel(input_size=2, hidden_size=1, output_size=1)
        # weights = model.get_weights()
        # print('; '.join(['{}: {}'.format(edge, weights[edge]) for edge in [(0,2), (1,2), (2,3), (3,4)]]))
        model.train(dataset=train_dataset, iterations=iterations)

        ground_truth = []
        model_output = []

        for input_data, output_data in train_dataset:
            model.clean()
            ground_truth.append(output_data[0])
            model_output.append(int(round(model.forward(input_data)[0])))

        self.assertSequenceEqual(ground_truth, model_output)


    def test_train_or(self):
        train_dataset = [
            ([0, 0] , [0]),
            ([0, 1] , [1]),
            ([1, 0] , [1]),
            ([1, 1] , [1]),
        ]
        iterations = 400
        backprop.LEARNING_RATE = 0.03
        model = BackpropModel(input_size=2, hidden_size=4, output_size=1)
        # weights = model.get_weights()
        # print('; '.join(['{}: {}'.format(edge, weights[edge]) for edge in [(0,2), (1,2), (2,3), (3,4)]]))
        model.train(dataset=train_dataset, iterations=iterations)

        ground_truth = []
        model_output = []

        for input_data, output_data in train_dataset:
            model.clean()
            ground_truth.append(output_data[0])
            model_output.append(int(round(model.forward(input_data)[0])))

        self.assertSequenceEqual(ground_truth, model_output)


    def test_train_switch(self):
        train_dataset = [
            ([0, 0] , [0, 0]),
            ([0, 1] , [1, 0]),
            ([1, 0] , [0, 1]),
            ([1, 1] , [1, 1]),
        ]
        iterations = 1000
        backprop.LEARNING_RATE = 0.03
        model = BackpropModel(input_size=2, hidden_size=8, output_size=2)
        # weights = model.get_weights()
        # print('; '.join(['{}: {}'.format(edge, weights[edge]) for edge in [(0,2), (1,2), (2,3), (3,4)]]))
        model.train(dataset=train_dataset, iterations=iterations)

        ground_truth = []
        model_output = []

        for input_data, output_data in train_dataset:
            model.clean()
            ground_truth.append(output_data[0] * 10 + output_data[1])
            sample = model.forward(input_data)
            print(sample)
            model_output.append(int(round(sample[0]))*10 + int(round(sample[1])))

        self.assertSequenceEqual(ground_truth, model_output)


    def test_train_mnist(self):
        data_path = '/tmp/mnist_data'
        iterations = 20000
        # see mnist.py in tensorflow
        return NotImplemented

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
