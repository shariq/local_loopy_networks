import unittest
import numpy as np
import random
import argparse
import time

from collections import defaultdict

import logging
logger = logging.getLogger()

from loopy.models.backprop import BackpropModel

class TestBackprop(unittest.TestCase):
    def test_forward_add(self):
        # implement an add network
        # works with positive numbers ; ReLU messes up negative numbers
        model = BackpropModel(input_size=2, hidden_size=1, output_size=1)
        # input_nodes = [0, 1]
        # hidden_nodes = [2]
        # output_nodes = [3]
        # secondary_output_nodes = [4]
        model.initialize_weights({
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
        iterations = 1000
        model = BackpropModel(input_size=2, hidden_size=1, output_size=1)
        model.train(dataset=train_dataset, iterations=iterations)

        ground_truth = []
        model_output = []

        for input_data, output_data in train_dataset:
            model.clean()
            ground_truth.append(output_data[0])
            model_output.append(int(round(model.forward(input_data)[0])))

        self.assertSequenceEqual(ground_truth, model_output)

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
