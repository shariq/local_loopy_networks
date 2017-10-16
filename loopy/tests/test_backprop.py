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
    def test_add(self):
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
