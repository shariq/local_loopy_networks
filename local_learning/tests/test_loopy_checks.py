#!/usr/bin/env python3
import unittest
import argparse
import time

unittest.run_fast = False

import local_learning
import logging
logger = logging.getLogger()

from types import ModuleType
from local_learning.models.loopy.search import sample_backprop_model_class, sample_factory_model_class, search_harness


class TestLoopyChecks(unittest.TestCase):
    def test_on_backprop(self):
        generator_harness, harness_code, results = next(search_harness(sample_backprop_model_class, catch_exceptions=False))
        average_score = sum(results) / len(results)
        # this is kind of nondeterministic... test sometimes doesn't pass if backprop has bad initialization or something
        logger.info('test_on_backprop got avg score={}'.format(average_score))
        self.assertGreater(average_score, 0.5)


    def test_on_factory(self):
        limit = 100
        if unittest.run_fast:
            limit = 3
        for generator_harness, harness_code, results in search_harness(limit=limit, catch_exceptions=False):
            logger.info('test_on_factory; score={}'.format(sum(results)/len(results)))


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
