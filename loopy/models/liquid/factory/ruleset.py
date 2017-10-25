import loopy
import random

leaf_generators = {
    'number': generate_number,
    'subtract': generate_subtract,
    'add': generate_add,
    'expression': generate_expression
}

leaf_complexities = {
    'number': (1, 1),
}
# by default it's assumed to be 1, infinite
# in the future we'll have leaves which are like edge vectors

def measure_complexity(obj):
    if isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, bool):
        return 1
    if isinstance(obj, dict):
        return len(obj) + sum([measure_complexity(v) for v in obj.values()])
    raise Exception('unsupported complexity type for obj={} type={}'.format(obj, type(obj)))

def generate_number(**kwargs):
    return random.choice([-1, -1, -1, 0, 0.03, 0.1, 0.3, 1, 1, 1])

def generate_subtract(**kwargs):
    candidates = {
        'number': generate_number,
        'expression': generate_expression
    }

def generate_add(**kwargs):
    candidates = {
        'number': generate_number,
        'expression': generate_expression
    }
    p = {
        'add': 0.5,
        'expression': 0.5
    }


def generate_expression(**kwargs):
    candidates = {
        'add': add_leaf,
    }
    root = {}


class Ruleset:
    def __init__(self):
