from functools import wraps, lru_cache
import numpy as np
import math
import inspect
import copy
import random

# operators are the elements of an ExpressionTree which have children

operator_sampler = []

# read leaves.py for an explanation of how operator_sampler works; it does the same thing as leaf_sampler

#############
# helper functions
#############

float_32_dtype = np.dtype('float32')
def np_float_wrapper(func):
    # only use this when you know that the function changes the vector type of an input
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, np.ndarray) and result.dtype != float_32_dtype:
            return result.astype('float32')
        return result
    return wrapper

##############
# operators which take in two floats or vectors, and keep the bigger type
##############

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def subtract(a, b):
    return a - b

@np_float_wrapper
def greater_than_or_equal(a, b):
    return a >= b

@np_float_wrapper
def less_than_or_equal(a, b):
    return a <= b

@np_float_wrapper
def equal(a, b):
    return a == b

operator_sampler.append((add, multiply, subtract))
operator_sampler.append((greater_than_or_equal, less_than_or_equal, equal))

##############
# operators which take in a single float or vector; return same type
##############

def abs_(o):
    return abs(o)

def clip(o):
    return np.clip(o, -1.0, 1.0)

def relu(o):
    # be careful about not modifying the input array in place, like some stack overflow answers say you should!
    return o * (o > 0)

def negative(o):
    return -o

def sigmoid(o):
    return np.exp(-np.logaddexp(0, -o))

def tanh(o):
    return np.tanh(o)

def sin(o):
    return np.sin(o)

def cos(o):
    return np.cos(o)

@np_float_wrapper
def sign(o):
    return np.sign(o)

operator_sampler.append((abs_, clip, relu, negative, (sigmoid, tanh, (sin, cos)), sign))

# these next operators are reducers - they force a float output; so if we need a float somewhere from a vector these are helpful; but generally they can be used for any input/output types; they will just do weird stuff (e.g, std of a float is 0.0; all other reducers of float are float itself)

# we have reduce operators defined on two children so that all cross sections of properties are defined - (number_children=2 x is_reducer=True) was previously not defined. keeping it low likelihood because it's pretty weird... and let's pretend we need to force the vector inputs of reduce_two_operators to be the same length

def sum_(o):
    return np.sum(o)
sum_.is_reducer = True

def sum_two(a, b):
    return np.sum(a) + np.sum(b)
sum_two.is_reducer = True

def max_(o):
    return np.amax(o)
max_.is_reducer = True

def max_two(a, b):
    return max([np.amax(a), np.amax(b)])
max_two.is_reducer = True

def min_(o):
    return np.amin(o)
min_.is_reducer = True

def min_two(a, b):
    return min([np.amin(a), np.amin(b)])
min_two.is_reducer = True

def mean(o):
    return np.mean(o)
mean.is_reducer = True

def mean_two(a, b):
    return float(np.sum(a) + np.sum(b))/(len(a)+len(b))
mean_two.is_reducer = True

def std(o):
    return np.std(o)
std.is_reducer = True

reduce_two_operators = (sum_two, (max_two, min_two), (mean_two))
# reduce_two_operators need to be low likelihood since they get rid of information in two entire child trees
operator_sampler.append((sum_, (max_, min_), (mean, std)) * 20 + reduce_two_operators)

##############
# sampling code
##############

operator_sampler = tuple(operator_sampler)

def get_is_reducer(func):
    return getattr(func, 'is_reducer', False)

def get_number_children(func):
    cached_number_children = getattr(func, 'cached_number_children', None)
    if cached_number_children is None:
        cached_number_children = len(inspect.signature(func).parameters)
        func.cached_number_children = cached_number_children
    return cached_number_children


# agh this requires args to be hashable... so need to turn nested list into nested tuple...
@lru_cache(maxsize=1000, typed=False)
def restrict_operator_sampler(suboperator_sampler, is_reducer, number_children):
    return_value = []
    for e in suboperator_sampler:
        if isinstance(e, list) or isinstance(e, tuple):
            subsuboperator_sampler = restrict_operator_sampler(e, is_reducer, number_children)
            if len(subsuboperator_sampler):
                return_value.append(subsuboperator_sampler)
        else:
            if is_reducer is not None and get_is_reducer(e) != is_reducer:
                continue
            if number_children is not None and get_number_children(e) != number_children:
                continue
            return_value.append(e)
    return tuple(return_value)

def sample_operator(is_reducer=None, number_children=None):
    # None means don't impose the constraint ; other values mean impose that constraint
    assert is_reducer in [None, True, False]
    assert number_children in [None, 1, 2]

    restricted_operator_sampler = restrict_operator_sampler(operator_sampler, is_reducer, number_children)

    while isinstance(restricted_operator_sampler, list) or isinstance(restricted_operator_sampler, tuple):
        restricted_operator_sampler = random.choice(restricted_operator_sampler)

    return restricted_operator_sampler

def render(operator):
    if isinstance(operator, str):
        return operator
    return 'operators.' + operator.__name__


####
# filter helper
####

def undo_filter(vector, filter_vector):
    # also called "blowing up" a filtered vector
    return NotImplemented
