import numpy as np
import math
import inspect
import copy
import random
from functools import lru_cache


# leaves are the elements of an ExpressionTree which cannot have children
# these functions almost all require some context to evaluate
# context is basically whatever was passed to the step/initialize rule: number of edges, edge_memory_size, node_memory_size,...

leaf_sampler = []

# this is a tree which is stored as arbitrarily nested lists
# at each level of the tree, child elements have an equal likelihood of being sampled

# requires_vector: must be used as a vector [float may be used as a vector or float (floats can be cast to vectors)]
# requires_step: must be used in a step function, not initialize function


###############
# leaves which can only run in an step function
###############

def edge_memory(context, edge_index):
    edge_memory_size = context['edge_memory_size']
    node_memory_size = context['node_memory_size']
    node_read_buffer = context['node_read_buffer']
    return node_read_buffer[node_memory_size + edge_index::edge_memory_size]
edge_memory.requires_vector = True
edge_memory.requires_step = True

def node_memory(context, node_index):
    node_read_buffer = context['node_read_buffer']
    return node_read_buffer[node_index]
node_memory.requires_step = True


leaf_sampler.append((edge_memory, node_memory))

##############
# leaves which can run in step/initialize
##############

# by default all vectors are number of edges long
# getting vectors to be smaller is done through filters
def gaussian(context, leaf_type):
    if leaf_type != 'vector':
        return np.random.normal()
    number_edges = context['edges']
    return np.random.normal(size=number_edges)

def uniform(context, leaf_type):
    if leaf_type != 'vector':
        return np.random.uniform(-1, 1)
    number_edges = context['edges']
    return np.random.uniform(-1, 1, size=number_edges)


leaf_sampler.append((gaussian, uniform))

###############
# leaves which don't take arguments/context
###############

def constant_leaf(constant, context, leaf_type):
    if leaf_type != 'vector':
        return constant
    leaf = np.zeros(context['edges'])
    leaf[:] = constant
    return leaf

def one(context, leaf_type):
    return constant_leaf(1.0, context, leaf_type)

def zero(context, leaf_type):
    return constant_leaf(0.0, context, leaf_type)

def minus_one(context, leaf_type):
    return constant_leaf(-1.0, context, leaf_type)

def two(context, leaf_type):
    return constant_leaf(2.0, context, leaf_type)

def minus_two(context, leaf_type):
    return constant_leaf(-2.0, context, leaf_type)

def point_one(context, leaf_type):
    return constant_leaf(0.1, context, leaf_type)

def point_zero_one(context, leaf_type):
    return constant_leaf(0.01, context, leaf_type)

def point_zero_zero_one(context, leaf_type):
    return constant_leaf(0.001, context, leaf_type)

leaf_sampler.append(((one, zero, minus_one), (one, zero, minus_one, two, minus_two, point_one, point_zero_one, point_zero_zero_one)))


#####################
# sampler functions
#####################

def context_renderer(func, expression_type, *args, **kwargs):
    assert expression_type in ['initialize', 'step', 'filter', 'conditional']

    # filter and conditional must never be used in an initialize function, only in a step function

    if expression_type != 'initialize':
        rendered_context = "{'node_read_buffer': node_read_buffer, 'edge_memory_size': edge_memory_size, 'node_memory_size': node_memory_size, 'edges': edges}"
    else:
        rendered_context = "{'node_memory_size': node_memory_size, 'edge_memory_size': edge_memory_size, 'edges': edges}"

    # requires that context is first arg to func
    # *args and **kwargs are bound to func at rendertime; *not* runtime!
    # rendered_context is evaluated at runtime

    func_args = [rendered_context] + [repr(arg) for arg in args] + ['{}={}'.format(str(k), repr(v)) for k, v in kwargs.items()]

    func_arg_string = ', '.join(func_args)

    function_call = 'leaves.' + func.__name__ + '(' + func_arg_string + ')'
    assert func.__module__.endswith('leaves'), 'Error: you tried running context_renderer on a function which was not in the leaves module; this is not supported!'

    return function_call


def get_requires_vector(leaf):
    return getattr(leaf, 'requires_vector', False)


def get_requires_step(leaf):
    return getattr(leaf, 'requires_step', False)


leaf_sampler = tuple(leaf_sampler)

@lru_cache(maxsize=1000, typed=False)
def restrict_leaf_sampler(subleaf_sampler, requires_step, requires_vector):
    return_value = []
    for e in subleaf_sampler:
        if isinstance(e, list) or isinstance(e, tuple):
            subsubleaf_sampler = restrict_leaf_sampler(e, requires_step, requires_vector)
            if len(subsubleaf_sampler):
                return_value.append(subsubleaf_sampler)
        else:
            if requires_step is not None and get_requires_step(e) != requires_step:
                continue
            if requires_vector is not None and get_requires_vector(e) != requires_vector:
                continue
            return_value.append(e)
    return tuple(return_value)


def sample_leaf(requires_step=None, requires_vector=None):
    assert requires_step in [None, True, False]
    assert requires_vector in [None, True, False]

    restricted_leaf_sampler = restrict_leaf_sampler(leaf_sampler, requires_step, requires_vector)

    while isinstance(restricted_leaf_sampler, list) or isinstance(restricted_leaf_sampler, tuple):
        restricted_leaf_sampler = random.choice(restricted_leaf_sampler)

    return restricted_leaf_sampler


def render_leaf(func, expression_type, leaf_type, node_memory_size, edge_memory_size):
    parameters = list(inspect.signature(func).parameters)
    if len(parameters) == 0:
        return 'leaves.' + func.__name__ + '()'
    assert 'context' in parameters, 'right now we always use context_renderer for leaves with args so not sure what this leaf is but it\'s bad because it doesn\'t have a context parameter'
    renderer_kwargs = {}
    if 'leaf_type' in parameters:
        renderer_kwargs['leaf_type'] = leaf_type
    if 'node_index' in parameters:
        renderer_kwargs['node_index'] = random.randint(0, node_memory_size-1)
    if 'edge_index' in parameters:
        renderer_kwargs['edge_index'] = random.randint(0, edge_memory_size-1)
    assert len(renderer_kwargs) + 1 == len(parameters), 'could not account for all parameters of this leaf; is it using any nonstandard parameter names?'
    return context_renderer(func, expression_type, **renderer_kwargs)


def sample_rendered_leaf(expression_type, leaf_type, node_memory_size, edge_memory_size):
    assert expression_type in ['step', 'initialize', 'filter', 'conditional']
    assert leaf_type in ['float', 'vector']

    requires_step = None
    requires_vector = None

    if expression_type == 'filter' or expression_type == 'conditional':
        requires_step = random.choice([True, True, True, True, False])

    if expression_type == 'step':
        requires_step = random.choice([True, False])

    if expression_type == 'initialize':
        requires_step = False

    if leaf_type == 'float':
        requires_vector = False

    if leaf_type == 'vector':
        requires_vector = random.choice([True, False])

    if requires_vector and not requires_step:
        # don't have any such operations defined currently
        requires_vector = False

    func = sample_leaf(requires_step, requires_vector)
    return render_leaf(func, expression_type, leaf_type, node_memory_size, edge_memory_size)
