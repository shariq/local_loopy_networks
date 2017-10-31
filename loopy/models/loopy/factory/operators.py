from functools import wrap
import numpy as np
import math
import inspect

float_32_dtype = np.dtype('float32')

# operators are the elements of an ExpressionTree which have children

#############
# helper functions
#############

def number_args(func):
    return len(inspect.signature(func).parameters)

def np_float_wrapper(func):
    # only use this when you know that the function changes the vector type of an input
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, np.array) and result.dtype != float_32_dtype:
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


##############
# operators which take in a single float or vector; return same type
##############

abs = abs

def clip(a):
    return np.clip(a, -1.0, 1.0)

def relu(a):
    # be careful about not modifying the input array in place!
    return a * (a > 0)

def negative(a):
    return -a

def sigmoid(a):
    return np.exp(-np.logaddexp(0, -x))

tanh = np.tanh
sin = np.sin

@np_float_wrapper
def sign(a):
    return np.sign(a)

##############
# reducing operators; take a vector return a float
##############

sum = np.sum
max = np.amax
min = np.amin
mean = np.mean
std = np.std
