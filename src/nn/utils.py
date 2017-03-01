# Copyright (C) 2016  Arvid Fahlström Myrman
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import functools
import inspect
import operator
import tempfile

import numpy as np

import theano
import theano.tensor as T

def ensure_tuple(x):
    if isinstance(x, tuple):
        return x
    else:
        return (x,)

def factory_function(f):
    sig = inspect.signature(f)
    keywords = [name for name, param in sig.parameters.items()
                if param.kind == inspect.Parameter.KEYWORD_ONLY]

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        def inner(*args2, **kwargs2):
            return f(*args, **kwargs, **dict(zip(keywords, args2)), **kwargs2)
        return inner
    return wrapper

def input_to_shared(inputs):
    to_shared = lambda x: theano.shared(np.zeros((0, *x.shape[1:]), dtype=x.dtype))
    return [to_shared(input) for input in inputs]

def shuffle_inputs(inputs):
    data_size = inputs[0].shape[0]
    perms = np.random.permutation(data_size)
    return [input[perms] for input in inputs]
