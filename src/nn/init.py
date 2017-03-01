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

import numpy as np

import theano

def initializer(f):
    @functools.wraps(f)
    def wrapper(*args, dtype=theano.config.floatX, **kwargs):
        def init(shape):
            return f(*args, **kwargs, shape=shape).astype(dtype)
        return init
    return wrapper

@initializer
def constant(c, *, shape):
    return np.ones(shape) * c

@initializer
def gaussian(mean, std, *, shape):
    return np.random.normal(mean, std, shape)

@initializer
def glorot(c=1, *, shape):
    return np.random.uniform(-np.sqrt(6 / (shape[-1] + shape[-2])),
                             np.sqrt(6 / (shape[-1] + shape[-2])), size=shape) * c

if __name__ == '__main__':
    print(gaussian(0, 1, dtype=int)((3, 3)))
