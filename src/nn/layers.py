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
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from . import init, activations, utils

class PickleableLayer(type):

    def __new__(cls, name, bases, namespace):
        # NOTE: root class name is hard coded
        if name != 'Layer' and ('__getstate__' in namespace and
                                '__setstate__' in namespace):
            gs, ss = namespace['__getstate__'], namespace['__setstate__']

            @functools.wraps(gs)
            def __getstate__(self):
                super_state = super(newcls, self).__getstate__()
                return super_state, gs(self)

            @functools.wraps(ss)
            def __setstate__(self, state):
                super_state, state = state
                super(newcls, self).__setstate__(super_state)
                ss(self, state)

            namespace['__getstate__'] = __getstate__
            namespace['__setstate__'] = __setstate__

        newcls = type.__new__(cls, name, bases, namespace)
        return newcls

# NOTE: also update PickleableLayer if changing the name of the class
class Layer(metaclass=PickleableLayer):

    def __init__(self, parents=(), inputs=(), output_shape=None):
        self.parents = utils.ensure_tuple(parents)
        self._inputs = inputs
        self.output_shape = output_shape

    def all_layers(self):
        for parent in self.parents:
            yield from parent.all_layers()

        yield self

    def parameters(self):
        return ()

    def all_parameters(self):
        for layer in self.all_layers():
            yield from layer.parameters()

    def inputs(self):
        return self._inputs

    def process_input_dict(self, inputs):
        givens = {}

        for layer in self.all_layers():
            if len(layer.inputs()) == 0:
                continue

            try:
                layer_inputs = utils.ensure_tuple(inputs[layer])
            except KeyError:
                raise ValueError("Missing input for layer {!r}".format(layer))

            if len(layer_inputs) != len(layer.inputs()):
                raise ValueError(("Wrong number of inputs for layer {!r}: "
                                "got {}, expected {}".format(
                                    layer, len(layer_inputs), len(layer.inputs()))))

            givens.update(zip(layer.inputs(), layer_inputs))

        return givens

    def all_inputs(self):
        for layer in self.all_layers():
            yield from layer.inputs()

    def output_for(self, *inputs, train_pass):
        raise NotImplementedError

    def output(self, inputs=None, train_pass=False):
        layer_output = self.output_for(*(parent.output(train_pass=train_pass)
                                         for parent in self.parents),
                                       train_pass=train_pass)
        if inputs is not None:
            givens = self.process_input_dict(inputs)
            layer_output = theano.clone(layer_output, givens)

        return layer_output

    def __getstate__(self):
        return self.parents, self._inputs, self.output_shape

    def __setstate__(self, state):
        self.parents, self._inputs, self.output_shape = state

class InputLayer(Layer):

    def __init__(self, shape, dtype=theano.config.floatX):
        ttype = T.TensorType(dtype, (False,) * (len(shape) + 1))
        self._input = ttype('input')
        super().__init__(inputs=(self._input,), output_shape=shape)

    def output_for(self, **kwargs):
        return self._input

    def __getstate__(self):
        return self._input

    def __setstate__(self, state):
        self._input = state

class ReshapeLayer(Layer):

    def __init__(self, parent, shape):
        super().__init__(parent, output_shape=shape)
        self._shape = shape

    def output_for(self, input, **kwargs):
        return input.reshape((-1, *self._shape))

    def __getstate__(self):
        return self._shape,

    def __setstate__(self, state):
        self._shape, = state

class FeedForwardLayer(Layer):

    def __init__(self, parent, units, W=init.gaussian(0, 0.01),
                 b=init.constant(0), activation=activations.linear, maxout=1):
        super().__init__(parent, output_shape=(units,))
        self._activation = activation
        self._W = theano.shared(W((maxout, *parent.output_shape, units)),
                                name='W')
        self._b = theano.shared(b((maxout, units)), name='b')

    def parameters(self, regularizable=False):
        yield self._W
        if not regularizable:
            yield self._b

    def output_for(self, input, train_pass):
        out = T.dot(input, self._W) + self._b
        maxout = out.max(axis=1)
        return self._activation(maxout)

    def __getstate__(self):
        return self._activation, self._W, self._b

    def __setstate__(self, state):
        self._activation, self._W, self._b = state

class GaussianLayer(Layer):

    def __init__(self, parent, std, seed=None):
        super().__init__(parent, output_shape=parent.output_shape)
        self.std = std
        seed = seed or np.random.randint(1, 1 << 30, (6,))
        self.rng = RandomStreams(seed=seed)

    def output_for(self, input, train_pass):
        if train_pass:
            noise = self.rng.normal(self.output_shape, std=self.std,
                                    dtype=theano.config.floatX)
            return input + noise
        else:
            return input

    def __getstate__(self):
        return self.std, self.rng

    def __setstate__(self, state):
        self.std, self.rng = state


class DropoutLayer(Layer):

    def __init__(self, parent, dropout_rate, seed=None):
        super().__init__(parent, output_shape=parent.output_shape)
        self.alive_prob = 1 - dropout_rate
        seed = seed or np.random.randint(1, 1 << 30, (6,))
        self.rng = RandomStreams(seed=seed)

    def output_for(self, input, train_pass):
        if train_pass:
            # NOTE: dtype needs to be specified in order to avoid casting to float64
            mask = self.rng.binomial((input.shape[1],), p=self.alive_prob,
                                     dtype=input.dtype)
            return input * mask / self.alive_prob
        else:
            return input

    def __getstate__(self):
        return self.alive_prob, self.rng

    def __setstate__(self, state):
        self.alive_prob, self.rng = state

if __name__ == '__main__':
    pass
