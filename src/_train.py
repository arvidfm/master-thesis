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

import numpy as np

import theano
import theano.tensor as T

import nn.activations
import nn.layers
import nn.init

__all__ = ['MultiSpeakerLayer', 'PositiveWeightLayer', 'GMMLayer', 'discretize']

def discretize(x):
    y = T.zeros(x.shape)
    return T.set_subtensor(y[T.arange(y.shape[0]),x.argmax(axis=1)], 1)

class MultiSpeakerLayer(nn.layers.Layer):

    def __init__(self, datalayer, idlayer, units, speakers,
                 W=nn.init.gaussian(0, 0.01), b=nn.init.constant(0),
                 activation=nn.activations.linear,
                 force_positive_weights=False):
        super().__init__((datalayer, idlayer), output_shape=(units,))
        self._activation = activation
        self._force_positive_weights = force_positive_weights
        self._W = theano.shared(
            W((speakers, *datalayer.output_shape, units)), name='W')
        if b is None:
            self._b = None
        else:
            self._b = theano.shared(b((speakers, units)), name='b')

    def parameters(self, regularizable=False):
        yield self._W
        if self._b is not None and not regularizable:
            return self._b

    def output_for(self, data, speakers, **kwargs):
        speakers = T.cast(speakers, 'int32')
        W = self._W
        if self._force_positive_weights:
            W = abs(W)
        #weighted = T.batched_dot(data, W[speakers])
        weighted = (data[...,np.newaxis] * W[speakers]).sum(axis=1)
        #weighted, updates = theano.scan(
            #fn=lambda x, y, W: T.dot(x, W[y]),
            #sequences=[data, speakers],
            #non_sequences=W)
        #assert len(updates) == 0
        #out = weighted / weighted.sum(axis=1).reshape((-1, 1))

        if self._b is not None:
            return self._activation(weighted + self._b[speakers])
        else:
            return self._activation(weighted)

    def __getstate__(self):
        return self._W, self._b, self._activation, self._force_positive_weights

    def __setstate__(self, state):
        self._W, self._b, self._activation, self._force_positive_weights = state


class PositiveWeightLayer(nn.layers.Layer):

    def __init__(self, parent, units, W=nn.init.gaussian(0, 0.01),
                 activation=nn.activations.linear, normalize_rows=False):
        super().__init__(parent, output_shape=(units,))
        self._normalize_rows = normalize_rows
        self._activation = activation
        self._W = theano.shared(W((*parent.output_shape, units)),
                                name='W')
        self._calculate_V()

    def _calculate_V(self):
        V = abs(self._W)
        if self._normalize_rows:
            V = V / V.sum(axis=1)[:,np.newaxis]
        self._V = V

    def parameters(self, regularizable=False):
        yield self._W

    def output_for(self, input, train_pass):
        return self._activation(T.dot(input, self._V))

    def __getstate__(self):
        return self._activation, self._W, self._normalize_rows

    def __setstate__(self, state):
        self._activation, self._W, self._normalize_rows = state
        self._calculate_V()


class GMMLayer(nn.layers.Layer):

    def __init__(self, parent, units, mu=nn.init.gaussian(0, 1),
                 A=nn.init.gaussian(0, 1), activation=nn.activations.linear):
        super().__init__(parent, output_shape=(units,))
        self._mu = theano.shared(mu((units, *parent.output_shape)), name='mu')
        self._A = theano.shared(A((units, *parent.output_shape)), name='A')
        self._weights = theano.shared(mu((units,)), name='weights')
        self._activation = activation

    def parameters(self, regularizable=False):
        return self._mu, self._A, self._weights

    def output_for(self, input, train_pass):
        diff = input.dimshuffle(0, 'x', 1) - self._mu.dimshuffle('x', 0, 1)
        precision = (self._A**2).dimshuffle('x', 0, 1)
        exponent = (diff**2 * precision).sum(axis=-1)
        logdet = T.log(precision).sum(axis=(0, 2))
        weights = abs(self._weights)
        weights = weights / weights.sum()
        likelihood = logdet/2 - T.log(2*np.pi)*self._mu.shape[-1]/2 - 1/2 * exponent
        return T.log(weights) + likelihood

    def __getstate__(self):
        return self._mu, self._A, self._weights, self._activation

    def __setstate__(self, state):
        self._mu, self._A, self._weights, self._activation = state
