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
import itertools

import numpy as np

import theano
import theano.tensor as T
from theano.compile import nanguardmode

from . import losses, utils

def optimizer(f):
    @functools.wraps(f)
    def wrapper(cost, parameters, fargs=[], fkwargs={}, givens={}, extra_outputs=[], **kwargs):
        updates = f(T.grad(cost, parameters), parameters, **kwargs)
        return theano.function(fargs, [cost, *extra_outputs],
                               updates=updates, givens=givens, **fkwargs)
    return wrapper

@optimizer
def sgd(gradients, parameters, learning_rate=0.01):
    return [(param, param - learning_rate * grad)
            for param, grad in zip(parameters, gradients)]

@optimizer
def adadelta(gradients, parameters, rho=0.95, epsilon=1e-6):
    updates = []
    for grad, param in zip(gradients, parameters):
        val = param.get_value(borrow=True)
        grad_accum = theano.shared(np.zeros(val.shape, dtype=val.dtype))
        update_accum = theano.shared(np.zeros(val.shape, dtype=val.dtype))

        new_grad_accum = rho * grad_accum + (1 - rho) * grad**2

        grad_rms = T.sqrt(new_grad_accum + epsilon)
        update_rms = T.sqrt(update_accum + epsilon)

        update = update_rms / grad_rms * grad

        new_update_accum = rho * update_accum + (1 - rho) * update**2

        updates.extend([(param, param - update),
                        (grad_accum, new_grad_accum),
                        (update_accum, new_update_accum)])
    return updates

@optimizer
def adam(gradients, parameters, learning_rate=0.001,
         beta1=0.9, beta2=0.999, epsilon=1e-8):
    updates = []
    beta1_decay = theano.shared(np.asarray(beta1, dtype=theano.config.floatX))
    beta2_decay = theano.shared(np.asarray(beta2, dtype=theano.config.floatX))

    a_t = learning_rate * T.sqrt(1 - beta2_decay) / (1 - beta1_decay)

    for grad, param in zip(gradients, parameters):
        val = param.get_value(borrow=True)
        grad_accum = theano.shared(np.zeros(val.shape, dtype=val.dtype))
        grad2_accum = theano.shared(np.zeros(val.shape, dtype=val.dtype))

        new_grad_accum = beta1 * grad_accum + (1 - beta1) * grad
        new_grad2_accum = beta2 * grad2_accum + (1 - beta2) * grad**2

        update = a_t * new_grad_accum / (T.sqrt(new_grad2_accum) + epsilon)

        updates.extend([(param, param - update),
                        (grad_accum, new_grad_accum),
                        (grad2_accum, new_grad2_accum)])

    updates.extend([(beta1_decay, beta1_decay * beta1),
                    (beta2_decay, beta2_decay * beta2)])
    return updates

@optimizer
def adamax(gradients, parameters, learning_rate=0.002,
           beta1=0.9, beta2=0.999, epsilon=1e-08):
    updates = []
    beta1_decay = theano.shared(np.asarray(beta1, dtype=theano.config.floatX))

    a_t = learning_rate / (1 - beta1_decay)

    for grad, param in zip(gradients, parameters):
        val = param.get_value(borrow=True)
        grad_accum = theano.shared(np.zeros(val.shape, dtype=val.dtype))
        grad2_accum = theano.shared(np.zeros(val.shape, dtype=val.dtype))

        new_grad_accum = beta1 * grad_accum + (1 - beta1) * grad
        new_grad2_accum = T.maximum(beta2 * grad2_accum, abs(grad))

        update = a_t * new_grad_accum / (new_grad2_accum + epsilon)

        updates.extend([(param, param - update),
                        (grad_accum, new_grad_accum),
                        (grad2_accum, new_grad2_accum)])

    updates.extend([(beta1_decay, beta1_decay * beta1)])
    return updates

class _Iterator:
    def __init__(self, iterator, inputs, shared, start, end, shuffle, chunk_size, **kwargs):
        self._iterator = iterator
        self._inputs = inputs
        self._input_size = self._inputs[0].shape[0]
        self._shared = shared
        self._shuffle = shuffle
        self._kwargs = kwargs
        self._start = start
        self._end = end
        self._chunk_size = chunk_size if chunk_size != -1 else self._input_size
        self._chunk_start = None

    def _update_chunk(self, i, j):
        i, j = min(i, self._input_size), min(j, self._input_size)

        if j - i > self._chunk_size:
            raise RuntimeError("Minibatch size ({}) larger than chunk size ({})".format(
                j - i, self._chunk_size))

        # check if new chunk needs to be loaded
        if self._chunk_start is not None and \
                (self._chunk_start <= i <= self._chunk_start + self._chunk_size and
                 self._chunk_start <= j <= self._chunk_start + self._chunk_size):
            return

        # load chunk starting at start of current minibatch
        self._chunk_start = i
        #print("Loading {}:{} into memory".format(
        #    self._chunk_start, self._chunk_start + self._chunk_size))
        for inp, sh in zip(self._inputs, self._shared):
            sh.set_value(inp[self._chunk_start:self._chunk_start + self._chunk_size])

    def __iter__(self):
        if self._shuffle:
            self._inputs = utils.shuffle_inputs(self._inputs)
            # flag for reload
            self._chunk_start = None

        for i, j in self._iterator(self._inputs, self._shared, **self._kwargs):
            self._update_chunk(i, j)
            self._start.set_value(i - self._chunk_start)
            self._end.set_value(j - self._chunk_start)
            yield

    def clear(self):
        # TODO
        pass

def batcher(f):
    @functools.wraps(f)
    def decorator(*inputs, shuffle=False, chunk_size=-1, **kwargs):
        shared = utils.input_to_shared(inputs)
        start, end = theano.shared(0, name='start'), theano.shared(0, name='end')
        iterator = _Iterator(f, inputs, shared, start, end, shuffle, chunk_size, **kwargs)
        sliced = [sh[start:end] for sh in shared]
        return (iterator, *sliced)
    return decorator

@batcher
def variedminibatches(inputs, shared, *, split_indices, batch_size=100):
    # get starting index of every `batch_size'th chunk,
    # starting at the `batch_size + 1'th chunk; compensate for
    # first chunk (0) not being included
    filtered = split_indices[batch_size - 1::batch_size]
    # add endpoints
    indices = np.concatenate(([0], filtered, [inputs[0].shape[0]]))
    yield from zip(indices[:-1], indices[1:])

@batcher
def minibatches(inputs, shared, batch_size=100):
    data_size = inputs[0].shape[0]
    for batch in range(0, data_size, batch_size):
        yield batch, batch + batch_size

@batcher
def unbatched(inputs, shared):
    yield 0, inputs[0].shape[0]

def _automatic_restart(iterable):
    while True:
        yield from iterable

def early_stopping(trainer, train_iterator, validator, valid_iterator,
                   patience=20, max_iters=None, tol=0, callback=None,
                   additional_iterators=[]):
    validator = utils.ensure_tuple(validator)
    valid_iterator = utils.ensure_tuple(valid_iterator)

    validation_error = np.inf
    patience_counter = 0

    additional_iterators = [_automatic_restart(i) for i in additional_iterators]
    for epoch in itertools.count(1):
        results = []
        for iteration, _ in enumerate(train_iterator, 1):
            for iterator in additional_iterators:
                next(iterator)

            results.append(trainer())
            #print(result)
            if np.isnan(results[-1]).any():
                print("got nan at epoch {}, iteration {}".format(epoch, iteration))
                return

        errs = [np.mean([val() for _ in valiter])
                for val, valiter in zip(validator, valid_iterator)]
        err = sum(errs)
        if callback is not None:
            callback(epoch, np.mean(results, axis=0), validation_error, *errs)

        if err < validation_error - tol:
            patience_counter = 0
        else:
            patience_counter += 1

        if err < validation_error:
            validation_error = err

        if patience_counter >= patience:
            break

if __name__ == '__main__':
    pass

