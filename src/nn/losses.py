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

import theano.tensor as T

def cross_entropy(output, target):
    target = T.cast(target, 'int32')
    return -T.mean(T.log(output[T.arange(target.shape[0]),target]))

def cross_entropy_bernoulli(output, target):
    return -(T.log(output) * target + T.log(1 - output) * (1 - target)).mean()

def cross_entropy_categorical(output, target):
    return -(T.log(output) * target).sum(axis=1).mean()

def mean_square_error(output, target):
    return T.mean(T.sqrt(((output - target)**2).sum(axis=1)))
