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
import pathlib
import pickle
import shutil
import itertools

import numpy as np
import theano
import theano.tensor as T
import sklearn.model_selection
import sklearn.utils

import click

import dataset
import features

import nn.activations
import nn.trainers
import nn.layers
import nn.init

from _train import *

def entropy(dist):
    return -(dist * T.log2(dist)).sum(axis=-1)

def norm_entropy(dist):
    return entropy(dist) / T.log2(dist.shape[-1])

def kullback_leibler(dist1, dist2):
    logged = T.log2(dist1 / dist2)
    # let 0 * log(0) -> 0
    #logged = T.set_subtensor(logged[T.eq(dist1, 0).nonzero()], 0)
    return (dist1 * logged).sum(axis=1)

# KL divergence inappropriate due to being unbounded
def jensen_shannon(dist1, dist2):
    average = (dist1 + dist2) / 2
    #return entropy(average) - (entropy(dist1) + entropy(dist2)) / 2
    return (kullback_leibler(dist1, average) +
            kullback_leibler(dist2, average)) / 2

def js_loss(x1, x2, types, epsilon=1e-6):
    # avoid softmax saturation
    x1, x2 = x1 + epsilon, x2 + epsilon
    # take the square root of the JS divergence (is a metric)
    # NOTE: avoid negative values caused by numerical errors
    # by adding a small value before taking the square root
    js = T.sqrt(jensen_shannon(x1, x2) + epsilon)
    js_cost = T.switch(types, js, 1 - js)
    return js_cost

def entropy_loss(x1, x2):
    return (norm_entropy(x1).mean() + norm_entropy(x2).mean()) / 2

def calculate_spread(V):
    spread = V.sum(axis=0) / V.shape[0]
    return norm_entropy(spread)

def cos(a, b):
    return (a * b).sum(axis=1) / (a.norm(2, axis=1) * b.norm(2, axis=1))

def cos_loss(x1, x2, types, angular=False):
    cossim = cos(x1, x2)
    if angular:
        cossim = 1 - 2 * T.arccos(cossim - 1e-6) / np.pi
    cos_cost = T.switch(types, 1 - cossim, cossim)
    return cos_cost

def coscos2(x1, x2, types):
    cossim = cos(x1, x2, types)
    coscos_cost = T.switch(types, (1 - cossim)/2, cossim**2)
    return coscos_cost

def build_deep(input, outputs, loss, reconstruction_penalty=0,
               ids=None, num_speakers=-1, layer_size=500):
    out_activation = T.nnet.sigmoid if loss == 'coscos' else T.nnet.softmax
    mslayer = nn.layers.FeedForwardLayer(input, layer_size, activation=T.nnet.sigmoid,
                                         W=nn.init.glorot(4))
    hidden = nn.layers.FeedForwardLayer(mslayer, layer_size, activation=T.nnet.sigmoid,
                                        W=nn.init.glorot(4))
    outlayer = nn.layers.FeedForwardLayer(hidden, outputs, activation=out_activation,
                                          W=nn.init.glorot(4))

    return mslayer, outlayer

def build_shallow(input, outputs, network_type, ids=None, num_speakers=-1):
    if ids is None:
        activation = (nn.activations.normalize
                      if network_type == 'shallow'
                      else nn.activations.linear)
        normalize_rows = network_type == 'shallow-rownorm'
        mslayer = PositiveWeightLayer(input, outputs, activation=activation,
                                      normalize_rows=normalize_rows)
    else:
        mslayer = MultiSpeakerLayer(input, ids, outputs,
                                    num_speakers, b=None, force_positive_weights=True,
                                    activation=nn.activations.normalize)
    return mslayer, mslayer

def build_rbf(input, outputs, loss):
    mslayer = GMMLayer(input, outputs)
    return mslayer, mslayer


@click.group()
def main():
    pass

@main.command()
@click.argument('hdf5file', type=dataset.Hdf5Type('r'))
@click.option('-i', '--inset', required=True)
@click.option('--dims', type=int, default=64, show_default=True)
@click.option('--layersize', type=int, default=800, show_default=True)
@click.option('-m', '--model-output', type=click.Path(dir_okay=False, writable=True))
@click.option('--save-errors', type=click.File('wb'))
@click.option('--trainer', type=click.Choice(['adam', 'adamax']),
              default='adam', show_default=True)
def train_autoencoder(hdf5file, inset, dims, layersize, model_output, trainer, save_errors):
    data = hdf5file[inset].data[:]
    train_data, valid_data = sklearn.model_selection.train_test_split(data, test_size=0.3)
    train_iter, train_shared = nn.trainers.minibatches(train_data, batch_size=1000, chunk_size=200000)
    valid_iter, valid_shared = nn.trainers.minibatches(valid_data, batch_size=1000, chunk_size=200000)

    in_size = data.shape[-1]

    input = T.matrix()
    print("Training {}/{}".format(layersize, dims))
    inputlayer = nn.layers.InputLayer((in_size,))
    hidden_enc1 = nn.layers.FeedForwardLayer(inputlayer, layersize, activation=T.nnet.sigmoid)
    hidden_enc2 = nn.layers.FeedForwardLayer(hidden_enc1, layersize, activation=T.nnet.sigmoid)
    hidden_enc3 = nn.layers.FeedForwardLayer(hidden_enc2, layersize, activation=T.nnet.sigmoid)
    replayer = nn.layers.FeedForwardLayer(hidden_enc3, dims, activation=T.nnet.sigmoid)
    hidden_dec1 = nn.layers.FeedForwardLayer(replayer, layersize, activation=T.nnet.sigmoid)
    hidden_dec2 = nn.layers.FeedForwardLayer(hidden_dec1, layersize, activation=T.nnet.sigmoid)
    hidden_dec3 = nn.layers.FeedForwardLayer(hidden_dec2, layersize, activation=T.nnet.sigmoid)
    outlayer = nn.layers.FeedForwardLayer(hidden_dec3, in_size, activation=nn.activations.linear)

    output = outlayer.output({inputlayer: input})
    error = T.mean((input - output).norm(2, axis=1)**2)

    train_func = nn.trainers.adam if trainer == 'adam' else nn.trainers.adamax

    trainer = train_func(error, list(outlayer.all_parameters()),
                        givens={input: train_shared})
    train_validator = theano.function([], error, givens={input: train_shared})
    validator = theano.function([], error, givens={input: valid_shared})

    def callback(epoch, training_error, best_error, current_error):
        train_error = np.mean([train_validator() for _ in train_iter])
        print(epoch, training_error, best_error, (train_error, current_error))

        if model_output is not None and current_error < best_error:
            with open(model_output, 'wb') as f:
                pickle.dump((None, None, inputlayer, replayer), f)

    nn.trainers.early_stopping(trainer, train_iter, validator, valid_iter,
                                tol=0.0001, patience=15, callback=callback)
    train_error = np.mean([train_validator() for _ in train_iter])
    valid_error = np.mean([validator() for _ in valid_iter])

    if save_errors is not None:
        pickle.dump((train_error, valid_error), save_errors)


def prepare_data(data, batch_size=1000, chunk_size=-1):
    types = np.array([1] * data['same'].data.shape[0] +
                     [0] * data['diff'].data.shape[0])

    pairs, types = sklearn.utils.shuffle(data.data[:], types)
    iterator, pairs_shared, types_shared = nn.trainers.minibatches(
        pairs, types, shuffle=False, batch_size=batch_size, chunk_size=chunk_size)

    return iterator, pairs_shared, types_shared

def prepare_model(input_size, network_type, outputs=64, loss='js', layer_size=500):
    inputlayer = nn.layers.InputLayer((input_size,))

    if network_type.startswith('deep'):
        mslayer, outlayer = build_deep(inputlayer, outputs, loss, layer_size=layer_size)
    elif network_type.startswith('shallow'):
        mslayer, outlayer = build_shallow(inputlayer, outputs,
                                          network_type=network_type)

    return inputlayer, mslayer, outlayer

def prepare_loss(inputlayer, outlayer, pairs, types, loss_function,
                 entropy_penalty=0, V=None, lamb=-1, train_pass=False):
    # reshape to 2d before sending through the network,
    # after which the original shape is recovered
    output = outlayer.output(
        {inputlayer: pairs.reshape((-1, pairs.shape[-1]))},
        train_pass=train_pass).reshape((pairs.shape[0], 2, -1))

    x1, x2 = output[:,0], output[:,1]
    cost = loss_function(x1, x2, types)
    same_loss = cost[T.nonzero(types)].mean()
    diff_loss = cost[T.nonzero(1 - types)].mean()

    if lamb >= 0:
        cost = 1 / (lamb + 1) * same_loss + lamb / (lamb + 1) * diff_loss
    else:
        cost = cost.mean()

    ent = entropy_loss(x1, x2)
    total_cost = cost + entropy_penalty * ent

    if V is not None:
        return total_cost, cost, same_loss, diff_loss, ent, calculate_spread(V)
    else:
        return total_cost, cost, same_loss, diff_loss, ent

@main.command()
@click.argument('hdf5file', type=dataset.Hdf5Type('r'))
@click.option('-i', '--inset', required=True)
@click.option('--outputs', type=int, default=64, show_default=True)
@click.option('--entropy-penalty', type=float, default=0, show_default=True)
@click.option('--diff-weight', type=float, default=-1, show_default=True)
@click.option('--loss', type=click.Choice(['js', 'coscos', 'cos', 'angular']), default='js',
              show_default=True)
@click.option('--network-type', type=click.Choice(['shallow', 'shallow-rownorm', 'deep']),
              default='shallow', show_default=True)
@click.option('--layer-size', type=int, default=500, show_default=True)
@click.option('-m', '--model-output', type=click.Path(dir_okay=False, writable=True))
@click.option('--tolerance', type=float, default=0.0001)
@click.option('--patience', type=int, default=15)
@click.option('--save-errors', type=click.File('wb'))
def train_fixedbatch(hdf5file, inset, outputs, entropy_penalty, loss,
                     network_type, model_output, tolerance, patience,
                     save_errors, diff_weight, layer_size):
    # ---- prepare data
    data = hdf5file[inset]
    feature_size = data.data.shape[-1]

    train_iterator, *train_shared = prepare_data(
        data['train'], batch_size=1000, chunk_size=100000)
    valid_iterator, *valid_shared = prepare_data(
        data['valid'], batch_size=100000, chunk_size=100000)

    inputlayer, mslayer, outlayer = prepare_model(feature_size, network_type, outputs=outputs,
                                                  loss=loss, layer_size=layer_size)

    # ---- prepare loss function
    if loss == 'coscos':
        loss_function = coscos2
    elif loss == 'cos':
        loss_function = cos_loss
    elif loss == 'angular':
        loss_function = functools.partial(cos_loss, angular=True)
    elif loss == 'js':
        loss_function = js_loss

    V = mslayer._V if network_type == 'shallow-rownorm' else None

    train_loss, *train_losses = prepare_loss(
        inputlayer, outlayer, *train_shared, loss_function,
        entropy_penalty=entropy_penalty, lamb=diff_weight, V=V, train_pass=True)
    valid_loss, *valid_losses = prepare_loss(
        inputlayer, outlayer, *valid_shared, loss_function,
        entropy_penalty=entropy_penalty, lamb=diff_weight, V=V, train_pass=False)

    all_train_losses = theano.function([], [train_loss, *train_losses])
    all_valid_losses = theano.function([], [valid_loss, *valid_losses])

    calc_errors = lambda f, it: np.array([f() for _ in it]).mean(axis=0)
    train_errors = [calc_errors(all_train_losses, train_iterator)]
    valid_errors = [calc_errors(all_valid_losses, valid_iterator)]
    def callback(epoch, training_error, best_error, *current_error):
        tot_err = sum(current_error)

        if model_output is not None and tot_err < best_error:
            with open(model_output, 'wb') as f:
                pickle.dump((inputlayer, None, mslayer, outlayer), f)

        print(epoch, training_error, tot_err, current_error)

        tr_errors = calc_errors(all_train_losses, train_iterator)
        val_errors = calc_errors(all_valid_losses, valid_iterator)
        print(tr_errors)
        print(val_errors)
        train_errors.append(tr_errors)
        valid_errors.append(val_errors)

        if network_type == 'shallow-rownorm':
            W = abs(mslayer._W.get_value())
            W = W / W.sum(axis=1)[:,np.newaxis]
            active = W > 0.1
            active_clusters = active.sum(axis=0)
            print("Active clusters: {} ({}); mean mapping strength: {}".format(
                (active_clusters > 0).sum(), active_clusters[active_clusters > 0],
                W.max(axis=1).mean()))

    print("Training", network_type, "with", loss, "and", outputs, "outputs")
    trainer = nn.trainers.adamax(train_loss, list(outlayer.all_parameters()))
    validator = theano.function([], valid_loss)
    nn.trainers.early_stopping(trainer, train_iterator, validator, valid_iterator,
                               tol=tolerance, patience=patience, callback=callback)

    if save_errors is not None:
        pickle.dump((train_errors, valid_errors), save_errors)

@main.command()
@click.argument('model', type=click.File('rb'))
@click.option('-m', '--model-output', type=click.Path(dir_okay=False, writable=True))
@click.option('--discretize-output', is_flag=True, show_default=True)
def discretize_model(model, model_output, discretize_output):
    _, _, mslayer, _ = pickle.load(model)
    V = mslayer._V.eval()
    W = np.zeros(V.shape, dtype=theano.config.floatX)
    W[np.arange(W.shape[0]),V.argmax(axis=1)] = 1

    inputlayer = nn.layers.InputLayer((W.shape[0],))
    outputlayer = nn.layers.FeedForwardLayer(inputlayer, W.shape[1],
                                             activation=(discretize if discretize_output
                                                         else nn.activations.linear))
    outputlayer._W.set_value(W[np.newaxis])

    with open(model_output, 'wb') as f:
        pickle.dump((inputlayer, None, outputlayer, outputlayer), f)

@main.command()
@click.argument('hdf5file', type=dataset.Hdf5Type('r'))
@click.option('-i', '--inset', required=True)
@click.option('--save-errors', type=click.File('wb'), required=True)
def grid_search(hdf5file, inset, save_errors):
    data = hdf5file[inset]

    train_iterator, *train_shared = prepare_data(data['train'], batch_size=1000)
    valid_iterator, *valid_shared = prepare_data(data['valid'], batch_size=100000)

    all_errors = []
    def run_once(outputs, entropy_penalty):
        inputlayer, mslayer, outlayer = prepare_model(data.data.shape[-1], 'shallow-rownorm',
                                                      outputs=outputs)

        loss_function = lambda *x: js_loss(*x, entropy_penalty=entropy_penalty, V=mslayer._V)

        train_loss, *train_losses = prepare_loss(
            inputlayer, outlayer, *train_shared, loss_function, train_pass=True)
        valid_loss, *valid_losses = prepare_loss(
            inputlayer, outlayer, *valid_shared, loss_function)

        full_trainer = theano.function([], [train_loss, *train_losses])
        full_validator = theano.function([], [valid_loss, *valid_losses])

        calc_errors = lambda f, it: np.array([f() for _ in it]).mean(axis=0)
        train_errors = [calc_errors(full_trainer, train_iterator)]
        valid_errors = [calc_errors(full_validator, valid_iterator)]
        def callback(epoch, best_error, current_error, training_error):
            print("Entropy penalty {}, outputs {}, epoch {}; {}, {}".format(
                entropy_penalty, outputs, epoch, current_error, training_error))
            train_errors.append(calc_errors(full_trainer, train_iterator))
            valid_errors.append(calc_errors(full_validator, valid_iterator))
            print(train_errors[-1])
            print(valid_errors[-1])

        trainer = nn.trainers.adamax(train_loss, list(outlayer.all_parameters()))
        validator = theano.function([], valid_loss)
        nn.trainers.early_stopping(trainer, train_iterator, validator, valid_iterator,
                                   tol=0.0001, patience=15, callback=callback)
        all_errors.append(((outputs, entropy_penalty), np.array(train_errors),
                           np.array(valid_errors)))

    for entropy_penalty in np.arange(0, 20) / 20:
        run_once(outputs=64, entropy_penalty=entropy_penalty)

    pickle.dump(all_errors, save_errors)


def write_section(f, sec, data, offset=0, window_shift=10, window_length=25):
    window_shift, window_length = window_shift / 1000, window_length / 1000
    start, end = sec.metadata
    middlepoints = (start + offset) + (np.arange(data.shape[0]) * window_shift +
                                       window_length / 2)

    output = np.hstack((middlepoints[:,np.newaxis], data))
    np.savetxt(f, output, fmt='%s')

def process_speaker(outdir, speaker, callback, **kwargs):
    for recording in speaker:
        with (outdir / (recording.name + '.fea')).open('wb') as f:
            for section in recording:
                data = callback(section.data[:])
                write_section(f, section, data, **kwargs)

def get_outdir(path):
    outdir = pathlib.Path(path)
    if outdir.exists():
        shutil.rmtree(str(outdir))
    outdir.mkdir()

    return outdir

def open_model(model):
    inputlayer, idlayer, mslayer, outputlayer = pickle.load(model)
    output = outputlayer.output()
    inputs = list(mslayer.all_inputs())
    get_output = theano.function(inputs, output)
    return get_output, len(inputs), outputlayer.output_shape

@main.command()
@click.argument('hdf5file', type=dataset.Hdf5Type('a'))
@click.option('-i', '--inset', required=True)
@click.option('--idset')
@click.option('-o', '--outdir', type=click.Path(file_okay=False, writable=True),
              required=True)
@click.option('--write-to-dataset', is_flag=True, show_default=True)
@click.option('--window-shift', default=10, show_default=True)
@click.option('--window-length', default=25, show_default=True)
@click.option('-m', '--model', type=click.File('rb'))
def process(hdf5file, inset, idset, outdir, window_shift,
            window_length, model, write_to_dataset):
    datset = hdf5file[inset]

    if model is not None:
        get_output, num_inputs, outshape = open_model(model)

        if num_inputs > 1 and idset is None:
            raise click.BadOptionUsage(
                "--idset must be provided for speaker-dependent models.")
    else:
        get_output = lambda x: x
        outshape = datset.data.shape[1:]

    if write_to_dataset:
        @features.feature_extractor()
        def transform(**kwargs):
            return get_output, outshape

        def callback(sec, level):
            if level == 1:
                print(sec.name)

        extractor, dims = transform()
        outset = hdf5file.create_dataset(outdir, dims, 'f4', overwrite=True)
        features.transform_dataset(extractor, datset, outset, callback=callback)
    else:
        outdir = get_outdir(outdir)

        def callback(data):
            if idset is None:
                out = get_output(data)
            else:
                out = get_output(
                    data, np.repeat(speaker_id, data.shape[0]).astype('float32'))
            return out

        for speaker in datset:
            print(speaker.name)
            if idset is not None:
                speaker_id = speakers[speaker.name]

            process_speaker(outdir, speaker, callback,
                            window_shift=window_shift, window_length=window_length)


@main.command()
@click.argument('hdf5file', type=dataset.Hdf5Type('r'))
@click.option('-i', '--inset', required=True)
@click.option('-o', '--outdir', type=click.Path(file_okay=False, writable=True),
              required=True)
@click.option('--window-shift', default=10, show_default=True)
@click.option('--window-length', default=25, show_default=True)
def dump_dataset(hdf5file, inset, outdir, window_shift, window_length):
    outdir = get_outdir(outdir)
    datset = hdf5file[inset]

    for speaker in datset:
        print(speaker.name)
        process_speaker(outdir, speaker, lambda data: data,
                        window_shift=window_shift, window_length=window_length)


if __name__ == '__main__':
    main()
