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

import click
import librosa.feature
import librosa.filters
import librosa.util
import numpy as np
import scipy.signal

import dataset

_shared_arguments = {
    'inset': click.Option(('-i', '--inset',), required=True),
    'outset': click.Option(('-o', '--outset',), required=True),
    'inplace': click.Option(('--inplace',), show_default=True, is_flag=True),
    'destructive': click.Option(('--destructive',), show_default=True, is_flag=True),
    'chunk_size': click.Option(('--chunk-size',), default=1000000, show_default=True),
    'hdf5file': click.Argument(('hdf5file',), type=dataset.HDF5TYPE),
}

def extractor_command(dtype=None):
    def decorator(comm):
        comm.params.append(_shared_arguments['inset'])
        comm.params.append(_shared_arguments['outset'])
        comm.params.append(_shared_arguments['hdf5file'])

        callback = comm.callback

        @functools.wraps(callback)
        def wrapper(hdf5file, inset, outset, **kwargs):
            try:
                inset = hdf5file[inset]
            except KeyError:
                raise click.BadOptionUsage("Dataset '{}' does not exist.".format(inset))

            if dtype is None:
                ddtype = inset.data.dtype
            else:
                ddtype = np.dtype(dtype)

            extractor, dims = callback(inset=inset, input_dims=inset.dims, **kwargs)
            outset = hdf5file.create_dataset(outset, dims, ddtype, overwrite=True)
            transform_dataset(extractor, inset, outset,
                              callback=lambda sec, level: level == 1 and print(sec.name))

        comm.callback = wrapper
        return comm
    return decorator


def inherit_flags(command, exclude=None):
    default_excludes = {param.human_readable_name
                        for param in _shared_arguments.values()}
    exclude = (default_excludes if exclude is None
               else {*exclude, *default_excludes})

    def decorator(f):
        f.params.extend(param for param in command.params
                        if param.human_readable_name not in exclude)
        return f
    return decorator


def feature_extractor(parent=None):
    def decorator(f):
        f._parent = parent

        @functools.wraps(f)
        def wrapper(input_dims=(), **kwargs):
            if f._parent is not None:
                parent, dims = f._parent(**kwargs)
                extractor, dims = f(dims=dims, input_dims=input_dims, **kwargs)
            else:
                parent = None
                extractor, dims = f(dims=input_dims, **kwargs)

            def iterator():
                data = yield
                while True:
                    if parent is not None:
                        data = parent.send(data)

                    if data is None:
                        data = yield None
                    else:
                        data = yield extractor(data)

            it = iterator()
            next(it)
            return it, dims
        return wrapper
    return decorator

def iterate_data(section, callback, chunk_size=1000):
    data = section.data
    for i in range(0, data.shape[0], chunk_size):
        callback(data[i:i+chunk_size])


def transform_dataset(extractor, inset, outset, callback=None):
    def _build_sections(insec, outsec, level):
        for sec in insec:
            if callback is not None:
                callback(sec, level)

            sectiondata = sec.sectiondata
            data = extractor.send(sectiondata[:] if sectiondata is not None else None)
            metadata = sec.metadata
            newsec = outsec.create_section(name=sec.name, data=data, metadata=metadata)
            _build_sections(sec, newsec, level + 1)

    outset.metadata = inset.metadata
    _build_sections(inset, outset, 1)


@feature_extractor()
def frame(*, dims, sample_rate=None, window_length=25, window_shift=10, **kwargs):
    if sample_rate is not None:
        window_length = int(sample_rate / 1000 * window_length)
        window_shift = int(sample_rate / 1000 * window_shift)

    def extractor(data):
        if data.shape[0] < window_length:
            return None

        try:
            indices = librosa.util.frame(
                np.arange(data.shape[0]), window_length, window_shift).T
            return data[indices]
        except librosa.ParameterError:
            return None

    return extractor, (window_length, *dims)

@feature_extractor(parent=frame)
def spectrum(*, dims, fft_bins=512, window_function='hamming', **kwargs):
    if window_function == 'hamming':
        window = scipy.signal.hamming(dims[0], sym=False)
    elif window_function == 'hann':
        window = scipy.signal.hann(dims[0], sym=False)
    else:
        window = None

    def extractor(data):
        return abs(np.fft.fft(data, fft_bins))**2

    return extractor, (fft_bins,)

@feature_extractor(parent=spectrum)
def fbank(*, dims, sample_rate, filters=40, low_frequency=0.0,
          high_frequency=None, **kwargs):
    filterbank = librosa.filters.mel(sample_rate, dims[0], filters,
                                     low_frequency, high_frequency).T

    def extractor(data):
        return data[:,:dims[0] // 2 + 1] @ filterbank

    return extractor, (filters,)

@feature_extractor(parent=fbank)
def mfcc(*, mfccs=13, first_order=False, second_order=False, **kwargs):
    def deltas(x):
        x = np.pad(x, ((2, 2), (0, 0)), 'edge')
        return ((x[2:] - x[:-2])[1:-1] + 2*(x[4:] - x[:-4])) / 10

    def extractor(data):
        coeffs = [librosa.feature.mfcc(n_mfcc=mfccs, S=np.log(data).T).T]

        if first_order or second_order:
            d = deltas(coeffs[0])
            if first_order:
                coeffs.append(d)
            if second_order:
                coeffs.append(deltas(d))

        return np.hstack(coeffs)

    return extractor, (mfccs * (1 + first_order + second_order),)


def log():
    pass

def normalize():
    pass

def randomize():
    pass

def lpc(samples, n):
    alphas = []
    for sample in np.atleast_2d(samples):
        corr = np.correlate(sample, sample, mode='full')[len(sample) - 1:]
        alpha = scipy.linalg.solve_toeplitz(corr[:n], corr[1:n+1])
        alphas.append(alpha)
    return np.array(alphas)

def envelope(samples, fs, coeffs=18, resolution=512,
             max_freq=5000, min_freq=100):
    alpha = lpc(samples, coeffs)
    steps = np.linspace(min_freq, max_freq, resolution)
    exponents = np.outer(1j * 2 * np.pi * steps / fs,
                         -np.arange(1, coeffs + 1))
    spec = 1 / (1 - (alpha * np.exp(exponents)).sum(axis=1))
    power = abs(spec) ** 2

    return power, steps

def formants(samples, fs, num_formants=3, return_spec=False, **kwargs):
    power, steps = envelope(samples, fs, **kwargs)

    # find values larger that both neighbours
    local_maxima = (power[:-1] > power[1:])[1:] & (power[1:] > power[:-1])[:-1]
    indices, = np.where(local_maxima)
    formants = steps[indices + 1][:num_formants]

    if return_spec:
        return power, formants
    else:
        return formants


@click.group()
def main():
    pass

@extractor_command()
@click.option('--window-shift', default=10, show_default=True)
@click.option('--window-length', default=25, show_default=True)
@click.option('--sample-rate', type=int)
@main.command('frame')
def frame_comm(**kwargs):
    return frame(**kwargs)

@extractor_command(dtype='f4')
@click.option('--window-function', default='hamming', show_default=True,
              type=click.Choice(['none', 'hamming', 'hann']))
@click.option('--fft-bins', default=512, show_default=True)
@inherit_flags(frame_comm)
@main.command('spectrum')
def spectrum_comm(**kwargs):
    return spectrum(**kwargs)

@extractor_command(dtype='f4')
@click.option('--filters', default=40, show_default=True)
@click.option('--low-frequency', default=0.0, show_default=True)
@click.option('--high-frequency', default=None, type=float)
@inherit_flags(spectrum_comm, exclude={'sample_rate'})
@click.option('--sample-rate', type=int, required=True)
@main.command('fbank')
def fbank_comm(**kwargs):
    return fbank(**kwargs)

@extractor_command(dtype='f4')
@click.option('--second-order', is_flag=True, show_default=True)
@click.option('--first-order', is_flag=True, show_default=True)
@click.option('--mfccs', default=13, show_default=True)
@inherit_flags(fbank_comm)
@main.command('mfcc')
def mfcc_comm(**kwargs):
    return mfcc(**kwargs)

if __name__ == '__main__':
    main()
