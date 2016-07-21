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
    'outset': click.Option(('-o', '--outset',)),
    'inplace': click.Option(('--inplace',), show_default=True, is_flag=True),
    'destructive': click.Option(('--destructive',), show_default=True, is_flag=True),
    'chunk_size': click.Option(('--chunk-size',), default=1000000, show_default=True),
    'hdf5file': click.Argument(('hdf5file',), type=dataset.HDF5TYPE),
}

def feature_extractor(dtype=None, dims=None, inherit=None, initializer=None,
                      inplace=False, destructive=False, extra_pass=None):
    inplace_possible = inplace
    is_destructive = destructive

    if not inplace and dims is None:
        raise ValueError("The dimensions must be supplied when in-place not supported.")

    def decorator(comm):
        if not isinstance(comm, click.Command):
            comm = click.command()(comm)

        if inherit is not None:
            comm.params.extend(param for param in inherit.command.params
                               if param not in _shared_arguments.values())

        comm.params.append(_shared_arguments['inset'])
        comm.params.append(_shared_arguments['outset'])

        if inplace_possible:
            comm.params.append(_shared_arguments['inplace'])
            if is_destructive:
                comm.params.append(_shared_arguments['destructive'])
            comm.params.append(_shared_arguments['chunk_size'])

        comm.params.append(_shared_arguments['hdf5file'])

        callback = comm.callback

        callback._initialized_object = None
        callback._initializer = initializer
        callback._parent = inherit.primitive if inherit is not None else None
        callback._dims = dims

        @functools.wraps(callback)
        def wrapper(hdf5file, inset, outset, inplace=False,
                    destructive=False, chunk_size=-1, **kwargs):
            try:
                inset = hdf5file[inset]
            except KeyError:
                raise click.BadOptionUsage("Dataset '{}' does not exist.".format(inset))

            if dtype is None:
                ddtype = inset.data.dtype
            else:
                ddtype = np.dtype(dtype)

            if outset is None and not inplace:
                raise click.BadOptionUsage("Please specify an output dataset using '--outset'"
                                           " (or specify '--inplace' if applicable).")
            if outset is not None and inplace:
                raise click.BadOptionUsage("Cannot use both '--outset' and '--inplace'.")
            if inplace and is_destructive and not destructive:
                raise click.BadOptionUsage("Attempted to run a destructive command "
                                           "in-place without specifying '--destructive'.")
            if inplace and inset.data.dtype != ddtype:
                raise click.BadOptionUsage(("Input dataset type '{}' does not match "
                                            "expected type '{}'; cannot "
                                            "process in-place.".format(
                                                inset.data.dtype, ddtype)))

            if not inplace_possible or not inplace:
                try:
                    del hdf5file[outset]
                except KeyError:
                    pass

            args = (inset.dims,)

            chain = []
            cur = callback
            while cur is not None:
                chain.append(cur)
                cur = cur._parent
            chain = chain[::-1]

            curdim = inset.dims
            for cb in chain:
                args = (curdim,)
                if cb._initializer is not None:
                    cb._initialized_object = cb._initializer(*args, **kwargs)
                    args += (cb._initialized_object,)

                if cb._dims is not None:
                    if callable(cb._dims):
                        curdim = cb._dims(*args, **kwargs)
                    else:
                        curdim = cb._dims

            def run_chain(data, *extra_args):
                for cb in chain:
                    if data is None:
                        break

                    args = (data,)
                    if cb._initializer is not None:
                        args += (cb._initialized_object,)
                    args += extra_args

                    data = cb(*args, **kwargs)

                return data

            if inplace_possible:
                args = args[1:]

                if inplace:
                    dataset = inset
                    if is_destructive:
                        dataset.clear_sections()
                else:
                    if is_destructive:
                        hdf5file.copy(inset, outset, keep_sections=False)
                    else:
                        hdf5file.copy(inset, outset)

                    dataset = hdf5file[outset]

                passes = extra_pass + 1
                for i in range(passes):
                    for chunk in range(0, dataset.data.shape[0], chunk_size):
                        print(chunk)

                        slice_obj = slice(chunk, chunk + chunk_size)
                        extra_args = (i,) if passes > 1 else ()
                        retval = run_chain(dataset.data[slice_obj], *extra_args)
                        if i == passes - 1:
                            dataset.data[slice_obj] = retval
            else:
                def _build_sections(insec, outsec, outer=True):
                    for sec in insec:
                        if outer:
                            print(sec.name)

                        data = run_chain(sec.sectiondata)
                        metadata = sec.metadata
                        newsec = outsec.create_section(
                            name=sec.name, data=data, metadata=metadata)
                        _build_sections(sec, newsec, outer=False)

                _build_sections(inset, hdf5file.create_dataset(
                    outset, curdim, ddtype))

        comm.callback = wrapper
        wrapper.primitive = callback
        wrapper.command = comm
        return wrapper
    return decorator

@click.group()
def main():
    pass

def _frame_initializer(dims, sample_rate, window_length, window_shift, **kwargs):
    window_length = int(sample_rate / 1000 * window_length)
    window_shift = int(sample_rate / 1000 * window_shift)
    return window_length, window_shift

@feature_extractor(dims=lambda dims, init_obj, **kwargs: (init_obj[0], *dims),
                   initializer=_frame_initializer)
@click.option('--window-shift', default=10, show_default=True)
@click.option('--window-length', default=25, show_default=True)
@click.option('--sample-rate', type=int, required=True)
@main.command()
def frame(data, _init_obj, sample_rate, window_length, window_shift, **kwargs):
    window_length, window_shift = _init_obj
    # avoid bug in librosa.util.frame (issue #385)
    if data.shape[0] < window_length:
        return None

    try:
        indices = librosa.util.frame(
            np.arange(data.shape[0]), window_length, window_shift).T
        return data[indices]
    except librosa.ParameterError:
        return None

def _spectrum_initializer(dims, window_function, **kwargs):
    if window_function == 'hamming':
        return scipy.signal.hamming(dims[0], sym=False)
    elif window_function == 'hann':
        return scipy.signal.hann(dims[0], sym=False)
    else:
        return None

@feature_extractor(dims=lambda dims, _, fft_bins, **kwargs: (fft_bins,),
                   dtype='f4', initializer=_spectrum_initializer,
                   inherit=frame)
@click.option('--window-function', default='hamming', show_default=True,
              type=click.Choice(['none', 'hamming', 'hann']))
@click.option('--fft-bins', default=512, show_default=True)
@main.command()
def spectrum(data, _window, fft_bins, window_function, **kwargs):
    if _window is not None:
        data = data * _window
    return abs(np.fft.fft(data, fft_bins))**2

def _fbank_initializer(dims, high_frequency, low_frequency, filters,
                       fft_bins, sample_rate, **kwargs):
    return librosa.filters.mel(sample_rate, fft_bins, filters,
                               low_frequency, high_frequency).T

@feature_extractor(dims=lambda dims, _, filters, **kwargs: (filters,),
                   dtype='f4', initializer=_fbank_initializer,
                   inherit=spectrum)
@click.option('--filters', default=40, show_default=True)
@click.option('--low-frequency', default=0.0, show_default=True)
@click.option('--high-frequency', default=None, type=float)
@main.command()
def fbank(data, _filterbank, fft_bins, **kwargs):
    return data[:,:fft_bins // 2 + 1] @ _filterbank

@feature_extractor(dims=lambda dims, mfccs, first_order, second_order, **kwargs:
                       (mfccs * (1 + first_order + second_order),),
                   dtype='f4', inherit=fbank)
@click.option('--second-order', is_flag=True, show_default=True)
@click.option('--first-order', is_flag=True, show_default=True)
@click.option('--mfccs', default=13, show_default=True)
@main.command()
def mfcc(data, mfccs, first_order, second_order, **kwargs):
    coeffs = [librosa.feature.mfcc(n_mfcc=mfccs, S=np.log(data).T).T]

    def deltas(x):
        x = np.pad(x, ((2, 2), (0, 0)), 'edge')
        return ((x[2:] - x[:-2])[1:-1] + 2*(x[4:] - x[:-4])) / 10

    if first_order or second_order:
        d = deltas(coeffs[0])
        if first_order:
            coeffs.append(d)
        if second_order:
            coeffs.append(deltas(d))

    return np.hstack(coeffs)

def log():
    pass

def normalize():
    pass

def randomize():
    pass

if __name__ == '__main__':
    main()
