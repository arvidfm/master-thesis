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

import math
import sys
sys.path.append("../../src")

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

import click
import librosa.filters

import features

def write_data(fname, data, header):
    if not isinstance(data, np.ndarray):
        data = np.vstack(data).T
    header, fmt = zip(*header)
    header = " ".join(header)
    np.savetxt(fname, data, fmt=fmt, header=header, comments='')

@click.group()
def main():
    pass

@main.command()
@click.option('--plot', is_flag=True)
def spectrum(plot):
    fs, data = scipy.io.wavfile.read('bed.wav')
    timesteps = np.arange(len(data)) / fs

    # 25 ms window starting at 65 ms
    window_start = int(65 / 1000 * fs)
    window_end = window_start + int(25 / 1000 * fs)

    print("Running STFT on window starting at {} and ending at {}".format(
        window_start, window_end))
    window = data[window_start:window_end]
    spec = (abs(np.fft.fft(data[window_start:window_end], 512))**2)[:256]
    frequencies = np.arange(256) / 512 * fs

    norm_data = window / np.iinfo(data.dtype).max
    envelope, steps = features.envelope(norm_data, fs, min_freq=0, max_freq=255 / 512 * fs,
                                        resolution=256)

    envelope *= spec.sum() / envelope.sum()
    if plot:
        plt.plot(timesteps, data)
        plt.show()
        plt.plot(frequencies, np.log(spec))
        plt.plot(frequencies, np.log(envelope))
        plt.show()

    write_data('samples.txt', (timesteps, data), [('time', '%f'), ('samples', '%d')])
    write_data('spectrum.txt', (frequencies, np.log(spec), np.log(envelope)),
               [('frequency', '%s'), ('energy', '%s'), ('envelope', '%s')])

@main.command()
@click.option('--plot', is_flag=True)
def melscale(plot):
    filterbanks = librosa.filters.mel(16000, 2048, n_mels=20, htk=True)
    filterfreqs = np.arange(filterbanks.shape[-1]) / (2048) * 16000

    # remove all zeros that have neighbouring zeros on both sides
    padded = np.pad(filterbanks, [(0, 0), (1, 1)], mode='edge')
    indices = (padded[:,2:] == 0) & (padded[:,1:-1] == 0) & (padded[:,:-2] == 0)
    filterbanks[indices] = math.nan

    if plot:
        for filt in filterbanks:
            plt.plot(filterfreqs, filt)
        plt.show()

    write_data('filterbanks.txt', (filterfreqs, *filterbanks),
               [('frequency', '%f'), *[('f{}'.format(i+1), '%f') for i in range(20)]])

@main.command()
@click.option('--plot', is_flag=True)
def circle_data(plot):
    class_a = np.random.randn(100, 2) / 4
    x = np.linspace(0, np.pi*2, 100)
    class_b = np.vstack((np.sin(x), np.cos(x))).T * 2
    class_b = class_b + np.random.normal(scale=0.2, size=class_b.shape)

    if plot:
        x, y = np.mgrid[-5:5:0.1, -5:5:0.1]
        plt.imshow(np.exp(-x**2) + np.exp(-y**2) - 1.1, extent=(-5, 5, -5, 5))
        plt.plot(*class_a.T, 'o')
        plt.plot(*class_b.T, 'o')
        plt.show()

    write_data('circle_data.txt', (class_a.T, class_b.T),
               [('ax', '%s'), ('ay', '%s'), ('bx', '%s'), ('by', '%s')])

if __name__ == '__main__':
    main()
