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

import collections
import pathlib
import re
import string
import warnings

import numpy as np
import scipy.io.wavfile
import librosa.core
import librosa.feature
import click

import dataset

# inspired by audiolabel:
# https://github.com/rsprouse/audiolabel/blob/master/audiolabel.py#L994
def parse_esps(labelpath):
    sep = ";"
    separator = re.compile(r'separator\s*(.+)')
    tiers = []
    start_times = []

    with labelpath.open() as f:
        for line_nr, line in enumerate(f):
            match = separator.match(line)
            if match:
                sep = match.group(1)

            if line.strip() == "#":
                break

        for line_nr, line in enumerate(f, line_nr):
            try:
                time, _, fields = line.split(None, 2)
            except ValueError:
                warnings.warn("Empty field encountered at line {} in {}".format(
                    line_nr + 1, str(labelpath)), RuntimeWarning)
                continue

            # convert time to sample number
            end_time = float(time)
            if len(fields) > 0:
                for i, field in enumerate(fields.split(sep)):
                    try:
                        tier = tiers[i]
                    except IndexError:
                        tier = []
                        tiers.append(tier)
                        start_times.append(0)

                    tier.append((start_times[i], end_time, field.strip()))
                    start_times[i] = end_time

    return tiers

# all labels:
# {'ih', 'ae+1', 'ay', 'no', 'uw+1', 'aw', 'iy+1', 'w', 'd', 'aw+1', 'oy', 'sh', 'aan', 'a', 'e', 'ayn', 'ih+1', 'hh', 'i', 'uw ix', 'ao+1', 'g', 'hhn', 'er', 'own', 'oy+1', 'tq', 'em', 'eh+1', 'el', 'b', 'ch', 'ao', 'aa', 'dh', 'id', 'iy', 'ow+1', 'nx', 'eh', 'awn', 'eng', 'r', 'ay+1', 'x', 'ihn', 'ow', 'ey', 'ae', 'ah r', 'p', 'h', 'ih l', 'ey+1', 'an', 'v', 'uwn', 'uhn', 'm', 'k', 'ern', 'ah l', 'ah', 't', 'ah+1n', 'ng', 'uw', 'iyn', 'j', 'uh', 'dx', 's', 'ahn', 'z', 'y', 'uh+1', 'oyn', 'en', 'aen', 'aa+1', 'l', 'f', 'ah n', 'ah ix', 'ehn', 'eyn', 'jh', 'q', 'aon', 'zh', 'ah+1', 'n', 'th'}
# {'SIL', '{E_TRANS}', '<EXCLUDE-name>', '<EXCLUDE>', 'LAUGH', 'VOCNOISE', 'IVER-LAUGH', 'UNKNOWN', 'NOISE', '{B_TRANS}', 'EXCLUDE', 'IVER', 'IVER y', '<exclude-Name>'}

SectionData = collections.namedtuple('SectionData', ['start', 'end'])
FileData = collections.namedtuple('FileData', ['transcription'])

def get_recording(filepath, section, exclude_nonspeech=True, **kwargs):
    with filepath.open('rb') as f:
        rate, samples = scipy.io.wavfile.read(f)
        samples = samples.astype('float32')

    nonspeech_labels = ('SIL', '{E_TRANS}', '{B_TRANS}', 'IVER',
                        'NOISE', 'EXCLUDE', '<EXCLUDE', '<exclude')

    tiers = parse_esps(filepath.parent / (filepath.stem + ".phones"))

    #if not exclude_nonspeech:
    #    return samples, tiers[0], [(0, (0, 0))]

    section_start = 0
    section_end = 0
    for start_time, end_time, label in tiers[0]:
        # convert time to sample number
        start, end = int(start_time * rate), int(end_time * rate)

        if any(label.startswith(nsl) for nsl in nonspeech_labels):
            if section_end > 0:
                section.create_section(
                    data=samples[section_start : section_end],
                    metadata=(section_start / rate, section_end / rate))
            section_start = end
            section_end = 0
        else:
            section_end = end

    section.metadata = (tiers[0])

def get_speaker(speaker, section):
    print(speaker)
    recordings = speaker.glob("{}*/*.wav".format(speaker.stem))
    for recording in recordings:
        get_recording(recording, section.create_section(recording.stem))

def get_corpus(corpuspath, hdf5file, outset):
    speakers = pathlib.Path(corpuspath).glob("s*")
    dataset = hdf5file.create_dataset(outset, (), 'i2')
    for speaker in speakers:
        get_speaker(speaker, dataset.create_section(speaker.stem))

@click.command()
@click.argument('hdf5file', type=dataset.HDF5TYPE)
@click.option('--corpuspath', type=click.Path(exists=True), required=True)
@click.option('-o', '--outset', default='samples')
def main(hdf5file, corpuspath, outset):
    get_corpus(corpuspath, hdf5file, outset)

if __name__ == '__main__':
    main()
