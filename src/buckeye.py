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
import itertools
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

def parse_vad(vad_file):
    vad = vad_file.read().splitlines()

    deliminator = ","
    header = vad[0].split(deliminator)
    if len(header) != 3:
        deliminator = " "
        header = vad[0].split(deliminator)
        assert len(header) == 3

    try:
        float(header[1])
        float(header[2])
    except ValueError:
        vad = vad[1:]

    vad = [(fname, float(start), float(end))
            for fname, start, end in (line.split(deliminator)
                                      for line in vad)]
    vad.sort()
    files = {fname: [(start, end) for _, start, end in values]
                for fname, values in itertools.groupby(vad, key=lambda x: x[0])}
    return files

# all labels:
# {'ih', 'ae+1', 'ay', 'no', 'uw+1', 'aw', 'iy+1', 'w', 'd', 'aw+1', 'oy', 'sh', 'aan', 'a', 'e', 'ayn', 'ih+1', 'hh', 'i', 'uw ix', 'ao+1', 'g', 'hhn', 'er', 'own', 'oy+1', 'tq', 'em', 'eh+1', 'el', 'b', 'ch', 'ao', 'aa', 'dh', 'id', 'iy', 'ow+1', 'nx', 'eh', 'awn', 'eng', 'r', 'ay+1', 'x', 'ihn', 'ow', 'ey', 'ae', 'ah r', 'p', 'h', 'ih l', 'ey+1', 'an', 'v', 'uwn', 'uhn', 'm', 'k', 'ern', 'ah l', 'ah', 't', 'ah+1n', 'ng', 'uw', 'iyn', 'j', 'uh', 'dx', 's', 'ahn', 'z', 'y', 'uh+1', 'oyn', 'en', 'aen', 'aa+1', 'l', 'f', 'ah n', 'ah ix', 'ehn', 'eyn', 'jh', 'q', 'aon', 'zh', 'ah+1', 'n', 'th'}
# {'SIL', '{E_TRANS}', '<EXCLUDE-name>', '<EXCLUDE>', 'LAUGH', 'VOCNOISE', 'IVER-LAUGH', 'UNKNOWN', 'NOISE', '{B_TRANS}', 'EXCLUDE', 'IVER', 'IVER y', '<exclude-Name>'}

SectionData = collections.namedtuple('SectionData', ['start', 'end'])
FileData = collections.namedtuple('FileData', ['transcription'])

def get_recording(filepath, section):
    with filepath.open('rb') as f:
        rate, samples = scipy.io.wavfile.read(f)
        samples = samples.astype('float32')

    nonspeech_labels = ('SIL', '{E_TRANS}', '{B_TRANS}', 'IVER',
                        'NOISE', 'EXCLUDE', '<EXCLUDE', '<exclude')

    tiers = parse_esps(filepath.parent / (filepath.stem + ".phones"))

    section.create_section(data=samples, metadata=(0, samples.shape[0] / rate))

    section.metadata = tiers[0]

def get_speaker(speaker, section, files=None, **kwargs):
    print(speaker)
    recordings = sorted(recording
                        for recording in speaker.glob("{}*/*.wav".format(speaker.stem))
                        if files is None or recording.stem in files)
    for recording in recordings:
        get_recording(recording, section.create_section(recording.stem), **kwargs)

    return recordings

def get_corpus(corpuspath, hdf5file, outset, filter_speakers=None, **kwargs):
    speakers = sorted(pathlib.Path(corpuspath).glob("s*"))
    dataset = hdf5file.create_dataset(outset, (), 'i2', overwrite=True)

    mapping = {}
    for speaker in speakers:
        if filter_speakers is not None and speaker.name not in filter_speakers:
            continue
        recordings = get_speaker(speaker, dataset.create_section(speaker.stem), **kwargs)
        mapping.update({recording.stem: speaker.stem for recording in recordings})

    dataset.metadata = mapping

@click.command()
@click.argument('hdf5file', type=dataset.HDF5TYPE)
@click.option('--corpuspath', type=click.Path(exists=True), required=True)
@click.option('-o', '--outset', default='samples')
@click.option('--files', type=click.File('r'))
def main(hdf5file, corpuspath, outset, files):
    if files is not None:
        files = {f.rsplit('.', 1)[0] for f in files}
        speakers = {f[:3] for f in files}
    else:
        speakers = None

    get_corpus(corpuspath, hdf5file, outset,
               filter_speakers=speakers, files=files)

if __name__ == '__main__':
    main()
