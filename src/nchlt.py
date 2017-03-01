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

import pathlib

import scipy.io.wavfile
import click

import dataset
import buckeye

def get_recording(filepath, section):
    with filepath.open('rb') as f:
        rate, samples = scipy.io.wavfile.read(f)
        samples = samples.astype('float32')

    section.create_section(data=samples, metadata=(0, samples.shape[0] / rate))

def get_speaker(speaker, section, files=None, **kwargs):
    print(speaker)
    recordings = sorted(recording for recording in speaker.glob("*.wav".format(speaker.stem))
                        if files is None or recording.stem in files)
    for recording in recordings:
        get_recording(recording, section.create_section(recording.stem), **kwargs)

    return recordings

def get_corpus(corpuspath, hdf5file, outset, filter_speakers=None, **kwargs):
    speakers = sorted(pathlib.Path(corpuspath).glob("*"))
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
        speakers = {f.split("_")[2][:-1] for f in files}
    else:
        speakers = None

    get_corpus(corpuspath, hdf5file, outset,
               filter_speakers=speakers, files=files)

if __name__ == '__main__':
    main()
