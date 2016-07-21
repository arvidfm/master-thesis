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
import math
import pickle

import numpy as np

import click
import h5py

ref_dtype = h5py.special_dtype(ref=h5py.Reference)

MAX_ATTRIBUTE_SIZE = 64000

class _Section:

    def __init__(self, parent, root, group):
        self._parent = parent
        self._root = root
        self._group = group

        if not '_subsections_num' in self._group.attrs:
            self._group.attrs['_subsections_num'] = 0

    def create_section(self, name=None, *, data=None, metadata=None):
        num_subsections = self._group.attrs['_subsections_num']
        key = '_subsection{}'.format(num_subsections)

        subsection = self._group.create_group(key)
        self._group.attrs['_subsections_num'] = num_subsections + 1
        if name is not None:
            subsection.attrs['_name'] = name
            self._group.attrs[name] = num_subsections

        return Section(self, self._root, subsection, data, metadata)

    @property
    def name(self):
        try:
            return self._group.attrs['_name']
        except KeyError:
            return None

    def __getitem__(self, key):
        if not isinstance(key, int):
            key = self._group.attrs[key]
        if key < 0:
            key = len(self) + key

        name = "_subsection{}".format(key)
        return Section(self, self._root, self._group[name], None, None)

    def __len__(self):
        return self._group.attrs['_subsections_num']

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def deepiter(self):
        for child in self:
            yield child
            yield from child.deepiter()

class Section(_Section):

    def __init__(self, parent, root, group, data, metadata):
        super().__init__(parent, root, group)

        dataset = self._root._data
        start = dataset.shape[0]
        if not '_start' in self._group.attrs:
            self._group.attrs['_start'] = start
            self._group.attrs['_end'] = start

        if data is not None:
            end = start + data.shape[0]
            dataset.resize((end, *self._root.dims))
            dataset[start:] = data

            cur = self
            while cur is not self._root:
                cur._group.attrs['_end'] = end
                cur = cur._parent

        if metadata is not None:
            self.metadata = metadata

    @property
    def metadata(self):
        try:
            metadata_num = self._group.attrs['_metadata_num']
            buff = bytearray()

            for i in range(metadata_num):
                buff.extend(self._group.attrs['_metadata{}'.format(i)])

            return pickle.loads(buff)
        except KeyError:
            return None

    @metadata.setter
    def metadata(self, value):
        try:
            del self.metadata
        except KeyError:
            pass

        dump = pickle.dumps(value)

        for i, start in enumerate(range(0, len(dump), MAX_ATTRIBUTE_SIZE)):
            self._group.attrs['_metadata{}'.format(i)] = np.void(
                dump[start : start + MAX_ATTRIBUTE_SIZE])
        self._group.attrs['_metadata_num'] = i + 1

    @metadata.deleter
    def metadata(self):
        metadata_num = self._group.attrs['_metadata_num']
        for i in range(metadata_num):
            del self._group.attrs['_metadata{}'.format(i)]

    @property
    def data(self):
        start, end = self._group.attrs['_start'], self._group.attrs['_end']
        return self._root.data[start:end]

    @property
    def sectiondata(self):
        start = self._group.attrs['_start']
        if len(self) > 0:
            end = self[0]._group.attrs['_start']
        else:
            end = self._group.attrs['_end']

        return self._root.data[start:end] if end > start else None

class Dataset(_Section):

    def __init__(self, group, dims, dtype):
        super().__init__(None, self, group)

        if dims is None:
            self._data = self._group['_data']
            self.dims = self._data.shape[1:]
        else:
            self._data = self._group.create_dataset(
                "_data", (0, *dims), dtype=dtype, chunks=True, maxshape=(None, *dims))
            self.dims = dims

    def clear_sections(self):
        for subsection in self._group:
            if subsection != '_data':
                del self._group[subsection]

        for attr in self._group.attrs:
            del self._group.attrs[attr]

        self._group.attrs['_subsections_num'] = 0

    @property
    def data(self):
        return self._data


class DataFile:

    def __init__(self, hfile, mode=None):
        self._hfile = h5py.File(hfile, mode=mode)

    def create_dataset(self, name, dims, dtype):
        return Dataset(self._hfile.create_group(name), dims, dtype)

    def copy(self, source, dest, keep_sections=True):
        if keep_sections:
            self._hfile.copy(source, dest)
        else:
            self._hfile.copy('{}/_data'.format(source), '{}/_data'.format(dest))
            self._hfile[dest].attrs['_subsections_num'] = 0

    def __getitem__(self, key):
        return Dataset(self._hfile[key], None, None)

    def __delitem__(self, key):
        del self._hfile[key]


class Hdf5Type(click.ParamType):
    def convert(self, value, param, ctx):
        try:
            return DataFile(value)
        except OSError:
            self.fail("{} is not a valid HDF5 file".format(value), param, ctx)

HDF5TYPE = Hdf5Type()

if __name__ == '__main__':
    pass
    #main()
