
import math
import pickle

import numpy as np

import h5py

ref_dtype = h5py.special_dtype(ref=h5py.Reference)

MAX_ATTRIBUTE_SIZE = 64000

def nondestructive():
    pass

def inplace():
    pass

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
            dataset.resize((end, *self._root._dims))
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
        return self._root._data[start:end]


class Dataset(_Section):

    def __init__(self, group, dims, dtype):
        super().__init__(None, self, group)

        if dims is None:
            self._data = self._group['_data']
            self._dims = self._data.shape[1:]
        else:
            self._data = self._group.create_dataset(
                "_data", (0, *dims), dtype=dtype, chunks=True, maxshape=(None, *dims))
            self._dims = dims

    @property
    def data(self):
        return self._data


class DataFile:

    def __init__(self, hfile, mode=None):
        self._hfile = h5py.File(hfile, mode=mode)

    def create_dataset(self, name, dims, dtype):
        return Dataset(self._hfile.create_group(name), dims, dtype)

    def __getitem__(self, key):
        return Dataset(self._hfile[key], None, None)

    def __delitem__(self, key):
        del self._hfile[key]

if __name__ == '__main__':
    import tempfile
    f = tempfile.NamedTemporaryFile()
    df = DataFile(f.name)
    samples = df.create_dataset('samples', (), 'int')
    s0 = samples.create_section('s0')
    s01 = s0.create_section('s01', data=np.array([1, 2, 3, 4]))
    s02 = s0.create_section('s02', data=2*np.array([1, 2, 3, 4]))
    print(s0.data, s01.data, s02.data)
    s1 = samples.create_section('s1')
    s11 = s1.create_section(data=np.array([1, 2, 3, 4, 5, 6, 7]), metadata="yes")
    s12 = s1.create_section(data=np.array([1, 2, 3]), metadata=(1, 2))
    print(s0.data, s01.data, s02.data, s1.data, s11.data, s12.data)
    print(samples.data[:])
    print(s11.metadata, s12.metadata)
    s12.metadata = b'\x00'
    print(s12.metadata)
    print(df['samples'].data[:])
    print(list(df['samples']['s1']._group))
    print(df['samples'][0][0].data)
