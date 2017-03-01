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
import concurrent.futures
import itertools
import math
import os
import pathlib
import pickle
import random

import numpy as np
import scipy.io.wavfile
import scipy.misc
import scipy.spatial.distance
import scipy.stats
import numba
import sklearn.metrics

import click

import buckeye
import dataset
import features

from _train import *

@click.group()
def main():
    pass

@main.command()
@click.argument('hdf5file', type=dataset.HDF5TYPE)
@click.option('-i', '--inset', required=True)
@click.option('-o', '--outfile', type=click.File('wb'), required=True)
@click.option('--sample-rate', type=int, required=True)
def to_wav(hdf5file, inset, outfile, sample_rate):
    data = hdf5file[inset].data[:]
    scipy.io.wavfile.write(outfile, sample_rate, data)

@main.command()
@click.argument('hdf5file', type=dataset.HDF5TYPE)
@click.option('-i', '--inset', required=True)
def list_sections(hdf5file, inset):
    for section in hdf5file[inset]:
        print(section.name)

@main.command()
@click.argument('infile', type=click.File('rb'))
@click.option('-D', required=True, type=int)
@click.option('--vad', required=True, type=click.File('r'))
@click.option('--window-shift', default=10)
@click.option('-o', '--output-dir', required=True,
              type=click.Path(file_okay=False))
def split_plp(infile, d, vad, window_shift, output_dir):
    data = np.fromfile(infile, dtype='f4').reshape(-1, d)
    vad = buckeye.parse_vad(vad)
    outpath = pathlib.Path(output_dir)
    name, ext = infile.name.split("/")[-1].split(".", 1)
    window_shift /= 1000

    os.makedirs(output_dir, exist_ok=True)

    for i, (start, end) in enumerate(vad[name]):
        fstart, fend = int(start / window_shift), int(end / window_shift)
        outname = "{}.{}".format(name, i)

        print(name, outname, start, end)
        data[fstart:fend].tofile(str(outpath / (outname + "." + ext)))

def sec_to_frame(start, stop, window_length, window_shift):
    fstart = max(0, math.ceil((start - window_length/2) / window_shift))
    fstop = math.floor((stop - window_length/2) / window_shift)
    return fstart, fstop

@main.command()
@click.argument('hdf5file', type=dataset.HDF5TYPE)
@click.option('-i', '--inset', required=True)
@click.option('-o', '--outset', required=True)
@click.option('--vad', required=True, type=click.File('r'))
@click.option('--window-shift', required=True, type=int)
@click.option('--window-length', required=True, type=int)
def extract_vad(hdf5file, inset, outset, vad, window_shift, window_length):
    datset = hdf5file[inset]
    outset = hdf5file.create_dataset(outset, datset.dims, datset.data.dtype,
                                     overwrite=True)
    vad = buckeye.parse_vad(vad)

    window_shift /= 1000
    window_length /= 1000
    for speaker in datset:
        print(speaker.name)
        speaker_sec = outset.create_section(speaker.name)
        for recording in speaker:
            recording_sec = speaker_sec.create_section(recording.name)
            data = recording.data
            for start, stop in vad[recording.name]:
                # get frames whose middle points are within (start, stop)
                # from n * window_shift + window_length/2 = t
                fstart, fstop = sec_to_frame(start, stop, window_length, window_shift)
                secdat = data[fstart:fstop]
                recording_sec.create_section(data=secdat,
                                             metadata=(start, stop))

@numba.jit(nopython=True, nogil=True)
def numba_sum(a):
    x = np.zeros((a.shape[0], 1))
    for i in range(a.shape[0]):
        x[i] = a[i].sum()
    return x

@numba.jit(nopython=True, nogil=True)
def numba_reverse(l):
    for i in range(len(l) // 2):
        l[i], l[-i-1] = l[-i-1], l[i]

@numba.jit(nopython=True, nogil=True)
def kullback_leibler(a, b, eps=np.finfo(np.dtype('f4')).eps):
    a = a + eps
    b = b + eps
    a /= numba_sum(a)
    b /= numba_sum(b)
    ent = numba_sum(a * np.log2(a))
    cross = a @ np.log2(b.T)
    return ent - cross

def jensen_shannon(a, b, eps=np.finfo(np.dtype('f4')).eps):
    a = a[:,np.newaxis] + eps
    b = b[np.newaxis,:] + eps
    a /= a.sum(axis=-1)[...,np.newaxis]
    b /= b.sum(axis=-1)[...,np.newaxis]
    m = (a + b)/2
    return ((a * np.log2(a/m) + b * np.log2(b/m))/2).sum(axis=-1)

@numba.jit(nopython=True, nogil=True)
def align(dist):
    D = np.zeros((dist.shape[0] + 1, dist.shape[1] + 1))
    B = np.zeros((dist.shape[0] + 1, dist.shape[1] + 1, 2), dtype=np.int32)

    D[:,0] = D[0,:] = np.inf
    D[0,0] = 0

    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            D[i+1,j+1] = min(D[i,j], D[i,j+1], D[i+1,j]) + dist[i,j]
            B[i+1,j+1] = ([i,j] if D[i,j] < D[i,j+1] and D[i,j] < D[i+1,j]
                          else ([i,j+1] if D[i,j+1] < D[i+1,j] else [i+1,j]))

    indices1 = []
    indices2 = []

    i = np.array(dist.shape, dtype=np.int32)
    while i[0] != 0 and i[1] != 0:
        indices1.append(i[0] - 1)
        indices2.append(i[1] - 1)
        i = B[i[0],i[1]]

    numba_reverse(indices1)
    numba_reverse(indices2)

    return indices1, indices2, D[-1,-1]

def binary_search(transcription, point):
    start, end = 0, len(transcription) - 1
    while start <= end:
        middle = (start + end) // 2
        x = transcription[middle]
        if point < x[0]:
            end = middle - 1
        elif point >= x[1]:
            start = middle + 1
        else:
            break
    else: # nobreak
        raise RuntimeError("Couldn't find corresponding transcription")
    return x


@main.command()
@click.argument('hdf5file', type=dataset.Hdf5Type('r'))
@click.option('-i', '--inset', required=True)
@click.option('--window-shift', type=int, required=True)
@click.option('--window-length', type=int, required=True)
@click.option('--model', type=click.File('rb'))
@click.option('--save-averages', type=click.File('wb'), required=True)
def label_frames(hdf5file, inset, window_shift, window_length, model, save_averages):
    datset = hdf5file[inset]
    window_shift, window_length = window_shift / 1000, window_length / 1000

    if model is not None:
        _, _, layer, _ = pickle.load(model)
        W = layer._W.get_value()[0]
        W = W[:,W.sum(axis=0) > 0]

    speakers = {}

    for speaker in datset:
        print(speaker.name)
        avgs = {}
        for recording in speaker:
            transcription = iter(recording.metadata)
            start, stop, phoneme = next(transcription)
            framewise = []
            frames = recording.data[:]
            if model is not None:
                frames = frames @ W

            for i in range(frames.shape[0]):
                t = window_shift * i + window_length / 2
                try:
                    while t > stop:
                        start, stop, phoneme = next(transcription)
                except StopIteration:
                    # just use the last value
                    pass

                if phoneme not in avgs:
                    avgs[phoneme] = np.zeros(frames.shape[-1])
                avgs[phoneme] += frames[i]

        speakers[speaker.name] = avgs

    pickle.dump(speakers, save_averages)


def parse_clusters(clusterfile, recording_speaker_mapping=None, return_clusters=False):
    clusters = []
    cur_cluster = None
    for line in clusterfile:
        if line.startswith("Class"):
            if cur_cluster is not None:
                clusters.append(cur_cluster)
            cur_cluster = []
        elif line == "\n":
            pass
        else:
            recording, start, stop = line.split()
            if recording_speaker_mapping is None:
                this = recording, float(start), float(stop)
            else:
                this = (recording_speaker_mapping[recording], recording,
                        float(start), float(stop))
            cur_cluster.append(this)
    clusters.append(cur_cluster)

    if return_clusters:
        return clusters

    same_pairs = []
    for i, cluster in enumerate(clusters, 1):
        for this, other in itertools.combinations(cluster, r=2):
            word1, word2 = (this, other) if random.random() < 0.5 else (other, this)
            same_pairs.append((i, i, word1, word2))

    return same_pairs

def samespeaker_ratio(same_pairs):
    return sum(1 for _, _, a, b in same_pairs if a[0] == b[0]) / len(same_pairs)

def sample_mismatches(same_pairs, ratio=None, mismatch_ratio=1, sample_triples=False):
    ratio = ratio or samespeaker_ratio(same_pairs)

    diff_pairs = []
    same_pairs_flattened = [(cluster, word)
                             for cluster, _, w1, w2 in same_pairs
                             for word in (w1, w2)]

    for i in range(int(len(same_pairs) * mismatch_ratio)):
        same_speaker = random.random() < ratio

        # repeat until successfully sampled
        while True:
            try:
                if sample_triples:
                    word1 = same_pairs[i][0], same_pairs[i][2]
                else:
                    word1 = random.choice(same_pairs_flattened)

                word2 = random.choice([
                    (cluster, word) for cluster, word in same_pairs_flattened
                    if cluster != word1[0] and same_speaker == (word[0] == word1[1][0])])
            except IndexError:
                pass
            else:
                break

        diff_pairs.append((word1[0], word2[0], word1[1], word2[1]))

    return diff_pairs

@numba.jit(nogil=True)
def calculate_similarities(similarities, fragments, i,
                           symmetrize=False, js=False, cos=False):
    if symmetrize or js or cos:
        start = i
    else:
        start = 0

    for j in range(start, len(fragments)):
        if js:
            dist = np.sqrt(jensen_shannon(fragments[i], fragments[j]) + 1e-7)
        elif cos:
            dist = scipy.spatial.distance.cdist(fragments[i], fragments[j], metric='cosine')
        elif symmetrize:
            dist = (kullback_leibler(fragments[i], fragments[j]) +
                    kullback_leibler(fragments[j], fragments[i]).T) / 2
        else:
            dist = kullback_leibler(fragments[i], fragments[j])
        similarities[i,j] = align(dist)[-1]

    if symmetrize or js or cos:
        similarities[i:,i] = similarities[i,i:]

    return i

@main.command()
@click.argument('hdf5file', type=dataset.Hdf5Type('r'))
@click.option('-i', '--inset', required=True)
@click.option('--clusters', type=click.File('r'), required=True)
@click.option('--num-clusters', type=int, default=1000, show_default=True)
@click.option('--distance', type=click.Choice(['kl', 'kl-sym', 'js', 'cos']), required=True)
@click.option('--window-shift', type=int, required=True)
@click.option('--window-length', type=int, required=True)
@click.option('--model', type=click.File('rb'))
@click.option('--save-score', type=click.File('w'))
@click.option('--num-threads', type=int, default=12, show_default=True)
def evaluate(hdf5file, inset, clusters, distance, window_shift, window_length,
             save_score, num_clusters, model, num_threads):
    datset = hdf5file[inset]
    mapping = datset.metadata
    clusters = parse_clusters(clusters, mapping, return_clusters=True)[:num_clusters]

    window_length, window_shift = window_length / 1000, window_shift / 1000

    def get_data(x):
        s, rec, start, stop = x
        start, stop = sec_to_frame(start, stop, window_length, window_shift)
        return datset[s][rec].data[start:stop]

    fragments = [get_data(fragment) for cluster in clusters for fragment in cluster]
    hdf5file.close()

    if model is not None:
        import train
        get_output, _, _ = train.open_model(model)
        fragments = [get_output(fragment) for fragment in fragments]

    similarities = np.zeros((len(fragments), len(fragments)))
    indices = list(range(len(fragments)))
    threads = concurrent.futures.ThreadPoolExecutor(num_threads)
    result = threads.map(lambda i: calculate_similarities(
        similarities, fragments, i,
        symmetrize=distance == 'kl-sym', js=distance == 'js',
        cos=distance == 'cos'), indices)
    print("Running threads...")
    for i, _ in enumerate(result, 1):
        print("{}/{}".format(i, len(fragments)))

    print("Calculating silhouette score...")
    labels = np.array([i for i, cluster in enumerate(clusters) for _ in cluster])
    score = sklearn.metrics.silhouette_score(similarities, labels, metric='precomputed')
    print("Silhouette score: {}".format(score))

    if save_score is not None:
        save_score.write(str(score) + '\n')

@main.command()
@click.argument('hdf5file', type=dataset.HDF5TYPE)
@click.option('-i', '--inset', required=True)
@click.option('-o', '--outset', required=True)
@click.option('--clusters', type=click.File('r'), required=True)
@click.option('--window-shift', type=int)
@click.option('--window-length', type=int)
@click.option('--sample-rate', type=int)
@click.option('--sample-triples', is_flag=True)
@click.option('--validation-ratio', default=0.3, show_default=True)
@click.option('--mismatch-ratio', default=1, show_default=True)
def from_matchlist(hdf5file, inset, outset, clusters, window_shift, window_length,
                   sample_rate, validation_ratio, sample_triples, mismatch_ratio):
    datset = hdf5file[inset]
    data = datset.data
    mapping = datset.metadata

    if sample_rate is None and (window_shift is None or window_length is None):
        raise click.UsageError("Must supply either --sample-rate or "
                               "--window-shift and --window-length.")

    if window_shift is not None and window_length is not None:
        window_shift /= 1000
        window_length /= 1000

    speakers = sorted(sec.name for sec in datset)
    speaker_ids = {s: i for i, s in enumerate(speakers)}

    outset_ids = hdf5file.create_dataset(
        outset + "_ids", (2,), 'i2', overwrite=True)
    outset_ids.metadata = speaker_ids

    outset = hdf5file.create_dataset(outset, (2, *data.shape[1:]),
                                     'f4', overwrite=True)

    same_pairs = parse_clusters(clusters, mapping)

    # percentage of positive examples belonging to the same speaker
    ratio = samespeaker_ratio(same_pairs)
    validation_split = int(len(same_pairs) * validation_ratio)

    random.shuffle(same_pairs)
    same_valid, same_train = same_pairs[:validation_split], same_pairs[validation_split:]

    diff_train = sample_mismatches(same_train, ratio=ratio, mismatch_ratio=mismatch_ratio,
                                   sample_triples=sample_triples)
    diff_valid = sample_mismatches(same_valid, ratio=ratio, mismatch_ratio=mismatch_ratio,
                                   sample_triples=sample_triples)

    all_pairs = [[same_train, diff_train], [same_valid, diff_valid]]

    def get_time(start, stop):
        if window_length is not None and window_shift is not None:
            return sec_to_frame(start, stop, window_length, window_shift)
        else:
            return int(start * sample_rate), int(stop * sample_rate)

    frame_count = 0
    for sec, data in zip(('train', 'valid'), all_pairs):
        sec_ids = outset_ids.create_section(sec)
        sec_data = outset.create_section(sec)
        for subsec, subdata in zip(('same', 'diff'), data):
            subsec_ids = sec_ids.create_section(subsec)
            subsec_data = sec_data.create_section(subsec)

            for i, (w1, w2, (s1, rec1, start1, stop1), \
                            (s2, rec2, start2, stop2)) in enumerate(subdata, 1):
                print("Generating {}-{} pair {}/{}...".format(sec, subsec, i, len(subdata)))
                start1, stop1 = get_time(start1, stop1)
                start2, stop2 = get_time(start2, stop2)

                feat1 = datset[s1][rec1].data[start1:stop1]
                feat2 = datset[s2][rec2].data[start2:stop2]

                if subsec == 'same':
                    dist = scipy.spatial.distance.cdist(feat1, feat2, metric='euclidean')
                    indices1, indices2, _ = align(dist)
                    feat1 = feat1[indices1]
                    feat2 = feat2[indices2]
                else:
                    length = min(feat1.shape[0], feat2.shape[0])
                    feat1, feat2 = feat1[:length], feat2[:length]

                subsec_ids.create_section(
                    data=np.stack((np.repeat(speaker_ids[s1], feat1.shape[0]),
                                   np.repeat(speaker_ids[s2], feat2.shape[0])),
                                  axis=1))
                subsec_data.create_section(data=np.stack((feat1, feat2), axis=1))

                frame_count += feat1.shape[0]

    print("Wrote a total of {} frames.".format(frame_count))
    print("Same-speaker ratio:", ratio)

    #with open('frame_matches.pkl', 'wb') as f:
    #    pickle.dump(frame_matches, f)



@main.command()
@click.option('--dedups', type=click.File('r'), required=True)
@click.option('--nodes', type=click.File('r'), required=True)
@click.option('--splitlist', type=click.File('r'))
def matchlist_to_clusters(dedups, nodes, splitlist):
    if splitlist is not None:
        splitlist = {fragment: (speaker, float(start), float(stop))
                     for speaker, fragment, start, stop in (
                         line.split() for line in splitlist)}

    nodelist = []
    for node in nodes:
        fragment, match_start, match_stop, _, _, _ = node.split()
        match_start, match_stop = int(match_start) / 100, int(match_stop) / 100
        if splitlist is None:
            f, start, stop = fragment, match_start, match_stop
        else:
            f, frag_start, frag_stop = splitlist[fragment]
            start, stop = frag_start + match_start, frag_start + match_stop

        nodelist.append((f, start, stop))

    for i, cluster in enumerate(dedups, 1):
        print("Class", i)
        for node in cluster.split():
            node = int(node) - 1
            f, start, stop = nodelist[node]
            print("{} {:.2f} {:.2f}".format(f, start, stop))

        print()

@main.command()
@click.option('--corpuspath', type=click.Path(exists=True), required=True)
@click.option('--clusters', type=click.File('w'), required=True)
@click.option('--nodes', type=click.File('w'), required=True)
@click.option('--files', type=click.File('r'), required=True)
def generate_gold_clusters(corpuspath, clusters, nodes, files):
    files = {f.split(".")[0] for f in files}
    ratio = 0.7318448883666275

    transcriptions = {}
    corpus = pathlib.Path(corpuspath)
    for speaker in corpus.iterdir():
        for recording in speaker.iterdir():
            if recording.name in files:
                transcription = recording / (recording.name + ".words")
                transcriptions[recording.name] = buckeye.parse_esps(transcription)[0]

    words = collections.defaultdict(lambda: [])
    for recording, transcription in transcriptions.items():
        for start, stop, word in transcription:
            if word[0] not in ('{', '<'):
                words[word].append((recording, start, stop))

    words = {word: cluster for word, cluster in words.items() if len(cluster) >= 2}
    for cluster in words.values():
        random.shuffle(cluster)

    final_clusters = []
    while len(final_clusters) < 4800:
        word, cluster = random.choice(list(words.items()))
        if len(cluster) == 2:
            cluster_len = 2
        else:
            p = min(1, 2 * np.log2(len(cluster)) / len(cluster)**1.4)
            cluster_len = max(2, np.random.binomial(len(cluster), p))

        extract, leave = cluster[:cluster_len], cluster[cluster_len:]
        if len(leave) <= 1:
            del words[word]
        else:
            words[word] = leave

        final_clusters.append(extract)

    i = 1
    for cluster in final_clusters:
        clusters.write(" ".join(str(x) for x in range(i, i + len(cluster))) + "\n")
        for recording, start, stop in cluster:
            nodes.write("{}\t{}\t{}\t1.0\t0.0\t0.0\n".format(recording, int(start*100), int(stop*100)))
        i += len(cluster)


def predict_proba_uniform(gmm, X):
    likelihoods = gmm._estimate_log_prob(X)
    norm = scipy.misc.logsumexp(likelihoods, axis=1)
    return np.exp(likelihoods - norm[:,np.newaxis])

@main.command()
@click.argument('hdf5file', type=dataset.HDF5TYPE)
@click.option('-i', '--inset', required=True)
@click.option('-o', '--outset', required=True)
@click.option('--gmm-set')
@click.option('--universal-gmm', type=click.File('rb'))
@click.option('--uniform-prior', is_flag=True)
def extract_posteriors(hdf5file, inset, outset, gmm_set, universal_gmm, uniform_prior):
    if universal_gmm is not None:
        gmm = pickle.load(universal_gmm)
    else:
        gmm = None

    datset = hdf5file[inset]
    if gmm_set is None:
        gmm_set = datset
    else:
        gmm_set = hdf5file[gmm_set]

    @features.feature_extractor()
    def extract_gmm(**kwargs):
        if universal_gmm is None:
            dims = gmm_set[0].metadata.n_components
        else:
            dims = gmm.n_components

        def extractor(data):
            if uniform_prior:
                return predict_proba_uniform(gmm, data)
            else:
                return gmm.predict_proba(data)

        return extractor, (dims,)

    def update_gmm(sec, level):
        nonlocal gmm
        if level == 1:
            print(sec.name)
        if universal_gmm is None and level == 1:
            gmm = gmm_set[sec.name].metadata

    extractor, dims = extract_gmm()
    outset = hdf5file.create_dataset(outset, dims, 'f4', overwrite=True)
    features.transform_dataset(extractor, datset, outset, callback=update_gmm)

@main.command()
@click.argument('hdf5file', type=dataset.HDF5TYPE)
@click.option('-i', '--inset', required=True)
@click.option('-c', '--clusters', type=int, default=128)
@click.option('--universal', type=click.Path())
@click.option('--covariance', type=click.Choice(['full', 'diag']), required=True)
def cluster(hdf5file, inset, clusters, universal, covariance):
    import pickle
    import sklearn.mixture

    def _cluster(data, **kwargs):
        default_params = {"init_params": 'random'}
        params = {**default_params, **kwargs}
        print("Clustering with parameters", params)
        gmm = sklearn.mixture.GaussianMixture(
            n_components=clusters, verbose=2,
            covariance_type=covariance,
            **params)
        gmm.fit(data)
        return gmm

    dset = hdf5file[inset]
    if universal is None:
        for section in dset:
            data = section.data[:]
            print("Clustering section {} of size {}".format(section.name, data.shape))
            section.metadata = _cluster(data)
    else:
        data = dset.data[:]
        print("Clustering all data of size {}".format(data.shape))
        gmm = _cluster(data, max_iter=200, tol=0.0001)
        with open(universal, 'wb') as f:
            pickle.dump(gmm, f)

if __name__ == '__main__':
    main()
