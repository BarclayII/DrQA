#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""

import logging
import numpy as np
import scipy.sparse as sp
import pickle
import tqdm
import heapq
import os

from multiprocessing.pool import ThreadPool
from functools import partial

from . import utils
from . import DEFAULTS
from .. import tokenizers

logger = logging.getLogger(__name__)


def readints(f):
    return [int(i) for i in f.readline().strip().split()]

def readsparsecol(f, n_rows):
    items = [i.split(':') for i in f.readline().strip().split()]
    i, d = zip(*[(int(i), float(d)) for i, d in items])
    return sp.csc_matrix((d, i, (0, len(i))), shape=(n_rows, 1))

class TfidfDocRanker(object):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, tfidf_path=None, strict=True):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        tfidf_path = tfidf_path or DEFAULTS['tfidf_path']
        logger.info('Loading %s' % tfidf_path)
        matrix, metadata = utils.load_sparse_csr(tfidf_path)
        self.doc_mat = matrix
        self.ngrams = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict

    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        return self.doc_dict[0][doc_id]

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        return self.doc_dict[1][doc_index]

    def closest_docs(self, query, k=1, approx=False):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        spvec = self.text2spvec(query)
        res = spvec * self.doc_mat

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        self._true_indices = res.indices[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None, approx=True):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k, approx=approx)
            results = threads.map(closest_docs, queries)
        return results

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=utils.filter_ngram)

    def text2spvec(self, query):
        """Create a sparse tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # Get hashed ngrams
        words = self.parse(utils.normalize(query))
        wids = [utils.hash(w, self.hash_size) for w in words]

        if len(wids) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((1, self.hash_size))

        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)

        # Count IDF
        Ns = self.doc_freqs[wids_unique]
        idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        data = np.multiply(tfs, idfs)

        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])
        spvec = sp.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.hash_size)
        )

        return spvec


class GreedyMIPSTfidfDocRanker(TfidfDocRanker):
    def __init__(self, tfidf_path=None, strict=True):
        TfidfDocRanker.__init__(self, tfidf_path, strict)

        for t in tqdm.trange(self.doc_mat.shape[0]):
            start, end = self.doc_mat.indptr[t:t+2]
            if start != end:
                data = self.doc_mat.data[start:end]
                indices = self.doc_mat.indices[start:end]
                order = data.argsort()[::-1]
                self.doc_mat.data[start:end] = data[order]
                self.doc_mat.indices[start:end] = indices[order]

        self.doc_mat_csc = self.doc_mat.tocsc()

    def conditer(self, t, w_t):
        assert w_t > 0
        start, end = self.doc_mat.indptr[t:t+2]
        if start == end:
            return None
        return iter(zip(self.doc_mat.indices[start:end], self.doc_mat.data[start:end]))

    def query(self, w, B):
        iters = {}
        Q = []
        C = set()

        for t, w_t in zip(w.indices, w.data):
            it = self.conditer(t, w_t)
            if it is None:
                continue
            j, h_jt = next(it)
            z = h_jt * w_t
            heapq.heappush(Q, (-z, t, j, h_jt))
            iters[t] = it

        while len(C) < B and len(Q) > 0:
            z, t, j, h_jt = heapq.heappop(Q)
            z = -z
            if j not in C:
                C.add(j)
            for j, h_jt in iters[t]:
                if not j in C:
                    z = h_jt * w_t
                    heapq.heappush(Q, (-z, t, j, h_jt))
                    break

        if len(C) == 0:
            C = set([0])    # return a default
        return C

    def closest_docs(self, query, k=1, approx=True):
        if not approx:
            return TfidfDocRanker.closest_docs(self, query, k, approx)
        spvec = self.text2spvec(query)
        candidates = list(self.query(spvec, k * 10))
        res = spvec * self.doc_mat_csc[:, candidates]

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(candidates[i]) for i in res.indices[o_sort]]
        return doc_ids, doc_scores


class CSXReader(object):
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)

        self.n_rows, self.n_cols, self.n_indptrs, self.n_indices, \
                self.n_data, self.a_indptrs, self.a_indices, self.a_data = \
                np.frombuffer(os.pread(self.fd, 64, 0), dtype='int64')
        self.o_indptrs = 64
        self.o_indices = self.o_indptrs + self.a_indptrs * self.n_indptrs
        self.o_data = self.o_indices + self.a_indices * self.n_indices

        self.indptr_dtype = 'int32' if self.a_indptrs == 4 else 'int64'
        self.indices_dtype = 'int32' if self.a_indices == 4 else 'int64'
        self.data_dtype = 'float32' if self.a_data == 4 else 'float64'

        self.indptr = np.frombuffer(
                os.pread(self.fd, self.a_indptrs * self.n_indptrs, self.o_indptrs),
                dtype=self.indptr_dtype
                )

    def fetch(self, t):
        start, end = self.indptr[t:t+2]
        indices = np.frombuffer(
                os.pread(self.fd,
                         self.a_indices * (end - start),
                         self.o_indices + self.a_indices * start),
                dtype=self.indices_dtype
                )
        data = np.frombuffer(
                os.pread(self.fd,
                         self.a_data * (end - start),
                         self.o_data + self.a_data * start),
                dtype=self.data_dtype
                )

        return indices, data


class DiskGreedyMIPSTfidfDocRanker(GreedyMIPSTfidfDocRanker):
    def __init__(self, tfidf_path=None, strict=True):
        tfidf_path = tfidf_path or DEFAULTS['tfidf_path']
        logger.info('Loading %s' % tfidf_path)
        with open(tfidf_path + '.meta', 'rb') as f:
            metadata = pickle.load(f)
        self.ngrams = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict

        self.csr = CSXReader(tfidf_path + '.csr')
        self.csc = CSXReader(tfidf_path + '.csc')

    def conditer(self, t, w_t):
        indices, data = self.csr.fetch(t)
        if len(indices) == 0:
            return None
        return iter(zip(indices, data))

    def closest_docs(self, query, k=1, approx=True):
        if not approx:
            return TfidfDocRanker.closest_docs(self, query, k, approx)
        spvec = self.text2spvec(query)
        candidates = list(self.query(spvec, k * 10))

        indices, data = zip(*(self.csc.fetch(c) for c in candidates))
        indptr = np.cumsum([0] + [len(i) for i in indices])
        indices = np.concatenate(indices)
        data = np.concatenate(data)
        candvec = sp.csc_matrix((data, indices, indptr), shape=(self.csr.n_rows, len(candidates)))

        res = spvec * candvec
        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(candidates[i]) for i in res.indices[o_sort]]
        return doc_ids, doc_scores


class MinHashTfidfDocRanker(TfidfDocRanker):
    def __init__(self, tfidf_path=None, strict=True):
        TfidfDocRanker.__init__(self, tfidf_path, strict)
        minhash = np.load(tfidf_path + '.minhash.npz')
        self.minhash_bases = minhash['bases']
        self.minhash_offsets = minhash['offsets']
        self.minhash = minhash['minhash'].T
        self.minhash_logfile = open('minhash.log', 'w')
        self.doc_mat_csc = self.doc_mat.tocsc()

    def closest_docs(self, query, k=1, approx=False):
        if not approx:
            result = TfidfDocRanker.closest_docs(self, query, k=k, approx=approx)
            return result
        spvec = self.text2spvec(query)
        minhash = (
                (spvec.nonzero()[1][:, None] * self.minhash_bases +
                    self.minhash_offsets) %
                spvec.shape[1] + 1
                ).min(0)
        matches = (self.minhash == minhash).sum(1)
        max_matches = matches.max()
        max_match_indices = (matches == max_matches).nonzero()[0]
        print(minhash)
        print(matches[self._true_indices], max_matches, max_match_indices)
        
        print(max_matches, len(max_match_indices), file=self.minhash_logfile)

        res = spvec * self.doc_mat_csc[:, max_match_indices]

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(max_match_indices[i]) for i in res.indices[o_sort]]
        return doc_ids, doc_scores


class FeatureHashingTfidfDocRanker(TfidfDocRanker):
    def __init__(self, tfidf_path=None, strict=True):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        TfidfDocRanker.__init__(self, tfidf_path, strict)
        self.doc_fhash, _ = utils.load_sparse_csr(tfidf_path + '.fhash')

    def closest_docs(self, query, k=1, approx=False):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        spvec = self.text2spvec(query) * self.doc_fhash.T
        res = spvec * self.doc_mat

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]
        return doc_ids, doc_scores


class SparseTfidfDocRanker(TfidfDocRanker):
    def __init__(self, tfidf_path=None, strict=True):
        tfidf_path = tfidf_path or DEFAULTS['tfidf_path']
        logger.info('Loading %s' % tfidf_path)

        with open(tfidf_path + '.meta', 'rb') as f:
            metadata = pickle.load(f)

        self.ngrams = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict

        with open(tfidf_path + '.hash', 'rb') as f:
            self.doc_hash = np.array(pickle.load(f))
            self.doc_hash_bins = np.array(list(set(self.doc_hash)))
        self.doc_base = np.load(tfidf_path + '.base')
        self.doc_file = open(tfidf_path)

        self.rows, self.cols = readints(self.doc_file)
        self.nonzero_cols = readints(self.doc_file)

        self.col_offset = []
        while True:
            self.col_offset.append(self.doc_file.tell())
            if len(self.doc_file.readline()) == 0:
                break

    def get_doc_spvec(self, index):
        self.doc_file.seek(self.col_offset[self.nonzero_cols.index(index)])
        return readsparsecol(self.doc_file, self.rows)

    def closest_docs(self, query, k=1, approx=True):
        if approx:
            return self.closest_docs_approx(query, k)

        q_spvec = self.text2spvec(query).todense().A[0]
        q_spvec = np.concatenate([q_spvec, np.zeros(self.rows - q_spvec.shape[0])])
        Q = sp.csr_matrix(q_spvec)

        dots = []
        P = []
        self.doc_file.seek(self.col_offset[0])
        for col in tqdm.tqdm(self.nonzero_cols):
            p_spvec = readsparsecol(self.doc_file, self.rows)
            P.append(p_spvec)
            if col == self.nonzero_cols[-1] or len(P) == 20000:
                P = sp.hstack(P)
                D = (Q * P).todense().A[0]
                dots.extend(D)
                P = []

        dots = np.array(dots)

        o_sort = np.argsort(dots)[::-1]
        if len(o_sort) > k:
            o_sort = o_sort[:k]

        print(np.array(self.nonzero_cols)[o_sort])

        doc_scores = dots[o_sort]
        doc_ids = [self.get_doc_id(i) for i in np.array(self.nonzero_cols)[o_sort]]

        return doc_ids, doc_scores

    def closest_docs_approx(self, query, k=1):
        q_spvec = self.text2spvec(query).todense().A[0]
        q_spvec = np.concatenate([q_spvec, np.zeros(self.rows - q_spvec.shape[0])])
        signs = np.clip(np.sign(self.doc_base @ q_spvec), 0, 1).astype('int')
        hashbin = int(''.join(str(s) for s in signs), 2)

        diff = np.array([bin(i).count('1') for i in self.doc_hash_bins ^ hashbin])
        diff_values = np.unique(np.sort(diff))
        closest = []

        for v in diff_values:
            new_bins = (diff == v).nonzero()[0]
            if len(closest) > 0 and len(closest) + len(new_bins) > 10:
                break
            closest.extend(new_bins)

        dots = []
        candids = np.isin(self.doc_hash, self.doc_hash_bins[closest]).nonzero()[0]
        for item in candids:
            p_spvec = self.get_doc_spvec(item).todense().A[:, 0]
            dot = p_spvec @ q_spvec
            dots.append(dot)
        dots = np.array(dots)

        o_sort = np.argsort(dots)[::-1]
        if len(o_sort) > k:
            o_sort = o_sort[:k]

        doc_scores = dots[o_sort]
        doc_ids = [self.get_doc_id(i) for i in candids[o_sort]]

        return doc_ids, doc_scores
