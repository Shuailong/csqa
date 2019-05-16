#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-15 12:26:31
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-15 12:26:35

"""Rank sentences with TF-IDF scores"""

from typing import List, Dict, Union
import logging
from collections import Counter

import prettytable
import numpy as np
from .spacy_tokenizer import SpacyTokenizer

from .utils import normalize, murmurhash, filter_ngram


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TfidfSentenceRanker():
    """Calculate TFIDF scores of the sentences.
    Scores new queries by taking dot products.
    """

    def __init__(self,
                 sentences: List[Union[Dict[str, str], str]],
                 hash_size: int = 10000,
                 ngrams: int = 2) -> None:
        self._hash_size = hash_size
        self._ngrams = ngrams
        self._tokenizer = SpacyTokenizer()
        self._sentences = sentences
        self._num_sents = len(sentences)
        if isinstance(sentences[0], dict):
            sentences = [sentence['sentence'] for sentence in sentences]
        self._sents_mat = self._build_sent_mat(sentences)

    def _build_sent_mat(self, sentences: List[str]) -> np.ndarray:
        self._sent_freqs = Counter()
        # Get hashed ngrams
        for sent in sentences:
            words = self.parse(normalize(sent))
            wids = (murmurhash(w, self._hash_size) for w in words)
            wids = set(wids)
            for wid in wids:
                self._sent_freqs[wid] += 1

        mat = np.stack((self._text2vec(sent) for sent in sentences))
        return mat.squeeze(1).transpose()

    def parse(self, query: str) -> List[str]:
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self._tokenizer.tokenize(query)
        return tokens.ngrams(n=self._ngrams, uncased=True,
                             filter_fn=filter_ngram)

    def closest_sents(self, query: str, sent_max: int = 5) -> List[Dict[str, Union[float, str]]]:
        """
        Closest docs by dot product between query and sentences
        in tfidf weighted word vector space.
        """
        vec = self._text2vec(query)
        res = np.matmul(vec, self._sents_mat).squeeze(0)

        if len(res) <= sent_max:
            o_sort = np.argsort(-res)
        else:
            o = np.argpartition(-res, sent_max)[:sent_max]
            o_sort = o[np.argsort(-res[o])]

        sent_scores = res[o_sort]
        top_sents = [self._sentences[o] for o in o_sort]

        result = []
        for score, sent in zip(sent_scores, top_sents):
            if isinstance(sent, str):
                result.append({
                    'score': score,
                    'sentence': sent,
                })
            else:
                result.append({
                    'score': score,
                    'sentence': sent['sentence'],
                    'title': sent['title']
                })

        return result

    @staticmethod
    def pprint(top_sents: List[Dict[str, Union[float, str]]]) -> None:
        """Print the results returned by the ranker.

        Parameters
        ----------
        top_sents : List[Dict[str, Union[float, str]]]
            Results returned from ranker
        """

        if 'title' in top_sents[0]:
            headers = ['Rank', 'Sentence', 'Origin', 'Sentence Score']
        else:
            headers = ['Rank', 'Sentence', 'Sentence Score']

        table = prettytable.PrettyTable(headers)
        for i, result in enumerate(top_sents):
            sent = result['sentence']
            sent = sent[:100] + ' ...' if len(sent) > 100 else sent
            if 'title' in result:
                origin = result['title']
                origin = origin[:30] + ' ...' if len(origin) > 30 else origin
                table.add_row([i, sent, origin, '%.5g' % result['score']])
            else:
                table.add_row([i, sent, '%.5g' % result['score']])
        print('Top Results:')
        print(table)

    def _text2vec(self, query: str) -> np.ndarray:
        """Create a tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # Get hashed ngrams
        words = self.parse(normalize(query))
        wids = [murmurhash(w, self._hash_size) for w in words]

        if not wids:
            logger.info('No valid word in %s. Using query "dummy".', query)
            words = self.parse('dummy')
            wids = [murmurhash(w, self._hash_size) for w in words]

        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)

        # Count IDF
        Ns = np.array([self._sent_freqs[w] for w in wids_unique])
        idfs = np.log((self._num_sents - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        data = np.multiply(tfs, idfs)

        vec = np.zeros((1, self._hash_size))
        for wid, tfidf in zip(wids_unique, data):
            vec[0, wid] = tfidf

        return vec
