#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-16 14:09:01
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-16 14:09:06

from typing import List, Dict, Union

import prettytable

from .search_docs import LuceneSearch
from .search_sentence import TfidfSentenceRanker
from .sentence_splitter import SpacySentenceSplitter


class SearchEngine():
    """
    A search engine to retrieve wikipedia articles and return most relevant
    sentences of a given query.

    Parameters
    ----------
    index_dir: ``str``
        Lucene search index
    num_search_workers: ``int`` (default=``5``)
        Used for multithreading to speedup retrieving process
    doc_max: ``int`` (default=``5``)
        Maximum number of docs returned by the search engine.
    sent_max: ``int`` (default=``5``)
        Maximum number of sentences returned by the sentence ranker.
    """

    def __init__(self,
                 index_dir: str,
                 num_search_workers: int = 5,
                 doc_max: int = 5,
                 sent_max: int = 5) -> None:
        self.doc_max = doc_max
        self.sent_max = sent_max
        self.searcher = LuceneSearch(
            index_dir, num_search_workers=num_search_workers)
        self.sentence_splitter = SpacySentenceSplitter()

    def search(self, query: str) -> List[Dict[str, Union[float, str]]]:
        """
        Given a query, return most relevant sentences and its titles

        Parameters
        ----------
        query : str
            Input query

        Returns
        -------
        top_sents:  ``List[Dict[str, Union[float, str]]]``
            A list of sentences scores, sentences and its wikipedia article title
            e.g.
            [{
                'score': 2.23,
                'sentence':'The quick brown fox jumps over the lazy dog.',
                'title': 'Famous sentence'
             },
             {
                 'score': 2.0,
                 'sentence':'ultimate question of life, the universe, and everything',
                 'title' :'The Hitchhiker's Guide to the Galaxy'
              }
            ]
        """
        results = self.searcher.search(query, doc_max=self.doc_max)
        sentences = self._extract_sentences(results)
        ranker = TfidfSentenceRanker(sentences)
        top_sents = ranker.closest_sents(query, sent_max=self.sent_max)
        return top_sents

    def batch_search(self, queries: List[str]) -> List[List[Dict[str, Union[float, str]]]]:
        """
        Given a batch of queries, return most relevant sentences and its
            titles for each query.

        Parameters
        ----------
        queries : List[str]
            A list of queries.

        Returns
        -------
        List[List[Dict[str, Union[float, str]]]]
            batch results
        """
        batch_results = self.searcher.batch_search(
            queries, doc_max=self.doc_max)
        batch_res = []
        for query, results in zip(queries, batch_results):
            sentences = self._extract_sentences(results)
            ranker = TfidfSentenceRanker(sentences)
            sents = ranker.closest_sents(query, sent_max=self.sent_max)
            batch_res.append(sents)
        return batch_res

    @staticmethod
    def pprint(top_sents: List[Dict[str, Union[float, str]]]) -> None:
        """Print the results returned by the search engine.

        Parameters
        ----------
        top_sents : List[Dict[str, Union[float, str]]]
            Results returned from ranker
        """
        if top_sents[0]['title']:
            headers = ['Rank', 'Sentence', 'Origin', 'Sentence Score']
        else:
            headers = ['Rank', 'Sentence', 'Sentence Score']

        table = prettytable.PrettyTable(headers)
        for i, result in enumerate(top_sents):
            sent = result['sentence']
            sent = sent[:100] + ' ...' if len(sent) > 100 else sent
            if result['title']:
                origin = result['title']
                origin = origin[:30] + ' ...' if len(origin) > 30 else origin
                table.add_row([i, sent, origin, '%.5g' % result['score']])
            else:
                table.add_row([i, sent, '%.5g' % result['score']])
        print('Top Results:')
        print(table)

    def _extract_sentences(self,
                           search_results: List[Dict[str, Union[float, str]]]
                           ) -> List[Dict[str, str]]:
        sentences = []
        for res in search_results:
            sents = self.sentence_splitter.split_sentences(res['text'])
            for sent in sents:
                sentences.append({
                    'sentence': sent,
                    'title': res['title']
                })
        return sentences
