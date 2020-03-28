#!/usr/bin/env python3
# pylint: disable=import-error
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-14 19:56:28
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-14 21:35:11

from typing import List, Dict, Union
import os
import logging
from multiprocessing.pool import ThreadPool

from tqdm import tqdm
from termcolor import colored
import prettytable

import lucene
from java.nio.file import Paths
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import DirectoryReader, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.standard import StandardAnalyzer

from .doc_db import DocDB


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class LuceneSearch():
    """Index and search docs.

    Parameters
    ----------
    index_dir : str
        Index of the documents produced by Lucene
    db_path: str
        File path of the SQLlite database containing articles of wikipedia dump.(from DrQA)
    num_search_workers: int (optional), default=8
        Workers to use to accelerate searching.
    """

    def __init__(self,
                 index_dir: str,
                 db_path: str = None,
                 num_search_workers: int = 8) -> None:

        self.env = lucene.getVMEnv()  # pylint: disable=no-member
        if not self.env:
            self.env = lucene.initVM(initialheap='28g',  # pylint: disable=no-member
                                     maxheap='28g',
                                     vmargs=['-Djava.awt.headless=true'])

        self.num_search_workers = num_search_workers

        if not os.path.exists(index_dir):
            self.doc_db = DocDB(db_path=db_path)
            logger.info('Creating index at %s', index_dir)
            self._create_index(index_dir)

        fs_dir = MMapDirectory(Paths.get(index_dir))
        self.searcher = IndexSearcher(DirectoryReader.open(fs_dir))
        self.analyzer = StandardAnalyzer()
        self.pool = ThreadPool(processes=num_search_workers)

    def _create_index(self, index_dir: str) -> None:
        """Index documents

        Parameters
        ----------
        index_dir : str
            The dir to store index
        """
        os.mkdir(index_dir)

        TITLE_FIELD = FieldType()  # pylint: disable=invalid-name
        TITLE_FIELD.setStored(True)
        TITLE_FIELD.setIndexOptions(IndexOptions.DOCS)

        TEXT_FIELD = FieldType()  # pylint: disable=invalid-name
        TEXT_FIELD.setStored(True)
        TEXT_FIELD.setIndexOptions(
            IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        fs_dir = MMapDirectory(Paths.get(index_dir))
        writer_config = IndexWriterConfig(StandardAnalyzer())
        writer_config.setRAMBufferSizeMB(16384.0)  # 14g
        self.writer = IndexWriter(fs_dir, writer_config)
        logger.info("%d docs in index", self.writer.numDocs())
        logger.info("Indexing documents...")

        doc_ids = self.doc_db.get_doc_ids()
        for doc_id in tqdm(doc_ids, total=len(doc_ids)):
            text = self.doc_db.get_doc_text(doc_id)

            doc = Document()
            doc.add(Field("title", doc_id, TITLE_FIELD))
            doc.add(Field("text", text, TEXT_FIELD))

            self.writer.addDocument(doc)

        logger.info("Indexed %d docs.", self.writer.numDocs())
        self.writer.forceMerge(1)  # to increase search performance
        self.writer.close()

    def _search_multithread(self,
                            queries: List[str],
                            doc_max: int) -> List[List[Dict[str, Union[float, str]]]]:
        args = [(query, doc_max) for query in queries]
        queries_results = self.pool.starmap(
            self._search_multithread_part, args)
        return queries_results

    def _search_multithread_part(self,
                                 query: str,
                                 doc_max: int) -> List[Dict[str, Union[float, str]]]:
        if not self.env.isCurrentThreadAttached():
            self.env.attachCurrentThread()

        try:
            query = QueryParser('text', self.analyzer).parse(
                QueryParser.escape(query))
        except Exception as exception:  # pylint: disable=broad-except
            logger.warning(
                colored(f'{exception}: {query}, use query dummy.'), 'yellow')
            query = QueryParser('text', self.analyzer).parse('dummy')

        query_results = []
        hits = self.searcher.search(query, doc_max)

        for hit in hits.scoreDocs:
            doc = self.searcher.doc(hit.doc)

            query_results.append({
                'score': hit.score,
                'title': doc['title'],
                'text': doc['text']
            })

        if not query_results:
            logger.warning(
                colored(f'WARN: search engine returns no results for query: {query}.', 'yellow'))

        return query_results

    def _search_singlethread(self,
                             queries: List[str],
                             doc_max: int) -> List[List[Dict[str, Union[float, str]]]]:
        queries_result = []
        for query in queries:
            try:
                query = QueryParser('text', self.analyzer).parse(
                    QueryParser.escape(query))
            except Exception as exception:   # pylint: disable=broad-except
                logger.warning(
                    colored(f'{exception}: {query}, use query dummy.'), 'yellow')
                query = QueryParser('text', self.analyzer).parse('dummy')

            query_results = []
            hits = self.searcher.search(query, doc_max)

            for hit in hits.scoreDocs:
                doc = self.searcher.doc(hit.doc)

                query_results.append({
                    'score': hit.score,
                    'title': doc['title'],
                    'text': doc['text']
                })

            if not query_results:
                logger.warning(
                    colored(f'WARN: search engine returns no results for query: {query}.', 'yellow'))

            queries_result.append(query_results)

        return queries_result

    def search(self,
               query: str,
               doc_max: int = 20) -> List[Dict[str, Union[float, str]]]:
        """Search a given query.

        Parameters
        ----------
        query : str
            Anything you want to search
        doc_max : int
            Maximum number of result to return

        Returns
        -------
        Tuple[Any]
            Search results.
        """
        return self.batch_search([query], doc_max=doc_max)[0]

    def batch_search(self,
                     queries: List[str],
                     doc_max: int = 20) -> List[List[Dict[str, Union[float, str]]]]:
        """
        Search a list of queries.

        Parameters
        ----------
        queries : List[str]
            queries list
        doc_max : int, optional, default=20
            maximum number of docs returned by the search engine.

        Returns
        -------
        List[Tuple[Any]]
            Result returned by the search engine.
        """
        if self.num_search_workers > 1:
            result = self._search_multithread(queries, doc_max)
        else:
            result = self._search_singlethread(queries, doc_max)

        return result

    @staticmethod
    def pprint(search_result: List[Dict[str, Union[float, str]]]) -> None:
        """Print the results returned by the doc searcher.

        Parameters
        ----------
        search_result : List[Dict[str, Union[float, str]]]
            Results returned from ranker
        """

        headers = ['Rank', 'Title', 'Text', 'Score']
        table = prettytable.PrettyTable(headers)
        for i, result in enumerate(search_result):
            text, title = result['text'], result['title']
            text = text[:100] + ' ...' if len(text) > 100 else text
            title = title[:30] + ' ...' if len(title) > 30 else title
            table.add_row([i, title, text, '%.5g' % result['score']])
        print('Top Results:')
        print(table)
