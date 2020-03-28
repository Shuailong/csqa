#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-11 11:40:00
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-11 11:41:08


from typing import Dict
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, Token, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("evidence_selector")
class QEReader(DatasetReader):
    """
    Reads a file from the CommonsenseQA with evidence dataset. Each line in a file
    contains question senntece, evidence sentence, choice text and label.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 skip_indexing: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}
        self.skip_indexing = skip_indexing

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as data_file:
            for line in data_file:
                fields = line.split('||')
                if len(fields) == 3:
                    question, choice, evidence = fields
                    label = None
                elif len(fields) == 4:
                    question, choice, evidence, label = fields
                    label = int(label)
                else:
                    raise ValueError("Unsupported dataset format!")

                yield self.text_to_instance(question, choice, evidence, label=label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question: str,
                         choice: str,  # pylint: disable=unused-argument
                         evidence: str,
                         label: int = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        question_tokens = self._tokenizer.tokenize(question)
        evidence_tokens = self._tokenizer.tokenize(evidence)
        merge_tokens = question_tokens + [Token('[SEP]')] + evidence_tokens
        fields['tokens'] = TextField(merge_tokens, self._token_indexers)

        if label is not None:
            fields['label'] = LabelField(
                label, skip_indexing=self.skip_indexing)

        return Instance(fields)
