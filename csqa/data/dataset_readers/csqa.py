#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-04-12 19:40:32
# @Last Modified by: Shuailong
# @Last Modified time: 2019-04-12 20:35:05

from typing import Dict, List
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, MetadataField, ListField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("csqa")
class CSQAReader(DatasetReader):
    """
    Reads a file from the CommonsenseQA dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """
    LABELS = ['A', 'B', 'C', 'D', 'E']

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as snli_file:
            logger.info(
                "Reading CSQA instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                example = json.loads(line)
                qid = example["id"]
                question = example["question"]["stem"]
                choices = [choice['text'] for choice in sorted(
                    example["question"]["choices"], key=lambda c: c['label'])]
                answer = example['answerKey'] if 'answerKey' in example else None
                yield self.text_to_instance(qid, question, choices, answer)

    @overrides
    def text_to_instance(self,  # type: ignore
                         qid: str,
                         question: str,
                         choices: List[str],
                         answer: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        question_tokens = self._tokenizer.tokenize(question)
        choice_tokens = self._tokenizer.batch_tokenize(choices)
        qa_pair_tokens = [[Token("[CLS]")] + question_tokens + [Token("[SEP]")] + tokens + [Token("[SEP]")]
                          for tokens in choice_tokens]
        qa_pairs_field = ListField(
            [TextField(tokens, self._token_indexers) for tokens in qa_pair_tokens])
        if answer:
            fields['answer_index'] = IndexField(
                self.LABELS.index(answer), qa_pairs_field)
        fields['qa_pairs'] = qa_pairs_field

        metadata = {"qid": qid,
                    "question": question,
                    "choices": choices,
                    "question_tokens": [x.text for x in question_tokens],
                    "choices_tokens": [[x.text for x in tokens] for tokens in choice_tokens]}
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
