#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-29 15:10:46
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-29 15:10:50

from typing import Dict, List
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, MetadataField, ListField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("csqa-reinforce")
class CSQAReaderReinforce(DatasetReader):
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
                 lazy: bool = False,
                 num_evidences: int = 5) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}
        self.num_evidences = num_evidences

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as snli_file:
            logger.info(
                "Reading CSQA instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                sample = json.loads(line)
                qid = sample["id"]
                question = sample["question"]["stem"]
                choices = [choice['text'] for choice in sorted(
                    sample["question"]["choices"], key=lambda c: c['label'])]
                answer = sample['answerKey'] if 'answerKey' in sample else None

                choice_evidences = [choice['evidence_ranked'][:self.num_evidences] for choice in sorted(
                    sample["question"]["choices"], key=lambda c: c['label'])]

                yield self.text_to_instance(qid, question, choices,
                                            choice_evidences,
                                            answer=answer)

    @overrides
    def text_to_instance(self,  # type: ignore
                         qid: str,
                         question: str,
                         choices: List[str],
                         choices_evidences: List[List[str]],
                         answer: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        question_tokens = self._tokenizer.tokenize(question)
        choice_tokens = self._tokenizer.batch_tokenize(choices)

        fields['question'] = TextField(question_tokens, self._token_indexers)
        fields['choices'] = ListField([TextField(c_tokens, self._token_indexers)
                                       for c_tokens in choice_tokens])

        choices_evidences_field = []
        for choice_evidences in choices_evidences:
            choice_evidence_sents = [evi for evi, _ in choice_evidences]
            if not choice_evidence_sents:
                choice_evidence_sents = ['[PAD]']
            evidence_tokens = self._tokenizer.batch_tokenize(
                choice_evidence_sents)
            choice_evidences = ListField([TextField(tokens, self._token_indexers)
                                          for tokens in evidence_tokens])
            choices_evidences_field.append(choice_evidences)
        choices_evidences_field = ListField(choices_evidences_field)

        fields['evidence'] = choices_evidences_field

        if answer:
            fields['answer_index'] = IndexField(
                self.LABELS.index(answer), fields['choices'])

        metadata = {"qid": qid,
                    "question": question,
                    "choices": choices}
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
