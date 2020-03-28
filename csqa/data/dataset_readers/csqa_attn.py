#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-18 17:05:29
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-18 17:07:20

from typing import Dict, List, Union
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


@DatasetReader.register("csqa-attn")
class CSQAReaderAttn(DatasetReader):
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
                 num_evidences: int = 3,
                 lazy: bool = False) -> None:
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
                example = json.loads(line)
                qid = example["id"]
                question = example["question"]["stem"]
                choices = [choice['text'] for choice in sorted(
                    example["question"]["choices"], key=lambda c: c['label'])]
                answer = example['answerKey'] if 'answerKey' in example else None
                if 'evidence_ranked' in example['question']['choices'][0]:
                    choice_evidences = [choice['evidence_ranked'] for choice in sorted(
                        example["question"]["choices"], key=lambda c: c['label'])]
                else:
                    choice_evidences = None
                yield self.text_to_instance(qid, question, choices,
                                            choice_evidences=choice_evidences,
                                            answer=answer)

    @overrides
    def text_to_instance(self,  # type: ignore
                         qid: str,
                         question: str,
                         choices: List[str],
                         choice_evidences: List[Union[str, List[str]]] = None,
                         answer: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        question_tokens = self._tokenizer.tokenize(question)
        fields['question'] = TextField(question_tokens, self._token_indexers)
        choice_tokens = self._tokenizer.batch_tokenize(choices)
        fields['choices'] = ListField([TextField(tokens, self._token_indexers)
                                       for tokens in choice_tokens])
        if choice_evidences:
            choice_evidence_list = []
            for c_evidences in choice_evidences:
                # for each choice
                if c_evidences:
                    c_evidences_sents = [evi for evi,
                                         _ in c_evidences[:self.num_evidences]]
                    c_evi_tokens = self._tokenizer.batch_tokenize(
                        c_evidences_sents)
                else:
                    c_evi_tokens = [[]]
                evidence_list = ListField(
                    [TextField(c_evi_t, self._token_indexers) for c_evi_t in c_evi_tokens])
                choice_evidence_list.append(evidence_list)
            fields['evidence'] = ListField(choice_evidence_list)

        if answer:
            fields['answer_index'] = IndexField(
                self.LABELS.index(answer), fields['choices'])

        metadata = {"qid": qid,
                    "question": question,
                    "choices": choices,
                    "question_tokens": [x.text for x in question_tokens],
                    "choices_tokens": [[x.text for x in tokens] for tokens in choice_tokens]}
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
