#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-14 16:11:19
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-14 16:11:27

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


@DatasetReader.register("csqa-question-mean")
class CSQAReaderQuestionMean(DatasetReader):
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
                 num_evidences: int = 1,
                 gold_evidence: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}
        self.num_evidences = num_evidences
        self.gold_evidence = gold_evidence

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

                if not self.gold_evidence:
                    if 'evidence_ranked' in sample['question']['choices'][0]:
                        evidences_score = [(evi, evi_score)
                                           for choice in sample["question"]["choices"]
                                           for evi, evi_score in choice['evidence_ranked']]
                    else:
                        evidences_score = None
                else:
                    evidences_score = None
                    for choice in sample['question']['choices']:
                        if choice['label'] == answer and 'evidence_ranked' in choice and choice['evidence_ranked']:
                            evidences_score = [choice['evidence_ranked'][0]]
                            break

                if evidences_score:
                    evidence_top = sorted(
                        evidences_score, key=lambda evi: -evi[1])
                    evidence_top_sentences = [
                        sent for sent, _ in evidence_top[:self.num_evidences]]
                else:
                    evidence_top_sentences = None
                yield self.text_to_instance(qid, question, choices,
                                            evidence_top=evidence_top_sentences,
                                            answer=answer)

    @overrides
    def text_to_instance(self,  # type: ignore
                         qid: str,
                         question: str,
                         choices: List[str],
                         evidence_top: [str] = None,
                         answer: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        question_tokens = self._tokenizer.tokenize(question)
        choice_tokens = self._tokenizer.batch_tokenize(choices)

        if evidence_top:
            evidence_tokens = self._tokenizer.batch_tokenize(evidence_top)
            evidence_tokens_flat = [t for evi in evidence_tokens for t in evi]
        else:
            evidence_tokens_flat = []

        qa_pair_tokens = []
        for c_tokens in choice_tokens:
            qa_pair = question_tokens + [Token("[SEP]")] + c_tokens
            qa_pair_tokens.append(qa_pair)
            qa_pair_evidence = question_tokens + evidence_tokens_flat + \
                [Token("[SEP]")] + c_tokens
            qa_pair_tokens.append(qa_pair_evidence)

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
