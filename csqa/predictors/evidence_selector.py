#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-12 10:20:29
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-12 10:20:39

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model


@Predictor.register('evidence-selector')
class EvidenceSelectorPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        fields = line.split('||')
        if len(fields) == 3:
            question, choice, evidence = fields
        elif len(fields) > 3:
            question, choice, evidence = fields[:3]
        return {
            'question': question,
            'choice': choice,
            'evidence': evidence
        }

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Converts a JSON object into an :class:`~allennlp.data.instance.Instance`
        and a ``JsonDict`` of information which the ``Predictor`` should pass through,
        such as tokenised inputs.
        """
        question = json_dict["question"]
        choice = json_dict["choice"]
        evidence = json_dict["evidence"]
        return self._dataset_reader.text_to_instance(question, choice, evidence)
