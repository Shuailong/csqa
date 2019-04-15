#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-04-15 23:23:04
# @Last Modified by: Shuailong
# @Last Modified time: 2019-04-15 23:23:27


import json
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model


@Predictor.register('csqa')
class CSQAPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, jsonline: str) -> JsonDict:
        """
        Make a common sense question answering prediction on the supplied input.
        The supplied input json must contain the key of question id, question stem,
        and corresponding 5 candidate choices.

        Parameters
        ----------
        jsonline: ``str``
            A json line that has the same format as the csqa data file.

        Returns
        ----------
        A dictionary that represents the prediction made by the system.
        """
        return self.predict_json(json.loads(jsonline))

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects json that looks like the original csqa data file.
        """
        qid = json_dict["id"]
        question = json_dict["question"]["stem"]
        choices = [choice['text'] for choice in sorted(
            json_dict["question"]["choices"], key=lambda c: c['label'])]
        answer = None
        instance = self._dataset_reader.text_to_instance(
            qid, question, choices, answer)

        return instance
