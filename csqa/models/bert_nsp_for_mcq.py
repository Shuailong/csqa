#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-31 11:08:18
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-31 11:08:30


from typing import Dict, List, Any
from overrides import overrides

import torch
from allennlp.modules import TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("bert_nsp_for_mcq")
class BertNSPForMCQ(Model):
    """
    An AllenNLP Model that runs pretrained BERT with Next Sentence Prediction head,
    and use the NSP score to calculate ranking loss.

    See `csqa/tests/fixtures/bert_nsp_for_mcq.jsonnet`
    for an example of what your config might look like.

    Parameters
    ----------
    vocab : ``Vocabulary``
    """

    def __init__(self,
                 vocab: Vocabulary,
                 bert: TextFieldEmbedder) -> None:
        super().__init__(vocab)

        self.bert = bert
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self._linear = torch.nn.Linear(2, 1)

    def forward(self,  # type: ignore
                qa_pairs: Dict[str, torch.LongTensor],
                answer_index: torch.IntTensor = None,
                metadata: List[Dict[str, Any]  # pylint:disable=unused-argument
                               ] = None
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        qa_pairs : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexer)
        answer_index : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        nsp_logits = self.bert(qa_pairs, num_wrapping_dims=1)
        # nsp_logits: batch, 5, 2
        nsp_probs = torch.nn.functional.softmax(nsp_logits, dim=-1)
        # nsp_probs: batch, 5, 2
        nsp_pos_probs = nsp_probs[..., 0]
        # nsp_pos_probs = self._linear(nsp_probs).squeeze(-1)

        label_probs = torch.nn.functional.softmax(nsp_pos_probs, dim=-1)
        # label_score: batch, 5

        output_dict = {
            "nsp_logits": nsp_logits,
            "nsp_probs": nsp_probs,
            "label_probs": label_probs}

        if answer_index is not None:
            answer_index = answer_index.squeeze(-1)
            loss = self._loss(nsp_pos_probs, answer_index)
            self._accuracy(nsp_pos_probs, answer_index)
            output_dict["loss"] = loss
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["label_probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i]
                                for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_token_from_index(
                label_idx, namespace="labels")
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics
