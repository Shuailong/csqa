#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-11 14:01:04
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-11 14:01:21


import logging
from typing import Dict, Optional
from overrides import overrides
import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("evidence-selector")
class EvidenceSelector(Model):
    """
    Use BERT model to classify whether the evidence of a question is pertinent or not.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``qa_pairs`` ``TextFields`` we get as input to the model.
    dropout : ``float``, optional (default=0.1)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    """

    def __init__(self, vocab: Vocabulary,
                 bert: TextFieldEmbedder,
                 classifier: FeedForward,
                 dropout: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._bert = bert
        self._classifier = classifier

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self._pooler = FeedForward(input_dim=bert.get_output_dim(),
                                   num_layers=1,
                                   hidden_dims=bert.get_output_dim(),
                                   activations=torch.tanh)

        check_dimensions_match(bert.get_output_dim(), classifier.get_input_dim(),
                               "bert embedding dim", "classifier input dim")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
            # pylint: disable=arguments-differ
        """ Parameters
        ----------
        tokens: Dict[str, torch.LongTensor]
        From a ``TextField``.
        label: ``torch.IntTensor``, optional
        From an ``LabelField``.  This is what we are trying to predict.
        If this is given, we will compute a loss that gets included in the output dictionary.
        Returns
        -------
        An output dictionary consisting of the followings.
        logits: torch.FloatTensor
        A tensor of shape ``(batch_size, 2)`` representing unnormalised log
        probabilities of the evidence selection confidence.
        probs: torch.FloatTensor
        A tensor of shape ``(batch_size, 2)`` representing probabilities of the
        evidence selection confidence.
        loss: torch.FloatTensor, optional
        A scalar loss to be optimised. 
        """

        # batch, seq_len
        cls_hidden = self._bert(tokens)
        # batch, seq_len, hidden
        cls_hidden = cls_hidden[:, 0, :]
        cls_hidden = self._pooler(cls_hidden)
        # batch, hidden

        if self.dropout:
            cls_hidden = self.dropout(cls_hidden)

        # the final MLP -- apply dropout to input, and MLP applies to hidden
        logits = self._classifier(cls_hidden)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {"logits": logits,
                       "probs": probs}

        if label is not None:
            label = label.squeeze(-1)
            loss = self._loss(logits, label)
            self._accuracy(logits, label)
            output_dict["loss"] = loss
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
