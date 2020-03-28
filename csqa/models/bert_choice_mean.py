#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-14 15:25:07
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-14 15:25:14


import logging
from typing import Any, Dict, List, Optional
from overrides import overrides
import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("csqa-choice-mean")
class CSQAChoiceMean(Model):
    """
    This class implements baseline Bert model for commonsenseqa dataset descibed in NAACL 2019 paper
    CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge [https://arxiv.org/abs/1811.00937].
    In this set-up, a single instance is a list of question answer pairs, and an answer index to indicate
    which one is correct.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``qa_pairs`` ``TextFields`` we get as input to the model.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    """

    def __init__(self, vocab: Vocabulary,
                 bert: TextFieldEmbedder,
                 classifier: FeedForward,
                 dropout: float = 0.1,
                 pooling: str = 'mean',
                 pooler: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.pooling = pooling
        self._bert = bert
        self._classifier = classifier

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        if pooling == 'cat':
            bert_out_dim = bert.get_output_dim() * 2
        else:
            bert_out_dim = bert.get_output_dim()

        self.pooler = pooler
        if pooler:
            self._pooler = FeedForward(input_dim=bert_out_dim,
                                       num_layers=1,
                                       hidden_dims=bert_out_dim,
                                       activations=torch.tanh)

        check_dimensions_match(bert_out_dim, classifier.get_input_dim(),
                               "bert embedding dim", "classifier input dim")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                qa_pairs: Dict[str, torch.LongTensor],
                answer_index: torch.IntTensor = None,
                metadata: List[Dict[str, Any]
                               ] = None  # pylint:disable=unused-argument
                ) -> Dict[str, torch.Tensor]:
            # pylint: disable=arguments-differ
        """ Parameters
        ----------
        qa_pairs: Dict[str, torch.LongTensor]
        From a ``ListField``.
        answer_index: ``torch.IntTensor``, optional
        From an ``IndexField``.  This is what we are trying to predict.
        If this is given, we will compute a loss that gets included in the output dictionary.
        metadata: ``List[Dict[str, Any]]``, optional
        If present, this should contain the question ID, question and choices for each instance
        in the batch. The length of this list should be the batch size, and each dictionary
        should have the keys ``qid``, ``question``, ``choices``, ``question_tokens`` and
        ``choices_tokens``.

        Returns
        -------
        An output dictionary consisting of the followings.

        qid: List[str]
        A list consisting of question ids.
        answer_logits: torch.FloatTensor
        A tensor of shape ``(batch_size, num_options=5)`` representing unnormalised log
        probabilities of the choices.
        answer_probs: torch.FloatTensor
        A tensor of shape ``(batch_size, num_options=5)`` representing probabilities of the
        choices.
        loss: torch.FloatTensor, optional
        A scalar loss to be optimised. 
        """

        # batch, 10, seq_len
        cls_hidden = self._bert(qa_pairs, num_wrapping_dims=1)
        # batch, 10, seq_len, hidden
        cls_hidden = cls_hidden[:, :, 0, :]
        # batch, 10, hidden

        if self.pooling == 'mean':
            A = cls_hidden[:, 0, :] + cls_hidden[:, 1, :]
            B = cls_hidden[:, 2, :] + cls_hidden[:, 3, :]
            C = cls_hidden[:, 4, :] + cls_hidden[:, 5, :]
            D = cls_hidden[:, 6, :] + cls_hidden[:, 7, :]
            E = cls_hidden[:, 8, :] + cls_hidden[:, 9, :]
        elif self.pooling == 'max':
            A = torch.stack([cls_hidden[:, 0, :],
                             cls_hidden[:, 1, :]], dim=1).max(dim=1)[0]
            B = torch.stack([cls_hidden[:, 2, :],
                             cls_hidden[:, 3, :]], dim=1).max(dim=1)[0]
            C = torch.stack([cls_hidden[:, 4, :],
                             cls_hidden[:, 5, :]], dim=1).max(dim=1)[0]
            D = torch.stack([cls_hidden[:, 6, :],
                             cls_hidden[:, 7, :]], dim=1).max(dim=1)[0]
            E = torch.stack([cls_hidden[:, 8, :],
                             cls_hidden[:, 9, :]], dim=1).max(dim=1)[0]
        elif self.pooling == 'cat':
            A = torch.cat([cls_hidden[:, 0, :], cls_hidden[:, 1, :]], dim=1)
            B = torch.cat([cls_hidden[:, 2, :], cls_hidden[:, 3, :]], dim=1)
            C = torch.cat([cls_hidden[:, 4, :], cls_hidden[:, 5, :]], dim=1)
            D = torch.cat([cls_hidden[:, 6, :], cls_hidden[:, 7, :]], dim=1)
            E = torch.cat([cls_hidden[:, 8, :], cls_hidden[:, 9, :]], dim=1)

        cls_hidden = torch.stack([A, B, C, D, E], dim=1)

        if self.pooler:
            cls_hidden = self._pooler(cls_hidden)
            # batch, 5, hidden

        if self.dropout:
            cls_hidden = self.dropout(cls_hidden)

        # the final MLP -- apply dropout to input, and MLP applies to hidden
        answer_logits = self._classifier(cls_hidden).squeeze(-1)
        answer_probs = torch.nn.functional.softmax(answer_logits, dim=-1)
        qids = [m['qid'] for m in metadata]
        output_dict = {"answer_logits": answer_logits,
                       "answer_probs": answer_probs,
                       "qid": qids}

        if answer_index is not None:
            answer_index = answer_index.squeeze(-1)
            loss = self._loss(answer_logits, answer_index)
            self._accuracy(answer_logits, answer_index)
            output_dict["loss"] = loss
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
