#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-04-27 12:05:56
# @Last Modified by: Shuailong
# @Last Modified time: 2019-04-27 12:06:11

import logging
from typing import Any, Dict, List, Optional
from overrides import overrides
import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.modules import Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import masked_max

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("csqa-bert-emb")
class CSQABertEmb(Model):
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
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 output_logit: FeedForward,
                 dropout: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._output_logit = output_logit
        self._encoder = encoder

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        if encoder:
            check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                                   "text field embedding dim", "encoder input dim")
            check_dimensions_match(encoder.get_output_dim(), output_logit.get_input_dim(),
                                   "encoder output dim", "output_logit input dim")
        else:
            check_dimensions_match(text_field_embedder.get_output_dim(), output_logit.get_input_dim(),
                                   "text field embedding dim", "output_logit input dim")

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
        """
        Parameters
        ----------
        qa_pairs : Dict[str, torch.LongTensor]
            From a ``ListField``.
        answer_index : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is what we are trying to predict.
            If this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, question and choices for each instance 
            in the batch. The length of this list should be the batch size, and each dictionary 
            should have the keys ``qid``, ``question``, ``choices``, ``question_tokens`` and 
            ``choices_tokens``.

        Returns
        -------
        An output dictionary consisting of the followings.

        qid : List[str]
            A list consisting of question ids.
        answer_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_options=5)`` representing unnormalised log
            probabilities of the choices.
        answer_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_options=5)`` representing probabilities of the
            choices.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embeded = self._text_field_embedder(qa_pairs, num_wrapping_dims=1)
        mask = qa_pairs['mask'].float()
        batch_size, choice_size, seq_len, hidden_size = embeded.size()
        embeded = embeded.view(-1, seq_len, hidden_size)
        mask = mask.view(-1, seq_len)
        if self.dropout:
            embeded = self.dropout(embeded)
        if self._encoder:
            embeded = self._encoder(embeded, mask)
        embeded = embeded.view(batch_size, choice_size, seq_len, -1)
        mask = mask.view(batch_size, choice_size, seq_len)
        embeded = masked_max(embeded, mask.unsqueeze(-1), dim=2)

        # the final MLP -- apply dropout to input, and MLP applies to hidden
        answer_logits = self._output_logit(embeded).squeeze(-1)
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
