#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-18 15:32:05
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-29 13:01:18

import logging
from typing import Any, Dict, List, Optional
from overrides import overrides
import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("csqa-bert-reinforce")
class CSQABertReinforce(Model):
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
                 model_type: str = None,  # None, 'first', 'reinforce'
                 dropout: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.model_type = model_type

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

        if model_type is None:
            bert_out_dim = bert.get_output_dim() * 2
        else:
            bert_out_dim = bert.get_output_dim() * 3

        check_dimensions_match(bert_out_dim, classifier.get_input_dim(),
                               "bert embedding dim", "classifier input dim")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                choices: Dict[str, torch.LongTensor],
                evidence: Dict[str, torch.LongTensor],
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

        # batch, seq_len -> batch, seq_len, emb
        question_hidden = self._bert(question)
        batch_size, emb_size = question_hidden.size(0), question_hidden.size(2)
        question_hidden = question_hidden[..., 0, :]  # batch, emb

        # batch, 5, seq_len -> batch, 5, seq_len, emb
        choice_hidden = self._bert(choices, num_wrapping_dims=1)
        choice_hidden = choice_hidden[..., 0, :]  # batch, 5, emb

        if 'first' in self.model_type:
            # batch, 5, evi_num, seq_len -> batch, 5, evi_num, seq_len, emb
            evidence_hidden = self._bert(evidence, num_wrapping_dims=2)
            # evi_num = evidence_hidden.size(2)
            # batch, 5, evi_num, emb
            evidence_hidden = evidence_hidden[..., 0, :]

        if self.dropout:
            question_hidden = self.dropout(question_hidden)
            choice_hidden = self.dropout(choice_hidden)
            if 'first' in self.model_type:
                evidence_hidden = self.dropout(evidence_hidden)

        if 'first' in self.model_type:
            evidence_summary = evidence_hidden[..., 0, :]

        question_hidden = question_hidden.unsqueeze(
            1).expand(batch_size, 5, emb_size)

        cls_hidden = torch.cat([question_hidden, choice_hidden], dim=-1)
        if 'first' in self.model_type:
            cls_hidden = torch.cat([cls_hidden, evidence_summary], dim=-1)

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
