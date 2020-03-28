#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-04-15 14:24:28
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-31 18:15:02
# Modified from https://github.com/allenai/allennlp/blob/master/allennlp/modules/token_embedders/bert_token_embedder.py


"""
A wrapper to use BERT and its NSP head.
"""

from typing import Dict


import torch

from pytorch_pretrained_bert.modeling import BertForNextSentencePrediction

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn import util


class PretrainedBertNSPModel:
    """
    In some instances you may want to load the same BERT model twice
    (e.g. to use as a token embedder and also as a pooling layer).
    This factory provides a cache so that you don't actually have to load the model twice.
    """
    _cache: Dict[str, BertForNextSentencePrediction] = {}

    @classmethod
    def load(cls, model_name: str, cache_model: bool = True) -> BertForNextSentencePrediction:
        if model_name in cls._cache:
            return PretrainedBertNSPModel._cache[model_name]

        model = BertForNextSentencePrediction.from_pretrained(model_name)
        if cache_model:
            cls._cache[model_name] = model

        return model


class BertNSPEmbedder(TokenEmbedder):
    """
    A ``TokenEmbedder`` that produces BERT embeddings for your tokens.
    Should be paired with a ``BertIndexer``, which produces wordpiece ids.

    Most likely you probably want to use ``PretrainedBertNSPEmbedder``
    for one of the named pretrained models, not this base class.

    Parameters
    ----------
    bert_model: ``BertForNextSentencePrediction``
        The BERT model being wrapped.
    """

    def __init__(self,
                 bert_model: BertForNextSentencePrediction) -> None:
        super().__init__()
        self.bert_model = bert_model

    def get_output_dim(self):
        return 2

    def forward(self,
                input_ids: torch.LongTensor,
                token_type_ids: torch.LongTensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : ``torch.LongTensor``
            The (batch_size, ..., max_sequence_length) tensor of wordpiece ids.
        token_type_ids : ``torch.LongTensor``, optional
            If an input consists of two sentences (as in the BERT paper),
            tokens from the first sentence should have type 0 and tokens from
            the second sentence should have type 1.  If you don't provide this
            (the default BertIndexer doesn't) then it's assumed to be all 0s.
        """

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        input_mask = (input_ids != 0).long()

        # input_ids may have extra dimensions, so we reshape down to 2-d
        # before calling the BERT model and then reshape back at the end.
        nsp_logits = self.bert_model(input_ids=util.combine_initial_dims(input_ids),
                                     token_type_ids=util.combine_initial_dims(
                                         token_type_ids),
                                     attention_mask=util.combine_initial_dims(input_mask))
        return nsp_logits


@TokenEmbedder.register("bert-nsp-pretrained")
class PretrainedBertNSPEmbedder(BertNSPEmbedder):
    # pylint: disable=line-too-long
    """
    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .tar.gz file with the model weights.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L41
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    requires_grad : ``bool``, optional (default = False)
        If True, compute gradient of BERT parameters for fine tuning.
    """

    def __init__(self, pretrained_model: str, requires_grad: bool = False) -> None:
        model = PretrainedBertNSPModel.load(pretrained_model)

        for param in model.parameters():
            param.requires_grad = requires_grad

        super().__init__(bert_model=model)
