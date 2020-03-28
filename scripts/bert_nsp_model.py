#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-30 20:13:00
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-30 20:13:08

from typing import List
import argparse
import json
import math

from tqdm import tqdm
import numpy as np

import torch
from torch.nn import functional as F

from pytorch_pretrained_bert.modeling import BertForNextSentencePrediction
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.vocabulary import Vocabulary


LABELS = ('A', 'B', 'C', 'D', 'E')


def read_instances(filename: str) -> List[Instance]:
    """Read samples from `filename` and convert to instances.

    Returns
    -------
    List[Instances]
        List of Allennlp `Instance`s for BERT NSP model
    """
    instances = []
    labels = []
    with open(filename) as input_f:
        for line in input_f:
            sample = json.loads(line)
            question = sample['question']['stem']
            choices = [choice['text']
                       for choice in sample['question']['choices']]
            sample_instances = make_instance(question, choices)
            instances += sample_instances
            if 'answerKey' in sample:
                labels.append(LABELS.index(sample['answerKey']))
    return instances, labels


def make_instance(question: str, choices: List[str]) -> List[Instance]:
    """Given a question and a list of choices text, convert to BERT NSP instances.

    Parameters
    ----------
    question : str
        Question
    choices : List[str]
        List of five choices

    Returns
    -------
    List[Instance]
        List of Allennlp Instances
    """
    question_tokens = TOKENIZER.tokenize(question)
    instances = []
    for choice in choices:
        choice_tokens = TOKENIZER.tokenize(choice)
        tokens = question_tokens + [Token('[SEP]')] + choice_tokens
        instance = Instance({"tokens": TextField(
            tokens, {"bert": WORD_INDEXER})})
        instances.append(instance)
    return instances


def batch(iterable, batch_size=1):
    """Make a batch from a iterable

    Parameters
    ----------
    iterable
        Python iterable such as list
    batch_size : int, optional
        by default 1
    """
    seq_len = len(iterable)
    for ndx in range(0, seq_len, batch_size):
        yield iterable[ndx:min(ndx + batch_size, seq_len)]


def predict(instances: List[Instance]) -> List[float]:
    """Output BERT NSP next sentence probability for a list of instances.

    Parameters
    ----------
    instances : List[Instance]

    Returns
    -------
    List[float]
        BERT NSP scores in range [0, 1].
    """
    scores = []
    for batch_instance in tqdm(batch(instances, batch_size=args.batch_size),
                               total=math.ceil(
                                   len(instances) / args.batch_size),
                               desc='Predicting'):
        batch_ins = Batch(batch_instance)
        batch_ins.index_instances(VOCAB)
        tensor_dict = batch_ins.as_tensor_dict(batch_ins.get_padding_lengths())
        tokens = tensor_dict["tokens"]
        input_ids = tokens['bert'].to(torch.device(f'cuda:{GPU_ID}'))
        token_type_ids = tokens['bert-type-ids'].to(
            torch.device(f'cuda:{GPU_ID}'))
        input_mask = (input_ids != 0).long()
        cls_out = BERT_NEXT_SENTENCE.forward(input_ids=input_ids,
                                             token_type_ids=token_type_ids,
                                             attention_mask=input_mask)
        probs = F.softmax(cls_out, dim=-1)
        next_sentence_score = probs[:, 0].detach().cpu().numpy().tolist()
        scores += next_sentence_score

    return scores


def score2index(scores: List[float]) -> List[int]:
    """Given a list of scores, every 5 scores are scores of a CSQA sample.
    Find the chioce with the max score.

    Parameters
    ----------
    scores : List[float]
        Scores

    Returns
    -------
    List[int]
        Answer indices.
    """
    predict_indices = []
    for batch_score in batch(scores, 5):
        predict_index = np.argmax(batch_score).item()
        predict_indices.append(predict_index)

    return predict_indices


def main():
    instances, labels = read_instances(args.input)
    scores = predict(instances)
    predict_labels = score2index(scores)
    correct = 0
    for label, pred_label in zip(labels, predict_labels):
        if label == pred_label:
            correct += 1
    print(
        f'Accuracy: {correct}/{len(labels)} = {correct/len(labels)*100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSQA using BERT NSP model')
    parser.add_argument('--input', help='input dataset')
    parser.add_argument('--bert-vocab', help='bert vocab file')
    parser.add_argument('--bert-model', help='pretrained bert model')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size for BERT')
    parser.add_argument('--gpu-id', '-g', type=int, default=0, help='GPU ID')

    args = parser.parse_args()

    print('Initialize BERT model...')

    TOKENIZER = WordTokenizer(word_splitter=BertBasicWordSplitter())
    WORD_INDEXER = PretrainedBertIndexer(pretrained_model=args.bert_vocab)
    VOCAB = Vocabulary()
    GPU_ID = args.gpu_id
    BERT_NEXT_SENTENCE = BertForNextSentencePrediction.from_pretrained(
        args.bert_model).to(torch.device(f"cuda:{GPU_ID}"))
    BERT_NEXT_SENTENCE.eval()

    main()
