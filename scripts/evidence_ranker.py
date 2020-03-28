#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-17 13:15:32
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-17 13:15:37

from typing import Dict, Set, List, Tuple
import argparse
import json
import random
from functools import partial

from tqdm import tqdm
import numpy as np

from allennlp.data import Instance


def rank_random(question: str, evidences: List[str]) -> List[Tuple[str, float]]:  # pylint: disable=unused-argument
    """Rank a list of evidence sentences according to relevancy with the question
    At random
    Parameters
    ----------
    question : str
        Question
    evidences : List[str]
        A list of evidences gathered

    Returns
    -------
    List[Tuple[str, float]]
        A list of ranked evidences
    Issue
    Just a proof of concept
    """
    random.shuffle(evidences)
    evidences_score = [0 for _ in evidences]
    return list(zip(evidences, evidences_score))


def rank_jaccard(question: str, evidences: List[str]) -> List[Tuple[str, float]]:
    """Rank a list of evidence sentences according to relevancy with the question
    Metrics: jaccard distance of the sets of words between question and evidence
    Parameters
    ----------
    question : str
        Question
    evidences : List[str]
        A list of evidences gathered

    Returns
    -------
    List[Tuple[str, float]]
        A list of ranked evidences
    """
    question_tokens = set(question.split())
    jaccard_scores = []
    for evi in evidences:
        evi_tokens = set(evi.split())
        jaccard_score = len(question_tokens & evi_tokens) / \
            len(question_tokens | evi_tokens)
        jaccard_scores.append(-jaccard_score)
    o_sort = np.argsort(jaccard_scores)
    evidences = [evidences[o] for o in o_sort]
    evidences_score = [-jaccard_scores[o] for o in o_sort]
    return list(zip(evidences, evidences_score))


def rank_universal(question: str, evidences: List[str]) -> List[Tuple[str, float]]:
    """Rank a list of evidence sentences according to relevancy with the question
    Use Google's Universal Sentence Encoder (large) model.
    Parameters
    ----------
    question : str
        Question
    evidences : List[str]
        A list of evidences gathered

    Returns
    -------
    List[Tuple[str, float]]
        A list of ranked evidences
    Issue
    -----
    Too slow.
    """
    embeddings = SESSION.run(EMBED([question] + evidences))
    embeddings = np.array(embeddings)
    question_emb = embeddings[0]
    evidence_embs = embeddings[1:]
    question_emb = np.expand_dims(question_emb, 0)  # 1 x E
    evidence_embs = evidence_embs.transpose()  # E x N
    score = np.matmul(question_emb, evidence_embs).squeeze(0)
    o_sort = np.argsort(-score)
    evidences = [evidences[o] for o in o_sort]
    evidences_score = [score[o].item() for o in o_sort]
    return list(zip(evidences, evidences_score))


def rank_word2vec(question: str, evidences: List[str]) -> List[Tuple[str, float]]:
    """Rank a list of evidence sentences according to relevancy with the question
    Metrics: average of word embedding cosine similarity.

    Parameters
    ----------
    question : str
        Question
    evidences : List[str]
        A list of evidences gathered

    Returns
    -------
    List[Tuple[str, float]]
        A list of ranked evidences
    """
    question_tokens = [t.text for t in TOKENIZER.tokenize(question)]
    question_emb = np.stack([WORD2VEC[word] if word in WORD2VEC else np.zeros(300)
                             for word in question_tokens]).mean(0)

    scores = []
    for evi in evidences:
        evi_tokens = [t.text for t in TOKENIZER.tokenize(evi)]
        evi_emb = np.stack([WORD2VEC[word] if word in WORD2VEC else np.zeros(300)
                            for word in evi_tokens]).mean(0)
        scores.append(-np.dot(question_emb, evi_emb))
    o_sort = np.argsort(scores)
    evidences = [evidences[o] for o in o_sort]
    evidences_score = [-scores[o] for o in o_sort]
    return list(zip(evidences, evidences_score))


def rank_bert(question: str, evidences: List[str], batch_size: int = 8) -> List[Tuple[str, float]]:
    """Rank a list of evidence sentences according to relevancy with the question
    Metrics: bert for next sentence prediction score.

    Parameters
    ----------
    question : str
        Question
    evidences : List[str]
        A list of evidences gathered
    batch_size: int
        Batch size for BERT running

    Returns
    -------
    List[Tuple[str, float]]
        A list of ranked evidences
    """

    def make_instance(question: str, evidences: List[str]) -> List[Instance]:
        question_tokens = TOKENIZER.tokenize(question)
        instances = []
        for evi in evidences:
            evi_tokens = TOKENIZER.tokenize(evi)
            tokens = question_tokens + [Token('[SEP]')] + evi_tokens
            instance = Instance({"tokens": TextField(
                tokens, {"bert": WORD_INDEXER})})
            instances.append(instance)
        return instances

    instances = make_instance(question, evidences)

    def batch(iterable, batch_size=1):
        seq_len = len(iterable)
        for ndx in range(0, seq_len, batch_size):
            yield iterable[ndx:min(ndx + batch_size, seq_len)]

    scores = []
    for batch_instance in batch(instances, batch_size=batch_size):
        batch = Batch(batch_instance)
        batch.index_instances(VOCAB)
        tensor_dict = batch.as_tensor_dict(batch.get_padding_lengths())
        tokens = tensor_dict["tokens"]
        input_ids = tokens['bert'].to(torch.device(f'cuda:{GPU_ID}'))
        token_type_ids = tokens['bert-type-ids'].to(
            torch.device(f'cuda:{GPU_ID}'))
        input_mask = (input_ids != 0).long()
        cls_out = BERT_NEXT_SENTENCE.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=input_mask)
        probs = F.softmax(cls_out, dim=-1)
        next_sentence_score = probs[:, 0].detach().cpu().numpy().tolist()
        next_sentence_score = [-s for s in next_sentence_score]
        scores += next_sentence_score

    o_sort = np.argsort(scores)

    evidences = [evidences[o] for o in o_sort]
    evidences_score = [-scores[o] for o in o_sort]
    return list(zip(evidences, evidences_score))


def load_vocab(filename: str) -> Set[str]:
    """Load vocabulary from dataset files

    Parameters
    ----------
    filename : str
        dataset file in jsonline format

    Returns
    -------
    Set[str]
        List of words
    """
    vocabulary = set()
    with open(filename) as data_f:
        total = sum((1 for _ in data_f))
    with open(filename) as data_f:
        for line in tqdm(data_f, total=total, desc='load vocab'):
            sample = json.loads(line)
            q_tokens = TOKENIZER.tokenize(sample['question']['stem'])
            q_tokens = {t.text for t in q_tokens}
            vocabulary |= q_tokens
            for choice in sample['question']['choices']:
                for evi in choice['evidence']:
                    evi_tokens = TOKENIZER.tokenize(evi)
                    evi_tokens = {t.text for t in evi_tokens}
                    vocabulary |= evi_tokens
    return vocabulary


def load_w2v(filename: str, vocabulary: Set[str]) -> Dict[str, np.ndarray]:
    """
    Load pretrained word vectors for words in ``vocabulary``.

    Parameters
    ----------
    filename : str
        pretrained word2vec vectors trained from GoogleNews
    vocabulary : Set[str]
        vocabulary words

    Returns
    -------
    Dict[str, np.ndarray]
        dictionary with word as key and embedding as value.
    """
    w2v = {}
    with open(filename) as word_vec_f:
        for line in tqdm(word_vec_f, total=3000000, desc='read w2v'):
            tokens = line.split(' ')
            word = tokens[0]
            if word in vocabulary:
                vec = np.array(list(map(float, tokens[1:])))
                w2v[word] = vec
    return w2v


def main():
    """Read csqa instances with evidences and rank the evidences according to several methods
    """
    with open(args.dataset) as input_f:
        total = sum((1 for _ in input_f))
    with open(args.dataset) as input_f, open(args.output, 'w') as out_f:
        for line in tqdm(input_f, total=total, desc='converting'):
            sample = json.loads(line)
            question = sample['question']['stem']
            evidences = []
            for choice in sample['question']['choices']:
                evidences += choice['evidence']
                choice['evidence_ranked'] = RANK_FUNC(
                    question, choice['evidence'])

            out_f.write(json.dumps(sample) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find a single most useful evidence by average of word embeddings')
    parser.add_argument('--dataset', '-i',
                        help='dataset file with evidences from conceptnet')
    parser.add_argument('--output', '-o',
                        help='output dataset file with most useful evidence')
    parser.add_argument('--rank-func', '-f', choices=['random', 'jaccard', 'sentenc', 'w2v', 'bert'],
                        default='random', help='function used for ranking')
    # if w2v
    parser.add_argument('--word2vec', '-v', help='pretrained word2vec vectors')
    # if bert
    parser.add_argument('--bert-vocab', help='bert vocab file')
    parser.add_argument('--bert-model', help='pretrained bert model')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size for BERT')
    parser.add_argument('--gpu-id', '-g', type=int, default=0, help='GPU ID')

    args = parser.parse_args()

    RANK_FUNC_MAP = {'random': rank_random,
                     'jaccard': rank_jaccard,
                     'sentenc': rank_universal,
                     'w2v': rank_word2vec,
                     'bert': rank_bert}
    if args.rank_func not in RANK_FUNC_MAP:
        raise ValueError(
            f"The provided rank function {args.rank_func} is not supported!")
    RANK_FUNC = RANK_FUNC_MAP[args.rank_func]

    if args.rank_func == 'bert' and args.batch_size:
        RANK_FUNC = partial(RANK_FUNC, batch_size=args.batch_size)

    if args.rank_func == 'sentenc':
        import tensorflow_hub as hub
        import tensorflow as tf
        PRETRAINED_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        EMBED = hub.Module(PRETRAINED_MODEL_URL)
        SESSION = tf.Session()
        SESSION.run([tf.global_variables_initializer(),
                     tf.tables_initializer()])
    elif args.rank_func == 'w2v':
        from allennlp.data.tokenizers import WordTokenizer
        TOKENIZER = WordTokenizer()
        VOCAB = load_vocab(args.dataset)
        print(f'Vocab size: {len(VOCAB)}.')
        WORD2VEC = load_w2v(args.word2vec, VOCAB)
        print(
            f'Loaded {len(WORD2VEC)} words. Coverage: { len(WORD2VEC) / len(VOCAB)*100:.2f}%')
    elif args.rank_func == 'bert':
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

        print('Initialize BERT model...')
        TOKENIZER = WordTokenizer(word_splitter=BertBasicWordSplitter())
        WORD_INDEXER = PretrainedBertIndexer(pretrained_model=args.bert_vocab)
        VOCAB = Vocabulary()
        GPU_ID = args.gpu_id
        BERT_NEXT_SENTENCE = BertForNextSentencePrediction.from_pretrained(
            args.bert_model).to(torch.device(f"cuda:{GPU_ID}"))
        BERT_NEXT_SENTENCE.eval()

    main()

    if args.rank_func == 'sentenc':
        SESSION.close()
