#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-04-16 10:58:12
# @Last Modified by: Shuailong
# @Last Modified time: 2019-04-16 10:58:21

import json
import argparse


LABELS = 'ABCDE'


def main(args):
    with open(args.input_file) as fin, open(args.output_file, 'w') as fout:
        for line in fin:
            jsondict = json.loads(line)
            qid = jsondict['qid']
            probs = jsondict['answer_probs']
            max_prob = max(probs)
            answer_indexs = [i for i in range(5) if probs[i] == max_prob]
            answer_labels = [LABELS[i] for i in answer_indexs]
            fout.write(f"{qid},{';'.join(answer_labels)}\n")


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert allennlp prediction logits output into answer index')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('input_file', help='input jsonl file')
    parser.add_argument('output_file', help='output csv file')
    args = parser.parse_args()
    main(args)
