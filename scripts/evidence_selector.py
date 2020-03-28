#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-11 11:09:05
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-11 11:09:08

"""
Prepare data to train an evidence matching model, which will classifify
whether a question and an evidence sentence are relevant or not.
The evidence sentence of the gold choice are considered relevant.
"""

import json
import argparse
from collections import Counter


def main():
    with open(args.input) as input_f:
        dataset = []
        label_count = Counter()
        for line in input_f:
            sample = json.loads(line)
            question = sample['question']['stem']
            answer_key = sample['answerKey']
            for choice in sample['question']['choices']:
                if 'evidence_ranked' in choice and choice['evidence_ranked']:
                    top_evidence = choice['evidence_ranked'][0][0]
                    label = 1 if choice['label'] == answer_key else 0
                    dataset.append(
                        (question, top_evidence, choice['text'], label))
                    label_count[label] += 1
        print(
            f'Collected {len(dataset)} samples, with positive {label_count[1]} '
            f'and negative {label_count[0]}.'
            f'Postive ratio: {label_count[1]/len(dataset)*100:.2f}%.')

    with open(args.output, 'w') as output_f:
        for sent1, sent2, choice, label in dataset:
            output_f.write(f'{sent1}||{sent2}||{choice}||{label}\n')
            if label:
                print(sent1)
                print(choice)
                print(sent2)
                print('-'*100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classify evidence sentence for a question')
    parser.add_argument(
        '--input', '-i', help='commonsense qa dataset with evidences')
    parser.add_argument(
        '--output', '-o', help='output file with sentence pairs and label')
    args = parser.parse_args()
    main()
