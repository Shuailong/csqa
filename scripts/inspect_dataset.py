#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-16 16:40:08
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-16 16:40:12

from collections import Counter
import json
import argparse
import plotille


def main(args):

    hit = 0
    total = 0

    top_evi_scores = []
    correct_top_evi_scores = []
    wrong_top_evi_score = []
    evi_scores = []

    with open(args.input) as f:
        for line in f:
            sample = json.loads(line)

            answer = None

            max_evi_score = max(
                (score for c in sample['question']['choices']
                 for _, score in c['evidence_ranked'])
            )

            for choice in sample['question']['choices']:
                if sample['answerKey'] == choice['label']:
                    answer = choice['text']
                for evi, evi_score in choice['evidence_ranked']:
                    if evi_score == max_evi_score:
                        evi_label = choice['label']
                        max_evi = evi
                    evi_scores.append(evi_score)

            if evi_label == sample['answerKey']:
                hit += 1
                # print('Question:',
                #       sample['question']['stem'])
                # print('Top evidence:', max_evi)
                # print('Top evidence confidence:', max_evi_score)
                # print('Answer:', answer)
                # correct_top_evi_scores.append(max_evi_score)
                # print('-'*100)
            else:
                wrong_top_evi_score.append(max_evi_score)
            total += 1
            top_evi_scores.append(max_evi_score)
    print(f'hit: {hit}, total: {total}, hit/total={hit/total*100:.2f}%')
    # print(plotille.histogram(evi_scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inspect choices of the QA pairs')
    parser.add_argument('input', help='input file')
    args = parser.parse_args()
    main(args)
