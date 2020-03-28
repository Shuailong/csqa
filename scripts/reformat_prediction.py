#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-17 11:26:15
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-17 11:26:17

"""
Reformat prediction file into OpenBookQA format in
 https://github.com/allenai/aristo-leaderboard/tree/master/openbookqa/evaluator
for formal evaluation purpose
"""

import json
import argparse

import numpy as np

LABELS = ['A', 'B', 'C', 'D', 'E']


def main():
    with open(args.predictions) as pred_f, open(args.output, 'w') as output_f:
        count = 0
        for line in pred_f:
            pred_dict = json.loads(line)
            qid = pred_dict['qid']
            label_index = [i for i, p in enumerate(pred_dict['answer_probs'])
                           if p == max(pred_dict['answer_probs'])]
            label = [LABELS[i] for i in label_index]
            label = ';'.join(label)
            output_f.write(f'{qid},{label}\n')
            count += 1
        print(f'Write {count} lines into {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reformat predictions into OpenbookQA format')
    parser.add_argument('--predictions', '-i',
                        help='prediction file in jsonl format')
    parser.add_argument('--output', '-o', help='output file in csv format')
    args = parser.parse_args()
    main()
