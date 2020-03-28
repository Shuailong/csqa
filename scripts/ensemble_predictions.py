#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-20 15:29:14
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-20 15:29:18

import json
import statistics as stats
import numpy as np
import argparse
from collections import defaultdict
from collections import Counter
from itertools import groupby

LABELS = ['A', 'B', 'C', 'D', 'E']


def main():
    predictions_by_file = defaultdict(dict)
    for pred_file in args.predictions:
        with open(pred_file) as pred_f:
            for line in pred_f:
                pred_json = json.loads(line)
                qid = pred_json['qid']
                probs = pred_json['answer_probs']
                predictions_by_file[qid][pred_file] = probs
    with open(args.output, 'w') as out_f:
        for qid, predictions in predictions_by_file.items():
            probs = predictions.values()  # N x 5
            if args.mode == 'soft':
                probs_transpose = list(map(list, zip(*probs)))  # 5 x N
                label_probs = [stats.mean(p) for p in probs_transpose]
                prob_max = max(label_probs)
                label_index = [i for i, p in enumerate(
                    label_probs) if p == prob_max]
            else:
                def mode(arr):
                    freqs = groupby(Counter(arr).most_common(), lambda x: x[1])
                    return [val for val, count in next(freqs)[1]]
                label_indices = [[i for i, _p in enumerate(p) if _p == max(p)]
                                 for p in probs]
                label_indices = [i for ii in label_indices for i in ii]
                label_index = mode(label_indices)

            labels = [LABELS[i] for i in label_index]
            label = ';'.join(labels)

            out_f.write(f'{qid},{label}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble predictions')
    parser.add_argument('--predictions', '-p', nargs='+',
                        help='prediction files in json format')
    parser.add_argument('--output', '-o',
                        help='output file in csv format for formal evaluation')
    parser.add_argument('--mode', choices=['hard', 'soft'], default='soft')
    args = parser.parse_args()
    main()
