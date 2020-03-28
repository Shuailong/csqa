#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-22 15:22:34
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-22 15:22:37

"""Create subsets of training set for bagging.
"""

import os
import json
import random
import argparse


def main():
    print(f'Seed: {args.random_seed}')
    with open(args.dataset) as data_f:
        ids = set()
        samples = data_f.readlines()
        for sample in samples:
            ids.add(json.loads(sample)['id'])
    print(f'Read {len(samples)} samples from {args.dataset}')

    output_files = [os.path.splitext(args.dataset)[0] + f'_sub_{i}' +
                    os.path.splitext(args.dataset)[1]
                    for i in range(args.n_estimators)]
    subset_ids = set()
    for output_file in output_files:
        sub_samples = random.choices(samples, k=args.sub_samples)
        for sample in sub_samples:
            subset_ids.add(json.loads(sample)['id'])
        with open(output_file, 'w') as out_f:
            out_f.writelines(sub_samples)
        print(f'Write {len(sub_samples)} samples into {output_file}.')

    out_of_bagging = 0
    for _id in ids:
        if _id not in subset_ids:
            out_of_bagging += 1
    print(f'{out_of_bagging} ({out_of_bagging/len(samples)*100:.2f}%) out of bagging samples.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create serveral subsets of training data for bagging')
    parser.add_argument('--dataset', '-d', help='train data')
    parser.add_argument('--n_estimators', '-n', type=int, default=6)
    parser.add_argument('--sub_samples', '-r', type=float, default=5000)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.random_seed)
    main()
