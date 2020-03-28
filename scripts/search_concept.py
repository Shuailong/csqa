#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-16 19:19:32
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-16 19:22:47

from typing import List
import json
import math
import argparse
import requests
import plotille
from multiprocessing.pool import ThreadPool

from tqdm import tqdm


def retrieve(query: str, max_evidences: int) -> List[str]:
    query = query.replace(' ', '_')
    evidences = []
    try:
        obj = requests.get('http://api.conceptnet.io/c/en/' + query).json()
        count = 0
        for edge in obj['edges']:
            if edge['surfaceText'] and 'is a translation of' not in edge['surfaceText']:
                evidence = edge['surfaceText'].replace(
                    '[[', '').replace(']]', '')
                evidences.append(evidence)
                count += 1
                if max_evidences > 0 and count >= max_evidences:
                    break
    except Exception as exc:
        print(f"Error processig {query}: {exc}. Continue.")

    return evidences


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def main(args):

    samples = []
    with open(args.input) as in_f:
        for line in in_f:
            samples.append(json.loads(line))
    samples = samples[args.start_index:]
    workers = ThreadPool(processes=args.num_workers)
    evidence_lens = []
    with open(args.output, 'a') as out_f:
        for batch_samples in tqdm(batch(samples, n=args.batch_size), total=math.ceil(len(samples) / args.batch_size)):
            choices_text = [choice['text'] for sample in batch_samples
                            for choice in sample['question']['choices']]
            evidences = workers.starmap(
                retrieve, [(text, args.max_evidences) for text in choices_text])
            for sample, evidences in zip(batch_samples, batch(evidences, 5)):
                for i, choice in enumerate(sample['question']['choices']):
                    choice['evidence'] = evidences[i]
                    evidence_lens.append(len(evidences[i]))
                out_f.write(json.dumps(sample) + '\n')
    print(plotille.histogram(evidence_lens))
    if args.stats_file:
        with open(args.stats_file, 'w') as stats_f:
            stats_f.write(','.join(map(str, evidence_lens)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add ConceptNet evidence into CSQA dataset')
    parser.add_argument('--input', '-i', help='input file')
    parser.add_argument('--output', '-o', help='output file')
    parser.add_argument('--stats-file', '-s', help='stats file')
    parser.add_argument('--num-workers', type=int,
                        default=8, help='multithreading')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--max-evidences', type=int, default=-1,
                        help='max number of evidence sentences')
    parser.add_argument('--start-index', type=int, default=0,
                        help='start from the index')
    args = parser.parse_args()
    main(args)
