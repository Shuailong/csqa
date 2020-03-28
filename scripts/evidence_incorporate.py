#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-12 14:42:12
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-12 14:42:15

from typing import List, Dict, Any
import json
import argparse


def read_score(filename: str) -> List[bool]:
    decisions = []
    with open(filename) as f:
        for line in f:
            jsondict = json.loads(line)
            confidence = jsondict['probs'][1]
            decisions.append(confidence > 0.5)
    return decisions


def read_samples(filename: str) -> List[Dict[str, Any]]:
    samples = []
    with open(filename) as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)
    return samples


def main():
    decisions = read_score(args.evidence_score)
    print(f'Read {len(decisions)} decisions.')
    samples = read_samples(args.input)
    print(f'Read {len(samples)} samples.')
    sel_count = 0
    decision_count = 0
    hit_count = 0
    for sample in samples:
        key = sample['answerKey']
        for choice in sample['question']['choices']:
            if 'evidence_ranked' in choice and choice['evidence_ranked']:
                if decisions[decision_count]:
                    choice['evidence_selected'] = choice['evidence_ranked'][0]
                    sel_count += 1
                decision_count += 1
            if key == choice['label'] and 'evidence_selected' in choice:
                hit_count += 1
    assert decision_count == len(decisions)
    with open(args.output, 'w') as output_f:
        for sample in samples:
            output_f.write(json.dumps(sample) + '\n')
    print(f'Write {len(samples)} samples with {sel_count} evidences selected.')
    print(f'{hit_count} samples hit. hit ratio: {hit_count/len(samples)*100:.2f}%.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Use the output of evidence selector to decide whether to use the evidence or not.')
    parser.add_argument('--input', '-i', help='csqa dataset with evidences')
    parser.add_argument('--evidence-score', '-e', help='evidence score file')
    parser.add_argument(
        '--output', '-o', help='csqa dataset with selected evidences')
    args = parser.parse_args()
    main()
