#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-17 21:23:55
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-17 21:24:13

import json
import argparse


def main():
    samples = {}
    evaluate = False
    with open(args.dataset) as df:
        for line in df:
            sample = json.loads(line)
            if not evaluate and 'answerKey' in sample:
                evaluate = True
            samples[sample['id']] = sample

    print(f'{len(samples)} samples loaded.')

    disagree_preds = []
    both_failure = 0
    with open(args.prediction1) as f1, open(args.prediction2) as f2:
        for line1, line2 in zip(f1, f2):
            qid1, pred1 = line1.strip().split(',')
            qid2, pred2 = line2.strip().split(',')
            assert qid1 == qid2
            if evaluate:
                answer = samples[qid1]['answerKey']
                if answer not in (pred1, pred2):
                    both_failure += 1
            if pred1 != pred2:
                disagree_preds.append((qid1, pred1, pred2))
    print(f'{len(disagree_preds)} predictions disagree.')
    if evaluate:
        print(f'{both_failure} both failure predictions.')

    correct1, correct2 = 0, 0
    if args.output_txt:
        with open(args.output_txt, 'w') as out_f:
            for q_no, (qid, pred1, pred2) in enumerate(disagree_preds):
                assert qid in samples
                jsondict = samples[qid]
                question = jsondict['question']['stem']
                choices = [(d['label'], d['text'])
                           for d in jsondict['question']['choices']]
                out_f.write(f'{q_no}. {question}\n')
                for label, text in choices:
                    out_f.write(f'{label}: {text}  ')
                out_f.write('\n')
                out_f.write(f'Pred1 answer: {pred1}\n')
                out_f.write(f'Pred2 answer: {pred2}\n')

                if evaluate:
                    answer = jsondict['answerKey']
                    out_f.write(f'Correct answer: {answer}\n')
                    if answer in pred1.split(';'):
                        correct1 += 1
                    if answer in pred2.split(';'):
                        correct2 += 1
                out_f.write('-' * 100 + '\n')
    if args.output_csv:
        with open(args.output_csv, 'w') as out_f:
            for q_no, (qid, pred1, pred2) in enumerate(disagree_preds):
                out_f.write(f'{q_no},{pred1},{pred2},\n')

    if evaluate:
        print(
            f'Among disagreement, correct1 = {correct1}, correct2 = {correct2}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare different test predictions and output difference')
    parser.add_argument('--prediction1', '-p1',
                        help='prediction file in csv format')
    parser.add_argument('--prediction2', '-p2',
                        help='another prediction file in csv format')
    parser.add_argument('--dataset', '-d', help='original dataset file')
    parser.add_argument('--output_txt', '-o', help='output file')
    parser.add_argument('--output_csv', '-v', help='output file for eval')
    args = parser.parse_args()
    main()
