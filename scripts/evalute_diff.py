#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-21 15:39:59
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-21 15:40:01

import argparse


def main():
    correct1, correct2, total = 0, 0, 0
    both_correct, both_wrong = 0, 0
    with open(args.input) as input_f:
        for line in input_f:
            fields = line.split(',')
            assert len(fields) == 4
            q_no, pred1, pred2, human = fields
            q_no = q_no.strip()
            pred1 = pred1.strip()
            pred2 = pred2.strip()
            human = human.strip().split(';')
            if pred1 in human:
                correct1 += 1
            if pred2 in human:
                correct2 += 1
            if pred1 in human and pred2 in human:
                both_correct += 1
            elif pred1 not in human and pred2 not in human:
                both_wrong += 1
            total += 1
    print(
        f'Pred1 correct = {correct1}, pred2 correct = {correct2}. Total = {total}.')
    print(f'Both correct: {both_correct}. Both wrong: {both_wrong}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate test difference')
    parser.add_argument('--input', '-i', help='input file in csv format')
    args = parser.parse_args()
    main()
