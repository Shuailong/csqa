#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-16 13:42:54
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-16 13:43:04

import json
import argparse

from termcolor import colored

from csqa.modules.retriever import SearchEngine


def main():
    print('Initializing search engine...')
    search_engine = SearchEngine(args.index_dir)
    print('Start...')
    with open(args.input) as in_f:
        for line in in_f:
            sample = json.loads(line)
            queries = []
            for choice in sample['question']['choices']:
                query = sample['question']['stem'] + ' ' + choice['text']
                queries.append(query)
            results = search_engine.batch_search(queries)
            for query, result in zip(queries, results):
                print(f'Query: {colored(query, "yellow")}')
                SearchEngine.pprint(result)
            user_input = input()
            if user_input == 'exit':
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Search Wiki Evidence for CSQA instances')
    parser.add_argument('--input', '-i', help='input file')
    parser.add_argument('--output', '-o', help='output file')
    parser.add_argument('--index-dir', help='wikipedia index')
    parser.add_argument('--doc-max', type=int, default=5,
                        help='max docs returned by search engine')
    parser.add_argument('--sent-max', type=int, default=5,
                        help='max sentences returned by ranker')
    args = parser.parse_args()
    main()
