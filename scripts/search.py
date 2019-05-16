#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-15 02:13:52
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-15 02:13:57


import argparse
import readline  # pylint: disable=unused-import

from csqa.modules.retriever import SearchEngine


def main_loop():
    while True:
        try:
            input_query = input(">> ")
            if input_query == 'exit':
                break
            if not input_query:
                continue
            results = search_engine.search(input_query)
            SearchEngine.pprint(results)

        except (KeyboardInterrupt, EOFError):
            print()
            break
        except Exception as exception:
            raise exception


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A Search Engine to search Wikipedia')
    parser.add_argument(
        '--index-dir', default='/Users/handsome/Workspace/data/wikipedia/index', help='lucene index')
    args = parser.parse_args()

    search_engine = SearchEngine(index_dir=args.index_dir)
    main_loop()
