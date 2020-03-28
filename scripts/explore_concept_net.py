#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-16 16:50:43
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-16 16:50:59

import argparse
import readline
import requests


def main(args):
    try:
        while True:
            query = input('>> ')
            obj = requests.get('http://api.conceptnet.io/c/en/' + query).json()
            for edge in obj['edges']:
                if edge['surfaceText']:
                    if 'is a translation of' not in edge['surfaceText']:
                        print(edge['surfaceText'])
            print('-'*100)
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explore Conceptnet API')
    args = parser.parse_args()
    main(args)
