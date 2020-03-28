#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-23 16:54:46
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-23 16:54:48

"""From various conceptnet raw sentences, select english sentences.
Data source: https://github.com/commonsense/conceptnet5/wiki/Downloads
"""

import argparse


def main():
    total, valid = 0, 0
    err = 0
    blank = 0
    with open(args.output, 'w') as out_f:
        for input_file in args.input:
            with open(input_file) as input_f:
                input_f.readline()  # skip header
                for line in input_f:
                    fields = line.strip().split('\t')
                    if len(fields) == 7:
                        text = fields[1]
                        language_id = fields[4]
                        if language_id == 'en':
                            text = text.strip()
                            if text:
                                out_f.write(text + '\n' + text + '\n\n')
                                valid += 1
                            else:
                                blank += 1
                    elif len(fields) == 2:
                        text = fields[1]
                        text = text.strip()
                        if text:
                            out_f.write(text + '\n' + text + '\n\n')
                            valid += 1
                        else:
                            blank += 1
                    elif len(fields) == 5:
                        continue
                    elif len(fields) == 6:
                        text = fields[0]
                        language_id = fields[3]
                        if language_id == 'en':
                            text = text.strip()
                            if text:
                                out_f.write(text + '\n' + text + '\n\n')
                                valid += 1
                            else:
                                blank += 1
                    else:
                        print(f'{len(fields)}: {line}')
                        err += 1
                    total += 1
    print(f'Write {valid} sentences into {args.output} from {total} sentences.')
    print(f'{err} errors. {blank} blank lines.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clean conceptnet raw sentences')
    parser.add_argument('--input', '-i', nargs='+', help='input files')
    parser.add_argument('--output', '-o', help='output file')
    args = parser.parse_args()
    main()
