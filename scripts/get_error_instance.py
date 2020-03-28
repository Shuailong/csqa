#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-04-18 17:27:21
# @Last Modified by: Shuailong
# @Last Modified time: 2019-04-18 17:28:33

import csv
import sys
import json
import logging
import argparse
from typing import List, Dict, Any

EXIT_STATUS_ANSWERS_MALFORMED = 1
EXIT_STATUS_PREDICTIONS_MALFORMED = 2
EXIT_STATUS_PREDICTIONS_EXTRA = 3
EXIT_STATUS_PREDICTION_MISSING = 4


def read_records(filename: str) -> Dict[str, Dict[str, Any]]:
    records = {}

    with open(filename, "rt", encoding="UTF-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            try:
                record = json.loads(line)
            except ValueError as e:
                logging.error("Error while reading file %s: %s", filename, e)
                sys.exit(EXIT_STATUS_ANSWERS_MALFORMED)

            question_id = record["id"]

            if question_id in records:
                logging.error("Key %s repeated in %s", question_id, filename)
                sys.exit(EXIT_STATUS_ANSWERS_MALFORMED)
            records[question_id] = record

    if not records:
        logging.error("No records found in file %s", filename)
        sys.exit(EXIT_STATUS_ANSWERS_MALFORMED)

    return records


def read_predictions(filename: str) -> Dict[str, List[str]]:
    predictions = {}

    with open(filename, "rt", encoding="UTF-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            for row in reader:
                try:
                    question_id = row[0]
                    prediction_raw = row[1]
                except IndexError as e:
                    logging.error(
                        "Error reading value from CSV file %s on line %d: %s", filename, reader.line_num, e)
                    sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

                if question_id in predictions:
                    logging.error("Key %s repeated in file %s on line %d",
                                  question_id, filename, reader.line_num)
                    sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

                if question_id == "":
                    logging.error(
                        "Key is empty in file %s on line %d", filename, reader.line_num)
                    sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

                prediction = prediction_raw.split(";")
                # prediction labels cannot be empty strings
                for p in prediction:
                    if p == "":
                        logging.error("Key %s has empty labels for prediction in file %s on line %d",
                                      question_id, filename, reader.line_num)
                        sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)
                predictions[question_id] = prediction

        except csv.Error as e:
            logging.error('file %s, line %d: %s', filename, reader.line_num, e)
            sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

    return predictions


def find_error(question_answers: Dict[str, Dict[str, Any]], predictions: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Given answer keys and predictions, return instances with wrong prediction.

    Parameters
    ----------
    question_answers : Dict[str, Dict[str, Any]]
        List of dict with key ``id`` and ``answerKey``
    predictions : Dict[str, List[str]]
        A dictionary with ``id`` as key and prediction label as value.

    Returns
    -------
    List[Dict[str, Any]]
        Error instances, format the same as question_answers, with additional key ``prediction``.
    """
    res = []
    for question_id, record in question_answers.items():
        try:
            predictions_for_q = predictions[question_id]
        except KeyError:
            logging.error("Missing prediction for question '%s'.", question_id)
            sys.exit(EXIT_STATUS_PREDICTION_MISSING)
        answer = record['answerKey']
        if answer not in predictions_for_q:
            record['prediction'] = ';'.join(predictions_for_q)
            res.append(record)

    return res


def main(args):
    question_answers = read_records(args.question_answers)
    predictions = read_predictions(args.predictions)

    error_instances = find_error(question_answers, predictions)

    with open(args.output, "wt", encoding="UTF-8") as output:
        for instance in error_instances:
            output.write(json.dumps(instance) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given answer key and predictions, output the wrong instance.')
    parser.add_argument(
        '--question-answers', '-qa',
        help='Filename of the question answers to read. Expects a JSONL file with documents that have field "id" and "answerKey".',
        required=True)
    parser.add_argument(
        '--predictions', '-p',
        help="Filename of the leaderboard predictions, in CSV format.",
        required=True)
    parser.add_argument(
        '--output', '-o',
        help='Output results to this file.',
        required=True)
    args = parser.parse_args()
    main(args)
