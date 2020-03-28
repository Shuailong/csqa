import json
import argparse


def main(args):
    with open(args.input) as ifile, open(args.output, 'w') as ofile:
        for q_no, line in enumerate(ifile):
            jsondict = json.loads(line)
            question = jsondict['question']['stem']

            ofile.write(f'{q_no}. {question}\n')
            for choice in jsondict['question']['choices']:
                ofile.write(f'{choice["label"]}: {choice["text"]}\n')
                evidence = choice['evidence'] if 'evidence' in choice else None
                evidence_ranked = choice['evidence_ranked'] if 'evidence_ranked' in choice else None
                if evidence_ranked:
                    for evi in evidence_ranked[:args.max_evidence]:
                        if args.with_score:
                            ofile.write(f'\t{evi[0]} ({evi[1]})\n')
                        else:
                            ofile.write(f'\t{evi[0]}\n')
                elif evidence:
                    for evi in evidence[:args.max_evidence]:
                        ofile.write(f'\t{evi}\n')

            ofile.write('\n')

            if 'answerKey' in jsondict:
                answer = jsondict['answerKey']
                ofile.write(f'True answer: {answer}\n')
            if 'prediction' in jsondict:
                prediction = jsondict['prediction']
                ofile.write(f'Predict answer: {prediction}\n')
            ofile.write('-' * 100 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Format csqa sample into human readable form')
    parser.add_argument('input', help='input file name')
    parser.add_argument('output', help='input file name')
    parser.add_argument('--max-evidence', default=3,
                        help='maximum number of evidences to show')
    parser.add_argument('--with-score', action='store_true')
    args = parser.parse_args()
    main(args)
