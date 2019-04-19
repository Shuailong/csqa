import json
import argparse


def main(args):
    with open(args.input) as ifile, open(args.output, 'w') as ofile:
        for line in ifile:
            jsondict = json.loads(line)
            question = jsondict['question']['stem']
            choices = [(d['label'], d['text'])
                       for d in jsondict['question']['choices']]
            answer = jsondict['answerKey']
            prediction = jsondict['prediction']
            ofile.write(f'{question}\n')
            for label, text in choices:
                ofile.write(f'{label}: {text}  ')
            ofile.write('\n')
            ofile.write(f'True answer: {answer}\n')
            ofile.write(f'Predict answer: {prediction}\n')
            ofile.write('-' * 100 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Format Error Instances into human readable form')
    parser.add_argument('input', help='input file name')
    parser.add_argument('output', help='input file name')
    args = parser.parse_args()
    main(args)
