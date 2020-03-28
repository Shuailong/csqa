# BECON: BERT with Evidence from CONceptNet for Commonsense Question Answering

## Motivation

[CommonsenseQA](https://arxiv.org/abs/1811.00937) dataset is created by crowdsourcing workers based on knowledge graphs on [ConceptNet](http://conceptnet.io). Solving the tasks requries the model to have commonsense knowledge. Can we utilize the commonsense knowkedge in ConceptNet to solve the commonsense QA task? How can we take advantage of BERT in our model?

## Evidence Finder

From [ConceptNet](http://conceptnet.io) English API `http://api.conceptnet.io/c/en/`, given a word ``w``, output all the possible information related with ``w``.
Example:

```python
>>> import requests
>>> obj = requests.get('http://api.conceptnet.io/c/en/example').json()
>>> obj.keys()
dict_keys(['view', '@context', '@id', 'edges'])

>>> len(obj['edges'])
20

>>> obj['edges'][2]
{'@id': '/a/[/r/IsA/,/c/en/example/n/,/c/en/information/n/]',
 'dataset': '/d/wordnet/3.1',
 'end': {'@id': '/c/en/information/n',
  'label': 'information',
  'language': 'en',
  'sense_label': 'n',
  'term': '/c/en/information'},
 'license': 'cc:by/4.0',
 'rel': {'@id': '/r/IsA', 'label': 'IsA'},
 'sources': [{'@id': '/s/resource/wordnet/rdf/3.1',
   'contributor': '/s/resource/wordnet/rdf/3.1'}],
 'start': {'@id': '/c/en/example/n',
  'label': 'example',
  'language': 'en',
  'sense_label': 'n',
  'term': '/c/en/example'},
 'surfaceText': [[example]] is a type of [[information]]',
 'weight': 2.0}
```

See [concept_example.json](concept_example.json) for a full json tree example.

Use ``surfaceText`` of all edges as evidence sentences and remove translation sentences.

```python
query = 'example'
obj = requests.get('http://api.conceptnet.io/c/en/' + query).json()
for edge in obj['edges']:
    if edge['surfaceText']:
        if 'is a translation of' not in edge['surfaceText']:
            print(edge['surfaceText'])
```

## Evidence Ranker

Ranking the evidences according to the relevant scores with the question.
The rankers are evaluated according to performance on CSQA dataset.
A simple model without training: select the choice with the highest evidence score.

| Ranker     | train | dev   | train_EASY | dev_EASY |
| ---------- | ----- | ----- | ---------- | -------- |
| random     | 21.14 | 19.82 | 21.07      | 19.57    |
| jaccard    | 23.12 | 22.44 | 44.43      | 41.28    |
| w2v        | 26.05 | 23.91 | 48.73      | 47.01    |
| bert-base  | 34.95 | 34.73 | 82.89      | 81.90    |
| bert-large | 36.50 | 36.86 | 84.41      | 82.88    |

Another simple model without training: select the choice with the highest score.

### NextSentencePrediction Pretrained BERT Model

| Model          | train | dev   | train_EASY | dev_EASY |
| -------------- | ----- | ----- | ---------- | -------- |
| BERT-base NSP  | 35.36 | 39.39 | 71.28      | 71.99    |
| BERT-large NSP | 38.41 | 40.38 | 73.54      | 73.14    |

## QA Models

### Baselines

Baselines

| Models                 | train | dev    | test | test-easy |
| ---------------------- | ----- | ------ | ---- | --------- |
| KagNet                 |       |        | 58.9 |           |
| CoS-E                  |       |        | 58.2 |           |
| SGN-lite               |       |        | 57.1 |           |
| BERT-large(Tel-Aviv U) |       | (63.4) | 56.7 |           |
| BERT-large             |       |        | 55.9 | 92.3      |
| BERT-base(UCL)         |       | (57.6) | 53.0 |           |
| GPT                    |       |        | 45.5 | 87.2      |
| ESIM+ELMo              |       |        | 34.1 | 76.9      |
| ESIM+glove             |       |        | 32.8 | 79.1      |

Reproduce

| Models     | train | dev  |
| ---------- | ----- | ---- |
| BERT-base  |       | 57.6 |
| BERT-large |       | 63.4 |

### Evidence

| Choice + Evidence (hard) Models | train | dev  | rank |
| ------------------------------- | ----- | ---- | ---- |
| BERT-base \| bert-base-ranker   |       | 56.2 | -    |
| BERT-base \| bert-large-ranker  |       | 57.6 | -    |
| BERT-large \| bert-base-ranker  |       | 61.9 | -    |
| BERT-large \| bert-large-ranker |       | 62.2 | -    |

| Choice + Evidence (both) Models | train | dev  | rank |
| ------------------------------- | ----- | ---- | ---- |
| BERT-base-mean                  |       | 58.7 | -    |
| BERT-large-mean                 |       | 64.0 | -    |
| BERT-large-max                  |       | 63.6 | -    |
| BERT-large-cat                  |       | 63.8 | -    |

| Question + Evidence Models      | train | dev  | rank |
| ------------------------------- | ----- | ---- | ---- |
| BERT-base \| bert-base-ranker   |       | 57.7 | -    |
| BERT-base \| bert-large-ranker  |       | 58.3 | -    |
| BERT-large \| bert-base-ranker  |       | 62.1 | -    |
| BERT-large \| bert-large-ranker |       | 62.8 | -    |

| Question + Evidence Models (Mean) | train | dev  | rank |
| --------------------------------- | ----- | ---- | ---- |
| BERT-base \| bert-large-ranker    |       | 58.1 | -    |
| BERT-large \| bert-large-ranker   |       | 61.3 | -    |

### BERT-Sep

| Model                            | dev  |
| -------------------------------- | ---- |
| question + choice                | 22.4 |
| question + choice + 1st-evidence | 22.7 |

Q+E:    question + evidence + choice
C+E:    question + choice + evidence
Base:   question + choice
