# CommonsenseQA

The codebase for project [commonsenseqa][commonsense-qa-website].

[[leaderboard]][commonsense-qa-leaderboard] [[github]][commonsense-qa-github]

[commonsense-qa-leaderboard]: https://www.tau-nlp.org/csqa-leaderboard
[commonsense-qa-website]: https://www.tau-nlp.org/commonsenseqa
[commonsense-qa-github]: https://github.com/jonathanherzig/commonsenseqa

### Data Sample
```json
{
    "answerKey": "B",
    "id": "70701f5d1d62e58d5c74e2e303bb4065",
    "question": {
        "choices": [
            {
                "label": "A",
                "text": "bunk"
            },
            {
                "label": "B",
                "text": "reading"
            },
            {
                "label": "C",
                "text": "think"
            },
            {
                "label": "D",
                "text": "fall asleep"
            },
            {
                "label": "E",
                "text": "meditate"
            }
        ],
        "stem": "What is someone doing if he or she is sitting quietly and his or her eyes are moving?"
    }
}
```

### Data Prepare
<pre>

|--data
    |
    |--csqa
    |    |
    |    |---QTokenSplit
    |    |        |
    |    |        |--train_qtoken_split.jsonl
    |    |        |--train_qtoken_split_EASY.jsonl
    |    |        |--dev_qtoken_split.jsonl
    |    |        |--dev_qtoken_split_EASY.jsonl
    |    |        |--test_qtoken_split_no_answers.jsonl
    |    |
    |    |---RandSplit
    |             |
    |             |--train_rand_split.jsonl
    |             |--dev_rand_split.jsonl
    |             |--dev_rand_split_EASY.jsonl
    |             |--test_rand_split_no_answers.jsonl
    |--bert
         |--bert-base-uncased-vocab.txt
         |--bert-base-uncased.tar.gz
         |--bert-large-uncased-vocab.txt
         |--bert-large-uncased.tar.gz

</pre>

### Train

#### on docker
```bash
# pull docker
docker pull shuailongliang/csqa:latest
# start training 
nvidia-docker run --rm -v "/home/long/csqa:/mnt/csqa"  shuailongliang/csqa:latest train -s /mnt/csqa/models/20190415-base /mnt/csqa/experiments/bert.jsonnet --include-package csqa
```
#### on local GPU machine
```bash
allennlp train experiments/bert.jsonnet -s models/20190415-base  --include-package csqa 
```

### Predict
#### on docker
```bash
nvidia-docker run --rm -v "/home/long/csqa:/mnt/csqa"  shuailongliang/csqa:latest predict /mnt/csqa/models/20190415-base/model.tar.gz  /mnt/csqa/data/csqa/RandSplit/test_rand_split_no_answers.jsonl --output-file /mnt/csqa/models/20190415-base/test_rand_split_answer.txt --cuda-device 0 --predictor csqa --include-package csqa
```
#### on local GPU machine
```bash
allennlp predict models/20190415-base/model.tar.gz data/csqa/RandSplit/test_rand_split_no_answers.jsonl --output-file models/20190415-base/test_rand_split_answer.txt --cuda-device 0 --predictor csqa --include-package csqa 
```