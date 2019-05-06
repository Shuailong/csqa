local split = "RandSplit"; // QTokenSplit | RandSplit
local bert_type = "base"; // base | large
local run_env = 'docker'; // local | docker

local batch_size_base = 16; // 8g GPU mem required
local batch_size_large = 5; // 16g GPU mem required
local feature_size_base = 768;
local feature_size_large = 1024;
local data_root = if run_env == 'local' then 'data' else '/mnt/csqa/data';
local batch_size = if bert_type == 'base' then batch_size_base else batch_size_large;
local feature_size = if bert_type == 'base' then feature_size_base else feature_size_large;

{
    "dataset_reader": {
        "type": "csqa",
        "tokenizer": {
            "type": "word",
            "word_splitter": {
                "type": "bert-basic"
            }
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": data_root + "/bert/bert-"+bert_type+"-uncased-vocab.txt"
            },
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
            },
        }
    },
    "train_data_path": data_root + "/csqa/" + split + "/train_rand_split.jsonl",
    "validation_data_path": data_root + "/csqa/" + split + "/dev_rand_split.jsonl",
    "test_data_path": data_root + "/csqa/" + split + "/test_rand_split_no_answers.jsonl",
    "evaluate_on_test": false,
    "model": {
        "type": "csqa-bert-emb",
        "dropout": 0.1,
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": [
                    "bert",
                    "bert-offsets",
                    "bert-type-ids"
                ],
                "tokens":[
                    "tokens"
                ]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": data_root + "/bert/bert-"+bert_type+"-uncased.tar.gz",
                    "requires_grad": true,
                    "top_layer_only": true
                },
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": data_root + "/numberbatch/numberbatch-en-17.06.txt.gz",
                    "embedding_dim": 300,
                    "trainable": false
                },
            }
        },
        "output_logit": {
            "input_dim": feature_size + 300,
            "num_layers": 1,
            "hidden_dims": 1,
            "activations": "linear"
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": batch_size
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 2e-5
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 3,
        "grad_norm": 10.0,
        "cuda_device": [4,5],
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        }
    }
}