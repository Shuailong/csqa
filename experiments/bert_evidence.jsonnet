# env
local run_env = 'local'; // local | docker
local device = 0;

# model
local bert_type = "base"; // base | large

# data
local split = "RandSplit"; // QTokenSplit | RandSplit

# training
local epochs = 3;
local learning_rate = 2e-5;
local train_samples = 9741;
local batch_size_base = 12; // 8g GPU mem required
local batch_size_large = 5; // 16g GPU mem required

# dependent
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
            }
        }
    },
    "train_data_path": data_root + "/csqa/" + split + "/train_rand_split_evidences_bertlarge_ranked.jsonl",
    "validation_data_path": data_root + "/csqa/" + split + "/dev_rand_split_evidences_bertlarge_ranked.jsonl",
    "test_data_path": data_root + "/csqa/" + split + "/test_rand_split_no_answers.jsonl",
    "evaluate_on_test": false,
    "model": {
        "type": "csqa-bert",
        "dropout": 0.1,
        "bert": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": [
                    "bert",
                    "bert-offsets",
                    "bert-type-ids"
                ]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained-sl",
                    "pretrained_model": data_root + "/bert/bert-"+bert_type+"-uncased.tar.gz",
                    "requires_grad": true,
                    "top_layer_only": true,
                    "type_vocab_size": 3
                }
            }
        },
        "classifier": {
            "input_dim": feature_size,
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
            "type": "bert_adam",
            "lr": learning_rate,
            "t_total": train_samples/batch_size*epochs,
            "warmup": 0.1,
            "weight_decay": 1e-2,
            "parameter_groups": [
                [["bias", "LayerNorm.bias", "LayerNorm.weight"], {"weight_decay": 0.0}],
            ]
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": epochs,
        "cuda_device": device
    }
}