# pylint: disable=no-self-use,invalid-name
from flaky import flaky
import pathlib

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from csqa.data.dataset_readers import CSQAReader
from csqa.models import CSQABert


class CSQABertTest(ModelTestCase):
    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / "tests" / "fixtures").resolve()

    def setUp(self):
        super(CSQABertTest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'experiment.jsonnet',
                          self.FIXTURES_ROOT / 'csqa_sample.jsonl')

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        qa_pair = training_tensors['qa_pairs']
        answer_index = training_tensors['answer_index']
        qa_pair_bert = qa_pair['bert']
        A_tokens = [t.text for t in self.instances[0].fields['qa_pairs'][0]]
        A_indexs = qa_pair_bert[0][0]
        A_tokens_recovered = [self.vocab.get_token_from_index(i, "bert")
                              for i in qa_pair_bert.tolist()[0][0]]

        assert A_tokens == ['what', 'is', 'someone', 'doing', 'if', 'he', 'or', 'she', 'is', 'sitting',
                            'quietly', 'and', 'his', 'or', 'her', 'eyes', 'are', 'moving', '?', '[SEP]', 'bunk']
        assert A_indexs.tolist() == [101, 2054, 2003, 2619, 2725, 2065, 2002, 2030, 2016, 2003, 3564,
                                     5168, 1998, 2010, 2030, 2014, 2159, 2024, 3048, 1029, 102, 25277, 102, 0, 0]
        assert A_tokens_recovered == ['[CLS]', 'what', 'is', 'someone', 'doing', 'if', 'he', 'or', 'she', 'is', 'sitting',
                                      'quietly', 'and', 'his', 'or', 'her', 'eyes', 'are', 'moving', '?', '[SEP]', 'bunk', '[SEP]', '[PAD]', '[PAD]']
        assert qa_pair_bert.dim() == 3
        assert qa_pair_bert.size(0) == 10
        assert qa_pair_bert.size(1) == 5
        assert answer_index.dim() == 2
        assert answer_index.size(0) == 10
        assert qa_pair_bert.tolist()[0] == [[101,  2054,  2003,  2619,  2725,  2065,  2002,  2030,  2016,  2003,
                                             3564,  5168,  1998,  2010,  2030,  2014,  2159,  2024,  3048,  1029,
                                             102, 25277,   102,     0,     0],
                                            [101,  2054,  2003,  2619,  2725,  2065,  2002,  2030,  2016,  2003,
                                             3564,  5168,  1998,  2010,  2030,  2014,  2159,  2024,  3048,  1029,
                                             102,  3752,   102,     0,     0],
                                            [101,  2054,  2003,  2619,  2725,  2065,  2002,  2030,  2016,  2003,
                                             3564,  5168,  1998,  2010,  2030,  2014,  2159,  2024,  3048,  1029,
                                             102,  2228,   102,     0,     0],
                                            [101,  2054,  2003,  2619,  2725,  2065,  2002,  2030,  2016,  2003,
                                             3564,  5168,  1998,  2010,  2030,  2014,  2159,  2024,  3048,  1029,
                                             102,  2991,  6680,   102,     0],
                                            [101,  2054,  2003,  2619,  2725,  2065,  2002,  2030,  2016,  2003,
                                             3564,  5168,  1998,  2010,  2030,  2014,  2159,  2024,  3048,  1029,
                                             102, 19960, 17570,   102,     0]]
        assert qa_pair['bert-type-ids'].tolist()[0] == [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]]

        assert qa_pair['mask'].tolist()[0] == [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
        assert qa_pair['bert-offsets'].tolist()[0] == [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 0],
                                                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 0],
                                                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 0],
                                                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0],
                                                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                        13, 14, 15, 16, 17, 18, 19, 20, 22, 0, 0]]

        output_dict = self.model(**training_tensors)
        assert "qid" in output_dict and "loss" in output_dict
        assert "answer_logits" in output_dict and "answer_probs" in output_dict

    # def test_model_can_train_save_and_load(self):
    #     self.ensure_model_can_train_save_and_load(
    #         self.param_file)

    # @flaky(max_runs=3)
    # def test_batch_predictions_are_consistent(self):
    #     self.ensure_batch_predictions_are_consistent()
    # Removed test since too flaky
