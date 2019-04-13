# pylint: disable=no-self-use,invalid-name

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch


class CSQABertTest(ModelTestCase):
    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / "tests" / "fixtures").resolve()

    def setUp(self):
        super(CSQABertTest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'experiment.json',
                          self.FIXTURES_ROOT / 'csqa_sample.jsonl')
        self.batch = Batch(self.instances)
        self.batch.index_instances(self.vocab)

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.batch.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert "qid" in output_dict and "loss" in output_dict
        assert "answer_logits" in output_dict and "answer_probs" in output_dict

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.param_file, tolerance=1e-4)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()
