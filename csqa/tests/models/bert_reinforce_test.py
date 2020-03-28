# pylint: disable=no-self-use,invalid-name
import pathlib
from allennlp.common.testing import ModelTestCase
from csqa.data.dataset_readers import CSQAReaderReinforce
from csqa.models import CSQABertReinforce


class CSQABertReinforceTest(ModelTestCase):
    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / "tests" / "fixtures").resolve()

    def setUp(self):
        super(CSQABertReinforceTest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'experiment_reinforce.jsonnet',
                          self.FIXTURES_ROOT / 'csqa_evidence_sample.jsonl')

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert "loss" in output_dict
