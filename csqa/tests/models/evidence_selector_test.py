# pylint: disable=no-self-use,invalid-name
import pathlib
from allennlp.common.testing import ModelTestCase
from csqa.data.dataset_readers import QEReader
from csqa.models import EvidenceSelector


class EvidenceSelectorTest(ModelTestCase):
    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / "tests" / "fixtures").resolve()

    def setUp(self):
        super(EvidenceSelectorTest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'experiment_qe.jsonnet',
                          self.FIXTURES_ROOT / 'qe_sample.txt')

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert "loss" in output_dict
