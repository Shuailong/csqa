# pylint: disable=no-self-use,invalid-name
from flaky import flaky
import pathlib

from allennlp.common.testing import ModelTestCase
from csqa.data.dataset_readers import CSQAReaderAttn
from csqa.models import CSQABertAttn


class CSQABertAttnTest(ModelTestCase):
    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / "tests" / "fixtures").resolve()

    def setUp(self):
        super(CSQABertAttnTest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'experiment_evidence_attn.jsonnet',
                          self.FIXTURES_ROOT / 'csqa_evidence_sample.jsonl')

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        question = training_tensors['question']
        answer_index = training_tensors['answer_index']
        choices = training_tensors['choices']
        evidence = training_tensors['evidence']

        print('question:', question)
        print('answer index:', answer_index)
        print('choice:', choices)
        print('evidence:', evidence)
        output_dict = self.model(**training_tensors)
        print('answer probs:', output_dict['answer_probs'])
        print('loss:', output_dict["loss"])
        assert "qid" in output_dict and "loss" in output_dict
        assert "answer_logits" in output_dict and "answer_probs" in output_dict
