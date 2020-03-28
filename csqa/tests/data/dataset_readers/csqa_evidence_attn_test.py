# pylint: disable=no-self-use,invalid-name
import pathlib

from allennlp.common import Params
from allennlp.common.util import ensure_list

from csqa.data.dataset_readers import CSQAReaderAttn


class TestCSQAReader:
    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / ".." / "tests" / "fixtures").resolve()

    def test_read(self):
        params = Params({'num_evidences': 3})
        reader = CSQAReaderAttn.from_params(params)
        instances = reader.read(
            str(self.FIXTURES_ROOT / 'csqa_evidence_sample.jsonl'))
        instances = ensure_list(instances)
        sample = instances[0]
        print(sample.fields['question'])
        answer_tokens = sample.fields['choices']
        print(answer_tokens)
        for choice_evidence in sample.fields['evidence']:
            # for each choice
            for evi in choice_evidence:
                print(evi)
