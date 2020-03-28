# pylint: disable=no-self-use,invalid-name
import pathlib
import pytest
from allennlp.common import Params
from allennlp.common.util import ensure_list

from csqa.data.dataset_readers import CSQAReaderReinforce


class TestCSQAReinforceReader:
    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / ".." / "tests" / "fixtures").resolve()

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read(self, lazy):
        params = Params({'lazy': lazy})
        reader = CSQAReaderReinforce.from_params(params)
        instances = reader.read(
            str(self.FIXTURES_ROOT / 'csqa_evidence_sample.jsonl'))
        instances = ensure_list(instances)

        assert len(instances) == 5
        fields = instances[0].fields
        question = fields['question']
        answer = fields['choices']
        answer_evidences = fields['evidence']
        answer_index = fields['answer_index']
        print(question)
        print(answer)
        print(answer_evidences)
        print(answer_index)
