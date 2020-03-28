# pylint: disable=no-self-use,invalid-name
import pathlib

from allennlp.common import Params
from allennlp.common.util import ensure_list

from csqa.data.dataset_readers import CSQAReaderQuestionEvidence


class TestCSQAReaderQuestionEvidence:
    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / ".." / "tests" / "fixtures").resolve()

    def test_read(self):
        params = Params({"num_evidences": 3})
        reader = CSQAReaderQuestionEvidence.from_params(params)
        instances = reader.read(
            str(self.FIXTURES_ROOT / 'csqa_evidence_sample.jsonl'))
        instances = ensure_list(instances)
        sample = instances[0]
        print(sample.fields['qa_pairs'])
