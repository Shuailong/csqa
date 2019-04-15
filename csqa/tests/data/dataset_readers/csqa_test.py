# pylint: disable=no-self-use,invalid-name
import pytest
import pathlib
from allennlp.common import Params
from allennlp.common.util import ensure_list

from csqa.data.dataset_readers import CSQAReader


class TestCSQAReader:
    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / ".." / "tests" / "fixtures").resolve()

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read(self, lazy):
        params = Params({'lazy': lazy})
        reader = CSQAReader.from_params(params)
        instances = reader.read(
            str(self.FIXTURES_ROOT / 'csqa_sample.jsonl'))
        instances = ensure_list(instances)

        assert len(instances) == 10
        tokens = [t.text for t in instances[0].fields['qa_pairs'][3]]
        assert tokens == ['[CLS]', 'What', 'is', 'someone', 'doing', 'if', 'he', 'or', 'she', 'is', 'sitting',
                          'quietly', 'and', 'his', 'or', 'her', 'eyes', 'are', 'moving', '?', '[SEP]', 'fall', 'asleep', '[SEP]']
        assert instances[0].fields['answer_index'].sequence_index == 1

    def test_can_build_from_params(self):
        reader = CSQAReader.from_params(Params({}))
        # pylint: disable=protected-access
        assert reader._token_indexers['tokens'].__class__.__name__ == 'SingleIdTokenIndexer'
