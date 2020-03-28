# pylint: disable=no-self-use,invalid-name
import pytest
import pathlib
from allennlp.common import Params
from allennlp.common.util import ensure_list

from csqa.data.dataset_readers import QEReader


class TestQEReader:
    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / ".." / "tests" / "fixtures").resolve()

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read(self, lazy):
        params = Params({'lazy': lazy})
        reader = QEReader.from_params(params)
        instances = reader.read(
            str(self.FIXTURES_ROOT / 'qe_sample.txt'))
        instances = ensure_list(instances)

        assert len(instances) == 10
        sample = instances[0]
        tokens = [t.text for t in sample.fields['tokens']]
        label = sample.fields['label']
        print(tokens)
        print(label)

    def test_can_build_from_params(self):
        reader = QEReader.from_params(Params({}))
        # pylint: disable=protected-access
        assert reader._token_indexers['tokens'].__class__.__name__ == 'SingleIdTokenIndexer'
