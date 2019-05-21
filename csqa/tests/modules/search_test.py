# pylint: disable=invalid-name,no-self-use,missing-docstring
import os

from termcolor import colored

from allennlp.common.testing import AllenNlpTestCase

from csqa.modules.retriever import SearchEngine
from csqa import DATA_DIR


class TestSearch(AllenNlpTestCase):
    INDEX_ROOT = os.path.join(DATA_DIR, "wikipedia/index")

    def setUp(self):
        super(TestSearch, self).setUp()
        self.search_engine = SearchEngine(index_dir=self.INDEX_ROOT,
                                          num_search_workers=20)

    def test_search(self):
        queries = [
            "Ultimate question of life, the universe, and everything",
            "People's Republic of China (PRC)",
            "Disaster risk reduction (DRR) is a systematic approach to identifying, assessing and reducing the risks of disaster."
        ]
        for query in queries:
            print(f'\nQuery: {colored(query, "yellow")}')
            result = self.search_engine.search(query)
            SearchEngine.pprint(result)
