# pylint: disable=invalid-name,no-self-use,missing-docstring
from termcolor import colored

from allennlp.common.testing import AllenNlpTestCase

from csqa.modules.retriever import SearchEngine


class TestSearch(AllenNlpTestCase):
    INDEX_ROOT = "/Users/handsome/Workspace/data/wikipedia/"

    def setUp(self):
        super(TestSearch, self).setUp()
        self.search_engine = SearchEngine(index_dir=self.INDEX_ROOT + "index",
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
