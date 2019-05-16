# pylint: disable=invalid-name,no-self-use,missing-docstring
from allennlp.common.testing import AllenNlpTestCase
from csqa.modules.retriever import LuceneSearch


class TestLuceneSearch(AllenNlpTestCase):
    INDEX_ROOT = "/Users/handsome/Workspace/data/wikipedia/"

    def setUp(self):
        super(TestLuceneSearch, self).setUp()

        self.search_engine = LuceneSearch(
            index_dir=self.INDEX_ROOT + "index",
            db_path=self.INDEX_ROOT + "docs.db",
            num_search_workers=20)

    def test_search_single(self):
        result = self.search_engine.search(query="unicorn")
        assert len(result) == 20
        assert 'title' in result[0]
        assert 'text' in result[0]
        assert 'score' in result[0]
        LuceneSearch.pprint(result)

    def test_search_long(self):
        result = self.search_engine.search(
            "Who is the President of the United States?")
        assert result

    def test_search_batch(self):
        queries = [
            "Faryl Smith",
            "Donald Trump",
            "Tesla"
        ]
        results = self.search_engine.batch_search(
            queries, doc_max=5)
        assert len(results) == 3
        assert len(results[0]) == 5
        assert 'title' in results[0][0]
        print(results[0][0]['title'])
        print(results[1][0]['title'])
        print(results[1][1]['title'])
        print(results[1][2]['title'])
        print(results[2][0]['title'])
