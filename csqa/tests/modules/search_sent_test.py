# pylint: disable=invalid-name,no-self-use,missing-docstring


from termcolor import colored
from allennlp.common.testing import AllenNlpTestCase

from csqa.modules.retriever import TfidfSentenceRanker


class TestSentenceSearch(AllenNlpTestCase):

    def setUp(self):
        super(TestSentenceSearch, self).setUp()

    def test_rank(self):
        sentences = [
            "I am a doctor.",
            "cat a mouse",
            "There is a dog in the yard.",
            "There is a cat in the yard yard.",
            "in a dark evening",
            "one evening, a dark figure comes out.",
            "a room in house",
            "in a meeting"
        ]
        sent_ranker = TfidfSentenceRanker(sentences)
        queries = ["yard",
                   "dog",
                   "cat",
                   "dark evening",
                   "in the yard",
                   "doctor",
                   "There is a dog in the yard."]
        for query in queries:
            res = sent_ranker.closest_sents(query, 2)
            print(f'\nQuery: {colored(query, "yellow")}')
            TfidfSentenceRanker.pprint(res)
