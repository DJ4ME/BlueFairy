import unittest
from test.cache import PATH as CACHE_PATH
from bluefairy.nouns.utils import NounsCollection, Stakeholder
from test.data import StakeholderName, load_textual_norms
from bluefairy.nouns.collection import collect_nouns_from_stakeholders


class TestNounsCollection(unittest.TestCase):
    def setUp(self):
        self.nouns = NounsCollection()
        self.stakeholders = [Stakeholder(enum.name) for enum in StakeholderName]
        for stakeholder in self.stakeholders:
            stakeholder.norms = load_textual_norms(StakeholderName[stakeholder.noun])

    def test_nouns_collection(self):
        collected_nouns = collect_nouns_from_stakeholders(self.stakeholders)
        # Save to cache later analysis if needed
        cache_file = CACHE_PATH / "collected_nouns.csv"
        collected_nouns.save_nouns_to_csv(str(cache_file))

        self.assertIsInstance(collected_nouns, NounsCollection)
        self.assertGreater(len(collected_nouns._nouns), 0)
