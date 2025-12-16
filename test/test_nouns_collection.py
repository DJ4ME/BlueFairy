import unittest
from test.cache import PATH as CACHE_PATH
from bluefairy.nouns.utils import NounsCollection, Stakeholder, Role
from test.data import StakeholderName, load_textual_norms
from bluefairy.nouns import (
    run_context_identification,
    run_nouns_generation,
    run_nouns_cleaning,
    run_nouns_classification
)


CONTEXT_FILE = CACHE_PATH / "test_context.txt"


class TestNounsCollection(unittest.TestCase):
    def setUp(self):
        self.nouns = NounsCollection()
        self.stakeholders = [Stakeholder(enum.name) for enum in StakeholderName]
        for stakeholder in self.stakeholders:
            stakeholder.norms = load_textual_norms(StakeholderName[stakeholder.noun])

    def test_run_context_identification(self):
        all_norms = []
        for stakeholder in self.stakeholders:
            all_norms.extend(stakeholder.norms)
        context = run_context_identification(all_norms)

        # Verify that context is a non-empty string
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)

        # Save context to a file for the next tests

        with open(CONTEXT_FILE, "w", encoding="utf-8") as file:
            file.write(context)


    def test_run_nouns_generation(self):
        all_norms = []
        for stakeholder in self.stakeholders:
            all_norms.extend(stakeholder.norms)
        # Load context from the file created in the previous test
        with open(CONTEXT_FILE, "r", encoding="utf-8") as file:
            context = file.read()

        test_file = CACHE_PATH / "test_nouns_generation.csv"
        run_nouns_generation(all_norms, context, test_file)

        # Verify that the nouns file is created and contains nouns
        with open(test_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            self.assertGreater(len(lines), 1)

        # Copy the file in a new file "test_nouns_cleaning.csv" for the next test
        cleaned_file = CACHE_PATH / "test_nouns_cleaning.csv"
        with open(cleaned_file, "w", encoding="utf-8") as file:
            file.writelines(lines)


    def test_run_nouns_cleaning(self):
        cleaned_file = CACHE_PATH / "test_nouns_cleaning.csv"
        run_nouns_cleaning(cleaned_file)

        # Verify that the cleaned nouns file is created and contains nouns
        with open(cleaned_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            self.assertGreater(len(lines), 1)

        # Copy the file in a new file "test_nouns_classification.csv" for the next test
        classified_file = CACHE_PATH / "test_nouns_classification.csv"
        with open(classified_file, "w", encoding="utf-8") as file:
            file.writelines(lines)


    def test_run_nouns_classification(self):
        classified_file = CACHE_PATH / "test_nouns_classification.csv"
        # Load context from the file created in the first test
        with open(CONTEXT_FILE, "r", encoding="utf-8") as file:
            context = file.read()

        run_nouns_classification(context, classified_file)

        # Verify that the classified nouns file is created and contains nouns with roles
        with open(classified_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            self.assertGreater(len(lines), 1)
            for line in lines[1:]:
                noun, role = line.strip().split(",")
                self.assertIn(role, Role._member_names_)
