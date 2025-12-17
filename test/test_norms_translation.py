import unittest
import pandas as pd
from bluefairy.norms import textual_norm_to_logic_norm
from ollamaUtils import OllamaService, OLLAMA_URL, OLLAMA_PORT
from test.data import load_examples


class TestNormsTranslation(unittest.TestCase):
    def setUp(self):
        self.healthy_norm = "A healthy lifestyle includes regular exercise, a balanced diet, and adequate sleep."
        self.unhealthy_norm = "A food item is unhealthy if it is high in sugar or high in fat."
        examples = load_examples()
        self.examples: str = examples.apply(
            lambda row: f"Textual Norm: {row['NL']}\nLogical Norm: {row['FOL']}\n", axis=1
        ).str.cat(sep="\n")
        self.provider = OllamaService(OLLAMA_URL, OLLAMA_PORT)

    def test_textual_norm_to_logic_norm_with_no_examples(self):
        logic_textual_norm = textual_norm_to_logic_norm(self.provider, self.unhealthy_norm)
        self.assertIsInstance(logic_textual_norm, str)
        self.assertGreater(len(logic_textual_norm), 0)

    def test_textual_norm_to_logic_norm_with_examples(self):
        logic_textual_norm = textual_norm_to_logic_norm(self.provider, self.unhealthy_norm, self.examples)
        self.assertIsInstance(logic_textual_norm, str)
        self.assertGreater(len(logic_textual_norm), 0)