import unittest

import pandas as pd
import spacy

from src.conll_ner_builder import ConllBuilder


class FrequencyMentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.blank("es")
        cls.nlp.add_pipe("sentencizer")

    def setUp(self):
        self.builder = ConllBuilder(txt_dir=".", entity_types=["FREQ"])
        self.builder._nlp = self.nlp

    def _labels(self, text: str, annotations: list[str]) -> list[str]:
        mentions = self.builder._collect_mentions(
            pd.DataFrame({"FRECUENCIA": annotations})
        )
        return [
            label
            for tokens, labels in self.builder._tag_text(text, mentions)
            for label in labels
        ]

    def test_singular_and_plural_annotations_match_both_forms(self):
        cases = (
            ("FRECUENTE", "FRECUENTES", "frecuente frecuentes", ["B-FREQ", "B-FREQ"]),
            (
                "MUY FRECUENTE",
                "MUY FRECUENTES",
                "muy frecuente muy frecuentes",
                ["B-FREQ", "I-FREQ", "B-FREQ", "I-FREQ"],
            ),
            (
                "POCO FRECUENTE",
                "POCO FRECUENTES",
                "poco frecuente poco frecuentes",
                ["B-FREQ", "I-FREQ", "B-FREQ", "I-FREQ"],
            ),
        )
        for singular, plural, text, expected in cases:
            for annotation in (singular, plural.lower()):
                with self.subTest(annotation=annotation):
                    self.assertEqual(self._labels(text, [annotation]), expected)

    def test_longer_frequency_phrases_keep_their_full_span(self):
        self.assertEqual(
            self._labels(
                "Frecuentes; MUY frecuentes; Poco frecuentes.",
                ["FRECUENTE", "MUY FRECUENTE", "POCO FRECUENTE"],
            ),
            ["B-FREQ", "O", "B-FREQ", "I-FREQ", "O", "B-FREQ", "I-FREQ", "O"],
        )

    def test_word_boundaries_are_preserved(self):
        self.assertEqual(
            self._labels("frecuentemente infrecuentes frecuentess frecuentes", ["FRECUENTE"]),
            ["O", "O", "O", "B-FREQ"],
        )

    def test_unconfirmed_categories_remain_literal(self):
        annotations = ["NO FRECUENTE", "SIN FRECUENCIA"]
        mentions = self.builder._collect_mentions(
            pd.DataFrame({"FRECUENCIA": annotations})
        )
        self.assertEqual(set(mentions["FREQ"]), set(annotations))
        self.assertTrue(
            all(label == "O" for label in self._labels("Raras; frecuencia no conocida.", annotations))
        )

    def test_other_entity_types_do_not_expand_frequency_words(self):
        builder = ConllBuilder(txt_dir=".", entity_types=["REACT", "ACTIVE", "SYS"])
        rows = pd.DataFrame(
            {"REACADV": ["FRECUENTE"], "PACTIVO": ["FRECUENTE"], "SISTEMA": ["FRECUENTE"]}
        )
        self.assertEqual(
            builder._collect_mentions(rows),
            {"REACT": ["FRECUENTE"], "ACTIVE": ["FRECUENTE"], "SYS": ["FRECUENTE"]},
        )

    def test_expansion_preserves_input_and_handles_empty_values(self):
        rows = pd.DataFrame({"frecuencia": [" FRECUENTE ", None, ""]})
        original = rows.copy(deep=True)
        mentions = self.builder._collect_mentions(rows)["FREQ"]
        pd.testing.assert_frame_equal(rows, original)
        self.assertEqual({mention.lower() for mention in mentions}, {"frecuente", "frecuentes"})
        self.assertEqual([len(mention) for mention in mentions], sorted(map(len, mentions), reverse=True))
        self.assertEqual(self._labels("frecuentes", []), ["O"])


if __name__ == "__main__":
    unittest.main()