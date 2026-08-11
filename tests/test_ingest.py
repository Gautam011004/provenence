"""Ingest tests. Stdlib only -- no pypdf, no PDF fixture.

Pages are hand-written from what the real extractor produces, so these run
anywhere and pin the segmentation judgements rather than the PDF library.

    python3 tests/test_ingest.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.admit import admit, dedupe, load_stores, memory_id
from ingest.pdf import RawPage
from ingest.segment import Candidate, despace, is_heading, segment, segment_page, split_footer
from memstrength.store import MemoryStore, SignalStore
from memstrength.strategies import equal_weight_scorecard, ScorecardInline


FOOTER = ["ASSETS BRAND MANUAL EDITION 12", "offIcIAl SlogAn 43"]


def page(number, *lines):
    return RawPage(number, list(lines) + FOOTER)


class TestFooter(unittest.TestCase):
    def test_footer_is_stripped_and_yields_the_section(self):
        body, section = split_footer(["Real content here."] + FOOTER)
        self.assertEqual(body, ["Real content here."])
        self.assertEqual(section, "Official Slogan")

    def test_small_caps_section_names_are_normalised(self):
        _, section = split_footer(["x", "ASSETS BRAND MANUAL EDITION 12",
                                   "hOST cOUNTRy EmBLEmS 61"])
        self.assertEqual(section, "Host Country Emblems")

    def test_a_page_without_a_footer_keeps_all_its_lines(self):
        body, section = split_footer(["a", "b"])
        self.assertEqual(body, ["a", "b"])
        self.assertIsNone(section)


class TestHeadings(unittest.TestCase):
    def test_all_caps_short_line_is_a_heading(self):
        self.assertTrue(is_heading("LEGAL MARKING"))

    def test_a_sentence_is_not_a_heading(self):
        self.assertFalse(is_heading("The Legal Notice is provided with all files."))

    def test_a_long_all_caps_line_is_not_a_heading(self):
        self.assertFalse(is_heading("THIS LINE IS FAR TOO LONG TO BE A HEADING IN THIS DOCUMENT"))


class TestProse(unittest.TestCase):
    def test_wrapped_lines_are_rejoined_into_one_memory(self):
        out, _ = segment_page(page(43,
            "LEGAL MARKING",
            "The Legal Notice TM is provided with all Artwork files,",
            "in the position and size best suited for the intended",
            "use of the Official Slogan."))
        self.assertEqual(len(out), 1)
        self.assertIn("Artwork files, in the position", out[0].text)
        self.assertEqual(out[0].kind, "guideline")
        self.assertEqual(out[0].heading, "LEGAL MARKING")

    def test_two_sentences_become_two_memories(self):
        out, _ = segment_page(page(43,
            "The Legal Notice is provided with every Artwork file.",
            "The minimum acceptable font size is three points."))
        self.assertEqual(len(out), 2)

    def test_a_sentence_below_the_minimum_length_is_dropped(self):
        # Short fragments are almost always captions or labels rather than
        # anything assertable.
        out, _ = segment_page(page(43, "The first rule is short."))
        self.assertEqual(out, [])

    def test_placeholder_copy_is_dropped(self):
        out, _ = segment_page(page(94,
            "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam."))
        self.assertEqual(out, [])

    def test_caption_grids_are_dropped(self):
        # A run of layout labels with no terminal punctuation is not a claim
        # anyone could later be right or wrong about.
        out, _ = segment_page(page(61,
            "Colour tonal pattern fill 4 Colour tonal 4 Colour flat Pantone solid"))
        self.assertEqual(out, [])

    def test_short_fragments_are_dropped(self):
        out, _ = segment_page(page(43, "Too short."))
        self.assertEqual(out, [])

    def test_repeated_image_labels_are_collapsed(self):
        out, _ = segment_page(page(111,
            "FIFA PARTNER", "FIFA PARTNER", "FIFA PARTNER",
            "To ensure visibility, Composite Logos have to be built correctly."))
        self.assertEqual(len(out), 1)


class TestRules(unittest.TestCase):
    def test_a_prohibition_is_classified_as_a_rule(self):
        out, _ = segment_page(page(111, "Don't omit the Legal Notice from the artwork."))
        self.assertEqual(out[0].kind, "rule")

    def test_a_typographic_apostrophe_still_matches(self):
        # The manual is typeset with U+2019, not an ASCII quote. Matching only
        # ASCII files most of the document's prohibitions as ordinary prose.
        out, _ = segment_page(page(111, u"Don’t omit the Legal Notice from the artwork."))
        self.assertEqual(out[0].kind, "rule")

    def test_other_imperatives_are_rules_too(self):
        for opener in ("Never use", "Always place", "Avoid placing", "Ensure that"):
            out, _ = segment_page(page(28, "%s the emblem over a busy photograph." % opener))
            self.assertEqual(out[0].kind, "rule", opener)

    def test_ordinary_prose_is_not_a_rule(self):
        out, _ = segment_page(page(43, "The Legal Notice is provided with all Artwork files."))
        self.assertEqual(out[0].kind, "guideline")


class TestSpecs(unittest.TestCase):
    def test_a_colour_table_becomes_one_memory(self):
        out, _ = segment_page(page(21,
            "WHITE",
            "CMYK 0/0/0/0", "RGB 255/255/255", "HEX #FFFFFF", "PMS White C"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].kind, "spec")
        self.assertTrue(out[0].text.startswith("WHITE:"))
        self.assertIn("HEX #FFFFFF", out[0].text)
        self.assertEqual(out[0].confidence, 1.0)

    def test_two_colours_in_columns_are_paired_with_their_names(self):
        out, _ = segment_page(page(21,
            "WHITE", "BLACK",
            "CMYK 0/0/0/0", "RGB 255/255/255",
            "CMYK 0/0/0/100", "RGB 0/0/0"))
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].text.startswith("WHITE:"))
        self.assertTrue(out[1].text.startswith("BLACK:"))
        self.assertEqual(out[0].confidence, 1.0)

    def test_unresolvable_attribution_is_flagged_not_asserted(self):
        # Three names but two value blocks: which name goes with which swatch
        # cannot be recovered from the text alone.
        out, _ = segment_page(page(60,
            "CANADA", "MEXICO", "USA",
            "CMYK 10/100/75/50", "RGB 117/19/18",
            "CMYK 0/100/99/0", "RGB 213/1/1"))
        self.assertEqual(len(out), 2)
        for c in out:
            self.assertLess(c.confidence, 1.0)
            self.assertIn("of 2", c.text)

    def test_letter_spaced_names_are_rejoined(self):
        self.assertEqual(despace("S I LV E R"), "SILVER")

    def test_ordinary_names_are_left_alone(self):
        self.assertEqual(despace("HOST COUNTRY EMBLEMS"), "HOST COUNTRY EMBLEMS")

    def test_a_lone_spec_row_is_not_a_memory(self):
        out, _ = segment_page(page(21, "WHITE", "CMYK 0/0/0/0"))
        self.assertEqual(out, [])


class TestSectionCarryForward(unittest.TestCase):
    def test_a_page_without_a_footer_inherits_the_previous_section(self):
        pages = [
            page(43, "The Legal Notice is provided with all Artwork files."),
            RawPage(44, ["A full bleed artwork page with one real sentence on it."]),
        ]
        out = list(segment(pages))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].section, "Official Slogan")
        self.assertEqual(out[1].section, "Official Slogan")


class TestAdmit(unittest.TestCase):
    def _cands(self):
        return [
            Candidate("Don't omit the Legal Notice.", "rule", "Stakeholders", 111),
            Candidate("Don't omit the Legal Notice.", "rule", "Wordmark", 86),
            Candidate("The emblem must not be rotated.", "guideline", "Official Emblem", 28),
        ]

    def test_duplicates_are_merged(self):
        self.assertEqual(len(list(dedupe(self._cands()))), 2)

    def test_ids_are_stable_across_runs(self):
        a = list(admit(self._cands(), "fifa"))
        b = list(admit(self._cands(), "fifa"))
        self.assertEqual([x.memory.id for x in a], [x.memory.id for x in b])

    def test_ids_are_keyed_on_text_not_page(self):
        # Re-paginating a later edition must not orphan every memory.
        c1 = Candidate("Don't omit the Legal Notice.", "rule", "Stakeholders", 111)
        c2 = Candidate("Don't omit the Legal Notice.", "rule", "Stakeholders", 999)
        self.assertEqual(memory_id("fifa", c1), memory_id("fifa", c2))

    def test_different_sources_do_not_collide(self):
        c = Candidate("Don't omit the Legal Notice.", "rule", "S", 1)
        self.assertNotEqual(memory_id("fifa", c), memory_id("uefa", c))

    def test_memory_carries_no_strength(self):
        a = list(admit(self._cands(), "fifa"))[0]
        self.assertFalse(hasattr(a.memory, "strength"))

    def test_cold_start_signals_are_a_marked_prior_not_a_measurement(self):
        a = list(admit(self._cands(), "fifa"))[0]
        self.assertTrue(a.provisional)
        self.assertEqual(a.signals.retrievals, 0)
        self.assertEqual(a.signals.validations, 0)
        self.assertEqual(a.signals.negative_feedback, 0)
        self.assertEqual(a.signals.eval_score, 0.5)
        self.assertFalse(a.signals.vetoed)

    def test_extraction_confidence_discounts_source_trust(self):
        certain = Candidate("A cleanly parsed sentence about the emblem.", "guideline", "S", 1)
        shaky = Candidate("CANADA (1 of 3): CMYK 10/100/75/50; RGB 117/19/18",
                          "spec", "S", 60, confidence=0.4)
        a, b = list(admit([certain, shaky], "fifa"))
        self.assertAlmostEqual(a.signals.source_trust, 0.9, places=6)
        self.assertAlmostEqual(b.signals.source_trust, 0.36, places=6)

    def test_load_stores_populates_both(self):
        memories, signals = MemoryStore(), SignalStore()
        n = load_stores(admit(self._cands(), "fifa"), memories, signals)
        self.assertEqual(n, 2)
        self.assertEqual(len(memories), 2)
        self.assertEqual(len(signals), 2)


class TestIngestFeedsTheStrengthModel(unittest.TestCase):
    """The point of the pipeline: extracted memories score without special-casing."""

    def setUp(self):
        cands = [
            Candidate("Don't omit the Legal Notice.", "rule", "Stakeholders", 111),
            Candidate("The emblem must not be rotated or distorted.", "guideline",
                      "Official Emblem", 28),
            Candidate("CANADA (1 of 3): CMYK 10/100/75/50", "spec", "Host", 60,
                      confidence=0.4),
        ]
        self.admitted = list(admit(cands, "fifa"))
        self.card = equal_weight_scorecard(cls=ScorecardInline)

    def test_fresh_memories_score_without_error(self):
        results = self.card.score_many([a.signals for a in self.admitted])
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertGreaterEqual(r.value, 0.0)
            self.assertLessEqual(r.value, self.card.max_score())

    def test_nothing_is_vetoed_on_arrival(self):
        for r in self.card.score_many([a.signals for a in self.admitted]):
            self.assertEqual(r.state, "ACTIVE")

    def test_a_shakily_extracted_record_scores_below_a_clean_one(self):
        by_id = {a.memory.id: r for a, r in
                 zip(self.admitted, self.card.score_many([a.signals for a in self.admitted]))}
        clean = by_id[self.admitted[0].memory.id].value
        shaky = by_id[self.admitted[2].memory.id].value
        self.assertLess(shaky, clean)



class TestColdStartCarriesNoInformation(unittest.TestCase):
    """The pipeline's real limitation, pinned so nobody mistakes it for a score.

    Every signal the strength model reads is accumulated evidence: evals that
    ran, validations given, retrievals counted, complaints received. Freshly
    extracted memories have none, so they all receive the same prior and all
    score the same. Ranking them is ranking noise.

    Strength becomes meaningful only once evals run and usage accrues. Until
    then the number is the prior read back.
    """

    def setUp(self):
        cands = [
            Candidate("Don't omit the Legal Notice from any artwork.", "rule", "S", 111),
            Candidate("The emblem must not be rotated or otherwise distorted.",
                      "guideline", "E", 28),
            Candidate("The minimum acceptable font size for the notice is 3 points.",
                      "guideline", "L", 43),
        ]
        self.admitted = list(admit(cands, "fifa"))
        self.card = equal_weight_scorecard(cls=ScorecardInline)

    def test_every_cleanly_extracted_memory_scores_identically(self):
        values = set(round(r.value, 6) for r in
                     self.card.score_many([a.signals for a in self.admitted]))
        self.assertEqual(len(values), 1)

    def test_the_only_thing_that_separates_them_is_extraction_confidence(self):
        shaky = Candidate("CANADA (1 of 3): CMYK 10/100/75/50", "spec", "H", 60,
                          confidence=0.4)
        mixed = list(admit([shaky], "fifa")) + self.admitted
        values = set(round(r.value, 6) for r in
                     self.card.score_many([a.signals for a in mixed]))
        self.assertEqual(len(values), 2)

    def test_a_validated_memory_immediately_outranks_an_untouched_one(self):
        # Once real evidence lands the score starts discriminating, which is
        # the point: the prior is a starting line, not a verdict.
        a, b = self.admitted[0], self.admitted[1]
        b.signals.validations = 6
        b.signals.eval_score = 0.95
        ra, rb = self.card.score_many([a.signals, b.signals])
        self.assertGreater(rb.value, ra.value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
