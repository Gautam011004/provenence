"""How the strategies respond when a memory weakens.

These pin the findings from bench/feedback.py so they cannot regress quietly.
Signals are constructed directly rather than loaded from the extraction, so the
tests run without the PDF or its output.

    python3 tests/test_feedback.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memstrength.models import ACTIVE, VETOED, Signals
from memstrength.strategies import (
    RawAdditive,
    ScorecardInline,
    equal_weight_scorecard,
)
from memstrength.veto import VetoPolicy

SUM_CAPS = {k: 12.5 for k in ("eval", "validation", "retrieval", "recency",
                              "trust", "negative", "history")}
SUM_CAPS["retrieval"] = 6.25


def capped_sum():
    return RawAdditive(w_retrieval=0.01, w_validation=0.5,
                       validation_cap=25.0, retrieval_cap=500.0,
                       contribution_caps=SUM_CAPS)


def mem(mid, **kw):
    base = dict(eval_score=0.8, validations=6, retrievals=120,
                last_used_days=20.0, source_trust=0.9)
    base.update(kw)
    return Signals(memory_id=mid, **base)


class TestComplaintsWeakenAMemory(unittest.TestCase):
    def test_every_strategy_lowers_the_score(self):
        clean = mem("a")
        complained = mem("b", negative_feedback=2)
        for name, arm in (("additive", RawAdditive()),
                          ("capped", capped_sum()),
                          ("scorecard", equal_weight_scorecard(cls=ScorecardInline))):
            a, b = arm.score_many([clean, complained])
            self.assertLess(b.value, a.value, name)

    def test_the_scorecard_vetoes_at_the_threshold(self):
        card = equal_weight_scorecard(cls=ScorecardInline, veto_negative_threshold=3)
        s = mem("a", negative_feedback=3)
        VetoPolicy(negative_threshold=3).apply(s)
        self.assertEqual(card.score_many([s])[0].state, VETOED)

    def test_the_capped_sum_cannot_tell_three_complaints_from_forty(self):
        # Once the negative contribution hits its ceiling, further complaints
        # change nothing. The scorecard saturates identically, but converts
        # saturation into removal; the sum leaves the memory sitting at the
        # bottom, indistinguishable and still retrievable.
        arm = capped_sum()
        scores = [arm.score_many([mem("a", negative_feedback=n)])[0].value
                  for n in (3, 4, 10, 40)]
        self.assertEqual(len(set(round(v, 6) for v in scores)), 1)
        # Below the ceiling it still discriminates, which is what makes the
        # saturation a ceiling effect rather than a broken penalty.
        mild = [arm.score_many([mem("a", negative_feedback=n)])[0].value
                for n in (0, 1, 2)]
        self.assertEqual(len(set(round(v, 6) for v in mild)), 3)

    def test_the_sum_never_vetoes_however_many_complaints(self):
        # Capping demotes a memory. Only the veto removes it. A capped sum
        # bottoms out but stays retrievable.
        s = mem("a", negative_feedback=40)
        for arm in (RawAdditive(), capped_sum()):
            self.assertEqual(arm.score_many([s])[0].state, ACTIVE)


class TestPopularityMasksComplaints(unittest.TestCase):
    """The finding that most separates the two approaches.

    Under an uncapped sum, retrieval volume buys immunity: a heavily used
    memory with complaints outranks a clean one that is used less. Capping
    retrieval's contribution is what removes the immunity.
    """

    def setUp(self):
        self.clean = mem("clean", retrievals=120, negative_feedback=0)
        self.popular_bad = mem("popular", retrievals=1020, negative_feedback=2)

    def test_the_raw_sum_ranks_the_complained_about_memory_higher(self):
        a, b = RawAdditive().score_many([self.clean, self.popular_bad])
        self.assertGreater(b.value, a.value)

    def test_capping_retrieval_reverses_that(self):
        for arm in (capped_sum(), equal_weight_scorecard(cls=ScorecardInline)):
            a, b = arm.score_many([self.clean, self.popular_bad])
            self.assertLess(b.value, a.value, arm.name)

    def test_retrieval_cannot_offset_a_complaint_in_the_scorecard(self):
        card = equal_weight_scorecard(cls=ScorecardInline)
        modest = mem("m", retrievals=50, negative_feedback=0)
        maxed = mem("x", retrievals=10 ** 7, negative_feedback=1)
        a, b = card.score_many([modest, maxed])
        self.assertLess(b.value, a.value)


class TestEvalFailure(unittest.TestCase):
    def test_the_scorecard_removes_it_immediately(self):
        s = mem("a", eval_failed=True, eval_score=0.2, last_failure_at=100.0)
        VetoPolicy().apply(s)
        card = equal_weight_scorecard(cls=ScorecardInline)
        self.assertEqual(card.score_many([s])[0].state, VETOED)

    def test_the_raw_sum_barely_notices(self):
        # eval_score is one bounded term among several unbounded ones, so a
        # failed eval moves the total by a few per cent.
        clean = mem("a", eval_score=0.8)
        failed = mem("b", eval_score=0.2, eval_failed=True)
        a, b = RawAdditive().score_many([clean, failed])
        self.assertGreater(b.value, a.value * 0.9)


class TestRecovery(unittest.TestCase):
    def test_validations_and_passing_evals_restore_a_complained_memory(self):
        card = equal_weight_scorecard(cls=ScorecardInline)
        before = card.score_many([mem("a", negative_feedback=2)])[0].value
        after = card.score_many(
            [mem("a", negative_feedback=2, validations=14, eval_score=0.97)]
        )[0].value
        self.assertGreater(after, before)

    def test_a_reinstated_memory_recovers_as_the_history_penalty_fades(self):
        card = equal_weight_scorecard(cls=ScorecardInline)
        fresh = mem("a", failure_count=1, days_since_failure=0.0,
                    approved_at=101.0, last_failure_at=100.0)
        healed = mem("a", failure_count=1, days_since_failure=400.0,
                     approved_at=101.0, last_failure_at=100.0)
        self.assertLess(card.score_many([fresh])[0].value,
                        card.score_many([healed])[0].value)


class TestNeglect(unittest.TestCase):
    def test_going_unused_weakens_a_memory(self):
        card = equal_weight_scorecard(cls=ScorecardInline)
        recent = mem("a", last_used_days=5.0)
        stale = mem("b", last_used_days=180.0)
        self.assertLess(card.score_many([stale])[0].value,
                        card.score_many([recent])[0].value)

    def test_scorecard_decay_saturates(self):
        # The recency term is a decaying credit, so it approaches zero rather
        # than growing without bound. Past a few half-lives, older and much
        # older are indistinguishable. Whether that is right is a policy
        # question; it is pinned here so the behaviour is at least known.
        card = equal_weight_scorecard(cls=ScorecardInline)
        recent = card.score_many([mem("r", last_used_days=5.0)])[0].value
        old = card.score_many([mem("a", last_used_days=180.0)])[0].value
        ancient = card.score_many([mem("b", last_used_days=730.0)])[0].value
        # Six months to two years costs almost nothing, while a week to six
        # months costs a lot: the decay has already bottomed out.
        self.assertLess(old - ancient, 1.0)
        self.assertGreater(recent - old, 10.0 * (old - ancient))

    def test_raw_sum_decay_does_not_saturate(self):
        # An unbounded penalty keeps growing, so a very old memory can be
        # driven arbitrarily negative by age alone.
        old = RawAdditive().score_many([mem("a", last_used_days=180.0)])[0].value
        ancient = RawAdditive().score_many([mem("b", last_used_days=3650.0)])[0].value
        self.assertLess(ancient, old - 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
