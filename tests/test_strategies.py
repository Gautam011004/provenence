"""Correctness guards, so the benchmark is not timing broken arithmetic.

    python3 tests/test_strategies.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memstrength.dataset import build_corpus, sample_batches
from memstrength.models import ACTIVE, VETOED, Memory, Signals
from memstrength.projection import StrengthProjection
from memstrength.store import ApprovalStore, MemoryStore, SignalStore
from memstrength.strategies import NormalizedScorecard, RawAdditive


def sig(**kw):
    kw.setdefault("memory_id", "m1")
    return Signals(**kw)


class TestScorecard(unittest.TestCase):
    def setUp(self):
        self.s = NormalizedScorecard()

    def test_output_is_bounded(self):
        extremes = [
            sig(memory_id="lo", eval_score=0.0, validations=0, retrievals=0,
                last_used_days=9999.0, negative_feedback=0, source_trust=0.0),
            sig(memory_id="hi", eval_score=1.0, validations=10 ** 6, retrievals=10 ** 9,
                last_used_days=0.0, negative_feedback=0, source_trust=1.0),
        ]
        for r in self.s.score_many(extremes):
            self.assertGreaterEqual(r.value, 0.0)
            self.assertLessEqual(r.value, 100.0)

    def test_caps_bound_a_runaway_signal(self):
        modest = sig(memory_id="a", eval_score=0.8, validations=25, retrievals=500,
                     last_used_days=1.0, source_trust=0.8)
        runaway = sig(memory_id="b", eval_score=0.8, validations=10 ** 6,
                      retrievals=10 ** 9, last_used_days=1.0, source_trust=0.8)
        a, b = self.s.score_many([modest, runaway])
        self.assertAlmostEqual(a.value, b.value, places=6)

    def test_eval_failure_vetoes(self):
        r = self.s.score_many([sig(eval_score=0.95, eval_failed=True, validations=40)])[0]
        self.assertEqual(r.state, VETOED)
        self.assertEqual(r.value, 0.0)

    def test_negative_feedback_at_threshold_vetoes(self):
        r = self.s.score_many([sig(eval_score=0.95, negative_feedback=3)])[0]
        self.assertEqual(r.state, VETOED)

    def test_below_threshold_penalizes_without_veto(self):
        clean = sig(memory_id="a", eval_score=0.9, negative_feedback=0)
        dinged = sig(memory_id="b", eval_score=0.9, negative_feedback=2)
        a, b = self.s.score_many([clean, dinged])
        self.assertEqual(b.state, ACTIVE)
        self.assertLess(b.value, a.value)

    def test_approval_lifts_the_veto(self):
        s = sig(memory_id="x", eval_score=0.9, eval_failed=True)
        vetoed = self.s.score_many([s])[0]
        approved = self.s.score_many([s], approved_ids=frozenset(["x"]))[0]
        self.assertEqual(vetoed.state, VETOED)
        self.assertEqual(approved.state, ACTIVE)
        self.assertGreater(approved.value, 0.0)

    def test_penalty_is_capped(self):
        # negative_feedback is huge but the score must not go below zero via
        # an unbounded penalty; it floors at 0 and stays a valid score.
        r = self.s.score_many(
            [sig(eval_score=1.0, negative_feedback=1000)], approved_ids=frozenset(["m1"])
        )[0]
        self.assertGreaterEqual(r.value, 0.0)
        self.assertLessEqual(r.value, 100.0)

    def test_veto_candidates_matches_scoring(self):
        corpus = build_corpus(size=500, veto_rate=0.3, approval_rate=0.0, seed=7)
        signals = corpus.signals.fetch_many(corpus.ids)
        candidates = set(self.s.veto_candidates(signals))
        vetoed = set(r.memory_id for r in self.s.score_many(signals) if r.state == VETOED)
        self.assertEqual(candidates, vetoed)

    def test_explain_components_sum_to_score(self):
        s = sig(eval_score=0.7, validations=5, retrievals=50,
                last_used_days=10.0, negative_feedback=1, source_trust=0.6)
        r = self.s.score_many([s], explain=True)[0]
        self.assertAlmostEqual(sum(r.components.values()) * 100.0, r.value, places=6)


class TestAdditive(unittest.TestCase):
    def setUp(self):
        self.s = RawAdditive()

    def test_never_vetoes(self):
        r = self.s.score_many([sig(eval_failed=True, negative_feedback=50)])[0]
        self.assertEqual(r.state, ACTIVE)

    def test_is_unbounded(self):
        hot = self.s.score_many([sig(eval_score=1.0, retrievals=100000)])[0]
        self.assertGreater(hot.value, 1000.0)

    def test_can_go_negative(self):
        bad = self.s.score_many(
            [sig(eval_score=0.0, negative_feedback=20, failure_count=2)]
        )[0]
        self.assertLess(bad.value, 0.0)

    def test_retrieval_count_can_outweigh_a_failed_eval(self):
        # The property that motivates the scorecard: volume drowns quality.
        failing_but_hot = sig(memory_id="a", eval_score=0.0, failure_count=3,
                              negative_feedback=10, retrievals=500)
        clean_but_cold = sig(memory_id="b", eval_score=1.0, validations=5, retrievals=1)
        a, b = self.s.score_many([failing_but_hot, clean_but_cold])
        self.assertGreater(a.value, b.value)

    def test_explain_components_sum_to_score(self):
        s = sig(eval_score=0.7, validations=5, retrievals=50, last_used_days=10.0,
                negative_feedback=1, source_trust=0.6, failure_count=2)
        r = self.s.score_many([s], explain=True)[0]
        self.assertAlmostEqual(sum(r.components.values()), r.value, places=6)


class TestProjection(unittest.TestCase):
    def setUp(self):
        self.corpus = build_corpus(size=1000, veto_rate=0.2, approval_rate=0.5, seed=11)

    def _proj(self, strategy):
        return StrengthProjection(
            self.corpus.memories, self.corpus.signals, strategy, self.corpus.approvals
        )

    def test_memory_record_carries_no_strength(self):
        m = self.corpus.memories.fetch_many(["m000001"])[0]
        self.assertFalse(hasattr(m, "strength"))
        self.assertEqual(set(Memory.__slots__), {"id", "text", "created_at_days"})

    def test_projection_preserves_input_order(self):
        ids = ["m000005", "m000001", "m000009"]
        pairs = self._proj(NormalizedScorecard()).project(ids)
        self.assertEqual([m.id for m, _ in pairs], ids)
        self.assertEqual([r.memory_id for _, r in pairs], ids)

    def test_compute_only_matches_full_path(self):
        ids = sample_batches(self.corpus, 200, 1, seed=3)[0]
        for strategy in (RawAdditive(), NormalizedScorecard()):
            proj = self._proj(strategy)
            full = [r.value for _, r in proj.project(ids)]
            signals = self.corpus.signals.fetch_many(ids)
            only = [r.value for r in proj.compute_only(signals)]
            self.assertEqual(full, only)

    def test_timed_path_agrees_with_production_path(self):
        ids = sample_batches(self.corpus, 100, 1, seed=4)[0]
        proj = self._proj(NormalizedScorecard())
        a = [(m.id, r.value, r.state) for m, r in proj.project(ids)]
        b = [(m.id, r.value, r.state) for m, r in proj.project_timed(ids).pairs]
        self.assertEqual(a, b)

    def test_additive_skips_the_approval_round_trip(self):
        ids = sample_batches(self.corpus, 200, 1, seed=5)[0]
        approvals = ApprovalStore(["m000001"])
        proj = StrengthProjection(
            self.corpus.memories, self.corpus.signals, RawAdditive(), approvals
        )
        proj.project(ids)
        self.assertEqual(approvals.calls, 0)

    def test_scorecard_makes_one_approval_round_trip_when_candidates_exist(self):
        ids = sample_batches(self.corpus, 200, 1, seed=6)[0]
        approvals = ApprovalStore(self.corpus.approvals.fetch_many(self.corpus.ids))
        approvals.calls = 0
        proj = StrengthProjection(
            self.corpus.memories, self.corpus.signals, NormalizedScorecard(), approvals
        )
        proj.project(ids)
        self.assertEqual(approvals.calls, 1)

    def test_no_candidates_means_no_approval_round_trip(self):
        clean = build_corpus(size=100, veto_rate=0.0, seed=12)
        approvals = ApprovalStore()
        proj = StrengthProjection(
            clean.memories, clean.signals, NormalizedScorecard(), approvals
        )
        proj.project(clean.ids)
        self.assertEqual(approvals.calls, 0)

    def test_ranked_drops_vetoed_and_sorts_descending(self):
        ids = sample_batches(self.corpus, 300, 1, seed=8)[0]
        pairs = self._proj(NormalizedScorecard()).ranked(ids)
        values = [r.value for _, r in pairs]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertTrue(all(r.state == ACTIVE for _, r in pairs))
        self.assertLess(len(pairs), len(ids))


class TestCorpus(unittest.TestCase):
    def test_veto_rate_is_honored(self):
        corpus = build_corpus(size=5000, veto_rate=0.25, approval_rate=0.4, seed=21)
        observed = corpus.n_veto_candidates / float(len(corpus.ids))
        self.assertAlmostEqual(observed, 0.25, delta=0.02)

    def test_candidates_are_what_the_scorecard_would_flag(self):
        corpus = build_corpus(size=2000, veto_rate=0.25, seed=22)
        signals = corpus.signals.fetch_many(corpus.ids)
        flagged = NormalizedScorecard().veto_candidates(signals)
        self.assertEqual(len(flagged), corpus.n_veto_candidates)

    def test_corpus_is_deterministic(self):
        a = build_corpus(size=200, seed=5).signals.fetch_many(["m000007"])[0]
        b = build_corpus(size=200, seed=5).signals.fetch_many(["m000007"])[0]
        self.assertEqual(a.eval_score, b.eval_score)
        self.assertEqual(a.retrievals, b.retrievals)


class TestStoreLatencySimulation(unittest.TestCase):
    def test_simulated_rtt_actually_costs_time(self):
        import time

        store = SignalStore([Signals("m1")], rtt_us=500.0)
        t0 = time.perf_counter_ns()
        store.fetch_many(["m1"])
        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        self.assertGreaterEqual(elapsed_us, 450.0)

    def test_zero_rtt_is_not_penalized(self):
        store = MemoryStore([Memory("m1", "t", 0.0)], rtt_us=0.0)
        self.assertEqual(len(store.fetch_many(["m1"])), 1)




# --------------------------------------------------------------- veto policy


from memstrength.strategies import ScorecardInline  # noqa: E402
from memstrength.veto import PartitionedMemoryStore, VetoPolicy  # noqa: E402
from memstrength.dataset import build_quality_corpus  # noqa: E402


class TestVetoPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = VetoPolicy(negative_threshold=3)

    def test_clean_record_is_not_vetoed(self):
        self.assertFalse(self.policy.should_veto(sig(eval_score=0.9)))

    def test_eval_failure_vetoes(self):
        self.assertTrue(self.policy.should_veto(sig(eval_failed=True, last_failure_at=10.0)))

    def test_negative_threshold_vetoes(self):
        self.assertTrue(self.policy.should_veto(sig(negative_feedback=3, last_failure_at=10.0)))

    def test_approval_after_failure_lifts_veto(self):
        s = sig(eval_failed=True, last_failure_at=10.0, approved_at=11.0)
        self.assertFalse(self.policy.should_veto(s))

    def test_approval_before_failure_does_not_lift_veto(self):
        # The memory was cleared, then failed again. The stale approval must
        # not carry over -- this is the bug a bare `approved` boolean has.
        s = sig(eval_failed=True, approved_at=5.0, last_failure_at=10.0)
        self.assertTrue(self.policy.should_veto(s))

    def test_reapproval_after_a_second_failure_lifts_it_again(self):
        s = sig(eval_failed=True, approved_at=5.0, last_failure_at=10.0)
        self.assertTrue(self.policy.should_veto(s))
        s.approved_at = 12.0
        self.assertFalse(self.policy.should_veto(s))

    def test_apply_reports_whether_standing_changed(self):
        s = sig(eval_failed=True, last_failure_at=1.0)
        self.assertTrue(self.policy.apply(s))
        self.assertTrue(s.vetoed)
        self.assertFalse(self.policy.apply(s))


class TestPartition(unittest.TestCase):
    def setUp(self):
        self.corpus = build_corpus(size=2000, veto_rate=0.25, approval_rate=0.3, seed=31)
        self.part = self.corpus.partition()

    def test_invariant_quarantine_holds_exactly_the_vetoed(self):
        for m in self.part.active.fetch_many(self.part.active.ids()):
            self.assertFalse(self.corpus.signals.fetch_many([m.id])[0].vetoed)
        for m in self.part.quarantine.fetch_many(self.part.quarantine.ids()):
            self.assertTrue(self.corpus.signals.fetch_many([m.id])[0].vetoed)

    def test_every_memory_is_in_exactly_one_table(self):
        counts = self.part.counts()
        self.assertEqual(counts["active"] + counts["quarantine"], len(self.corpus.ids))
        overlap = set(self.part.active.ids()) & set(self.part.quarantine.ids())
        self.assertEqual(overlap, set())

    def test_retrieval_never_returns_a_vetoed_memory(self):
        ids = self.part.active_ids()[:500]
        for m in self.part.fetch_many(ids):
            self.assertFalse(self.corpus.signals.fetch_many([m.id])[0].vetoed)

    def test_pending_approval_is_the_quarantine_table(self):
        pending = self.part.pending_approval()
        self.assertEqual(len(pending), self.part.counts()["quarantine"])
        for m in pending:
            self.assertTrue(self.corpus.signals.fetch_many([m.id])[0].vetoed)

    def test_failure_moves_a_memory_to_quarantine(self):
        mid = self.part.active_ids()[0]
        s = self.corpus.signals.fetch_many([mid])[0]
        s.eval_failed = True
        self.assertTrue(self.part.record_failure(mid, at_day=500.0))
        self.assertIn(mid, self.part.quarantine)
        self.assertNotIn(mid, self.part.active)

    def test_approval_moves_it_back(self):
        mid = self.part.quarantine.ids()[0]
        self.assertTrue(self.part.approve(mid, at_day=10000.0))
        self.assertIn(mid, self.part.active)
        self.assertNotIn(mid, self.part.quarantine)

    def test_transition_is_idempotent(self):
        mid = self.part.active_ids()[0]
        before = self.part.transitions
        self.assertFalse(self.part.apply_signal_change(mid))
        self.assertEqual(self.part.transitions, before)

    def test_resweep_finds_nothing_when_already_consistent(self):
        self.assertEqual(self.part.resweep(), 0)

    def test_resweep_repairs_drift(self):
        mid = self.part.active_ids()[0]
        s = self.corpus.signals.fetch_many([mid])[0]
        s.eval_failed = True
        s.last_failure_at = 900.0  # signal changed without notifying the partition
        self.assertIn(mid, self.part.active)
        self.assertEqual(self.part.resweep(), 1)
        self.assertIn(mid, self.part.quarantine)


class TestScorecardInline(unittest.TestCase):
    def setUp(self):
        self.inline = ScorecardInline()
        self.readtime = NormalizedScorecard()

    def test_makes_no_approval_round_trip(self):
        corpus = build_corpus(size=500, veto_rate=0.4, seed=41)
        approvals = ApprovalStore()
        proj = StrengthProjection(
            corpus.memories, corpus.signals, self.inline, approvals
        )
        proj.project(corpus.ids)
        self.assertEqual(approvals.calls, 0)

    def test_scores_identically_to_the_read_time_version(self):
        corpus = build_corpus(size=1000, veto_rate=0.3, approval_rate=0.4, seed=42)
        signals = corpus.signals.fetch_many(corpus.ids)
        approved = corpus.approvals.fetch_many(self.readtime.veto_candidates(signals))
        a = [(r.memory_id, r.value, r.state) for r in self.readtime.score_many(signals, approved)]
        b = [(r.memory_id, r.value, r.state) for r in self.inline.score_many(signals)]
        self.assertEqual(a, b)

    def test_honours_the_vetoed_flag(self):
        s = sig(eval_score=0.95, vetoed=True)
        self.assertEqual(self.inline.score_many([s])[0].state, VETOED)

    def test_fails_closed_if_the_partition_drifts(self):
        # A vetoed record that somehow reached the active table must still not
        # be served with a positive strength.
        s = sig(eval_score=1.0, validations=40, retrievals=900, vetoed=True)
        r = self.inline.score_many([s])[0]
        self.assertEqual(r.state, VETOED)
        self.assertEqual(r.value, 0.0)


class TestQualityCorpus(unittest.TestCase):
    def setUp(self):
        self.corpus = build_quality_corpus(size=3000, seed=77)

    def test_signals_track_latent_quality(self):
        from bench.quality import spearman

        q = [self.corpus.true_quality[s.memory_id] for s in self.corpus.signals]
        evals = [s.eval_score for s in self.corpus.signals]
        self.assertGreater(spearman(evals, q), 0.6)

    def test_popularity_is_not_determined_by_quality(self):
        # The confound must be real: high popularity must not imply high
        # quality, or the comparison is rigged toward traffic-chasing scorers.
        from bench.quality import spearman

        q = [self.corpus.true_quality[s.memory_id] for s in self.corpus.signals]
        pop = [self.corpus.popularity[s.memory_id] for s in self.corpus.signals]
        self.assertLess(spearman(pop, q), 0.5)

        top_pop = sorted(self.corpus.ids, key=lambda i: self.corpus.popularity[i], reverse=True)
        worst = min(self.corpus.true_quality[i] for i in top_pop[:200])
        self.assertLess(worst, 0.4)

    def test_active_signals_excludes_vetoed(self):
        active = self.corpus.active_signals()
        self.assertTrue(all(not s.vetoed for s in active))
        self.assertLess(len(active), len(self.corpus.signals))



class TestReinstatement(unittest.TestCase):
    """An approved memory is retrieved like any other. Only its score marks it.

    The failure history must not survive as a gate -- no lingering flag, no
    re-check, no second lookup. It survives as a penalty on strength that fades
    as the memory stays clean.
    """

    def _clean(self, **kw):
        base = dict(eval_score=0.9, validations=10, retrievals=100,
                    last_used_days=5.0, source_trust=0.8)
        base.update(kw)
        return sig(**base)

    def setUp(self):
        self.inline = ScorecardInline()
        self.additive = RawAdditive()

    def test_reinstated_memory_is_active_not_vetoed(self):
        s = self._clean(memory_id="r", eval_failed=True, last_failure_at=10.0,
                        approved_at=11.0, failure_count=1, days_since_failure=1.0)
        VetoPolicy().apply(s)
        self.assertFalse(s.vetoed)
        self.assertEqual(self.inline.score_many([s])[0].state, ACTIVE)

    def test_reinstated_memory_is_retrieved(self):
        corpus = build_corpus(size=800, veto_rate=0.4, approval_rate=1.0, seed=61)
        part = corpus.partition()
        # Every candidate was approved, so nothing should be quarantined.
        self.assertEqual(part.counts()["quarantine"], 0)
        reinstated = [
            mid for mid in part.active_ids()
            if corpus.signals.fetch_many([mid])[0].failure_count > 0
        ]
        self.assertGreater(len(reinstated), 0)
        fetched = part.fetch_many(reinstated[:50])
        self.assertEqual(len(fetched), min(50, len(reinstated)))

    def test_reinstated_scores_lower_than_an_identical_clean_memory(self):
        clean = self._clean(memory_id="a")
        reinstated = self._clean(memory_id="b", failure_count=1, days_since_failure=0.0)
        a, b = self.inline.score_many([clean, reinstated])
        self.assertEqual(b.state, ACTIVE)
        self.assertLess(b.value, a.value)

    def test_penalty_fades_as_the_memory_stays_clean(self):
        fresh = self._clean(memory_id="a", failure_count=1, days_since_failure=0.0)
        aging = self._clean(memory_id="b", failure_count=1, days_since_failure=45.0)
        old = self._clean(memory_id="c", failure_count=1, days_since_failure=400.0)
        clean = self._clean(memory_id="d")
        f, a, o, c = self.inline.score_many([fresh, aging, old, clean])
        self.assertLess(f.value, a.value)
        self.assertLess(a.value, o.value)
        self.assertAlmostEqual(o.value, c.value, places=1)

    def test_repeat_offenders_are_penalized_more_but_the_penalty_is_capped(self):
        once = self._clean(memory_id="a", failure_count=1, days_since_failure=0.0)
        thrice = self._clean(memory_id="b", failure_count=3, days_since_failure=0.0)
        many = self._clean(memory_id="c", failure_count=50, days_since_failure=0.0)
        o, t, m = self.inline.score_many([once, thrice, many])
        self.assertLess(t.value, o.value)
        self.assertAlmostEqual(m.value, t.value, places=6)
        self.assertGreaterEqual(m.value, 0.0)

    def test_additive_also_records_history(self):
        clean = self._clean(memory_id="a")
        reinstated = self._clean(memory_id="b", failure_count=1, days_since_failure=0.0)
        a, b = self.additive.score_many([clean, reinstated])
        self.assertLess(b.value, a.value)

    def test_additive_history_penalty_never_fades(self):
        # Approach A has no decay term, so a single old failure marks a memory
        # permanently. Recorded as a difference, not a defect.
        fresh = self._clean(memory_id="a", failure_count=1, days_since_failure=0.0)
        old = self._clean(memory_id="b", failure_count=1, days_since_failure=1000.0)
        f, o = self.additive.score_many([fresh, old])
        self.assertEqual(f.value, o.value)

    def test_reinstated_memory_costs_no_extra_lookup(self):
        corpus = build_corpus(size=500, veto_rate=0.5, approval_rate=1.0, seed=62)
        part = corpus.partition()
        approvals = ApprovalStore()
        proj = StrengthProjection(part, corpus.signals, self.inline, approvals)
        proj.project(part.active_ids()[:200])
        self.assertEqual(approvals.calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
