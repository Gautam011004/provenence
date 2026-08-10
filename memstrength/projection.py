"""The projection module.

Strength is not stored next to a memory. Retrieval returns memories; this
module joins them against the signal store (and, for veto-capable strategies,
the approval ledger) and computes strength on the way out to the caller.

    ids -> MemoryStore.fetch_many   -> memories
        -> SignalStore.fetch_many   -> signals
        -> strategy.veto_candidates -> ApprovalStore.fetch_many (conditional)
        -> strategy.score_many      -> [(memory, strength)]

`project` is the production read path and carries no instrumentation, because
five perf_counter calls would swamp the thing being measured at small batch
sizes. `project_timed` is the same path with a phase breakdown, used only for
attribution passes. `compute_only` runs just the strategy over pre-fetched
signals, isolating arithmetic from join and round-trip cost.
"""

import time

from .models import ACTIVE
from .store import ApprovalStore

_EMPTY = frozenset()


class Projection(object):
    """Result of one instrumented projection call."""

    __slots__ = ("pairs", "t_retrieve_ns", "t_signals_ns", "t_approvals_ns", "t_score_ns")

    def __init__(self, pairs, t_retrieve_ns, t_signals_ns, t_approvals_ns, t_score_ns):
        self.pairs = pairs
        self.t_retrieve_ns = t_retrieve_ns
        self.t_signals_ns = t_signals_ns
        self.t_approvals_ns = t_approvals_ns
        self.t_score_ns = t_score_ns

    @property
    def total_ns(self):
        return (
            self.t_retrieve_ns
            + self.t_signals_ns
            + self.t_approvals_ns
            + self.t_score_ns
        )


class StrengthProjection(object):
    __slots__ = ("memories", "signals", "approvals", "strategy")

    def __init__(self, memory_store, signal_store, strategy, approval_store=None):
        self.memories = memory_store
        self.signals = signal_store
        self.strategy = strategy
        self.approvals = approval_store if approval_store is not None else ApprovalStore()

    def project(self, memory_ids, explain=False):
        """Production read path. Returns [(Memory, StrengthResult)] in input order."""
        strategy = self.strategy
        memories = self.memories.fetch_many(memory_ids)
        signals = self.signals.fetch_many(memory_ids)

        candidates = strategy.veto_candidates(signals)
        approved = self.approvals.fetch_many(candidates) if candidates else _EMPTY

        return list(zip(memories, strategy.score_many(signals, approved, explain)))

    def project_timed(self, memory_ids, explain=False):
        """Same path, instrumented per phase. For attribution only."""
        strategy = self.strategy

        t0 = time.perf_counter_ns()
        memories = self.memories.fetch_many(memory_ids)

        t1 = time.perf_counter_ns()
        signals = self.signals.fetch_many(memory_ids)

        t2 = time.perf_counter_ns()
        candidates = strategy.veto_candidates(signals)
        approved = self.approvals.fetch_many(candidates) if candidates else _EMPTY

        t3 = time.perf_counter_ns()
        results = strategy.score_many(signals, approved, explain)
        t4 = time.perf_counter_ns()

        return Projection(list(zip(memories, results)), t1 - t0, t2 - t1, t3 - t2, t4 - t3)

    def compute_only(self, signals, explain=False):
        """Strategy arithmetic over pre-fetched signals, no memory-store access.

        The approval lookup is still performed for veto-capable strategies,
        since skipping it would not be a valid implementation of the veto.
        """
        strategy = self.strategy
        candidates = strategy.veto_candidates(signals)
        approved = self.approvals.fetch_many(candidates) if candidates else _EMPTY
        return strategy.score_many(signals, approved, explain)

    def ranked(self, memory_ids, include_vetoed=False, explain=False):
        """Convenience read path: strongest first, vetoed memories dropped."""
        pairs = self.project(memory_ids, explain)
        if not include_vetoed:
            pairs = [p for p in pairs if p[1].state == ACTIVE]
        pairs.sort(key=lambda p: p[1].value, reverse=True)
        return pairs
