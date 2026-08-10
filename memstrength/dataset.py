"""Deterministic synthetic corpora.

Two generators, because the two questions need different things from the data.

build_corpus  -- for latency. Signal semantics do not matter; what matters is
                 the distribution and, above all, a directly controllable veto
                 rate. `veto_rate` is the fraction of memories that trip a veto
                 condition, `approval_rate` the fraction of those already
                 re-approved (so they land back in the active table and pay the
                 full arithmetic cost).

build_quality_corpus -- for quality. Here semantics are the whole point. Each
                 memory gets a latent `true_quality` that no scorer sees, and
                 the signals are noisy, biased observations of it. Retrieval
                 frequency is deliberately driven mostly by an independent
                 `popularity` term, because that confound is the thing the two
                 approaches disagree about.
"""

import math
import random

from .models import Memory, Signals
from .store import MemoryStore, SignalStore, ApprovalStore
from .veto import VetoPolicy, PartitionedMemoryStore


class Corpus(object):
    __slots__ = (
        "memories",
        "signals",
        "approvals",
        "ids",
        "veto_rate",
        "approval_rate",
        "n_veto_candidates",
        "n_approved",
    )

    def __init__(self, memories, signals, approvals, ids, veto_rate, approval_rate,
                 n_veto_candidates, n_approved):
        self.memories = memories
        self.signals = signals
        self.approvals = approvals
        self.ids = ids
        self.veto_rate = veto_rate
        self.approval_rate = approval_rate
        self.n_veto_candidates = n_veto_candidates
        self.n_approved = n_approved

    def partition(self, policy=None, rtt_us=0.0):
        """Split into active and quarantine tables per the veto policy.

        This is the write-time placement that the partitioned read path relies
        on: after this, the active table holds exactly the non-vetoed memories.
        """
        policy = policy if policy is not None else VetoPolicy()
        active, quarantined = [], []
        for m in self.memories.fetch_many(self.ids):
            s = self.signals.fetch_many([m.id])[0]
            policy.apply(s)
            (quarantined if s.vetoed else active).append(m)
        return PartitionedMemoryStore(
            MemoryStore(active, rtt_us=rtt_us),
            MemoryStore(quarantined, rtt_us=rtt_us),
            self.signals,
            policy,
        )


class QualityCorpus(object):
    """A corpus plus the ground truth the scorers are trying to recover."""

    __slots__ = ("signals", "ids", "true_quality", "popularity")

    def __init__(self, signals, ids, true_quality, popularity):
        self.signals = signals
        self.ids = ids
        self.true_quality = true_quality
        self.popularity = popularity

    def active_signals(self, policy=None):
        """Signals for the non-vetoed memories only.

        The quality comparison runs on the active table, since under the
        partitioned design that is the only set either scorer ever ranks.
        """
        policy = policy if policy is not None else VetoPolicy()
        out = []
        for s in self.signals:
            policy.apply(s)
            if not s.vetoed:
                out.append(s)
        return out


def _poisson(rng, lam):
    """Knuth's algorithm. Fine for the small lambdas used here."""
    if lam <= 0.0:
        return 0
    target = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= target:
            return k
        k += 1


def _clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def build_corpus(
    size=20000,
    veto_rate=0.15,
    approval_rate=0.30,
    veto_negative_threshold=3,
    seed=1337,
    memory_rtt_us=0.0,
    signal_rtt_us=0.0,
    approval_rtt_us=0.0,
):
    rng = random.Random(seed)

    memories = []
    signals = []
    approved = []
    ids = []
    n_candidates = 0
    n_approved = 0

    for i in range(size):
        mid = "m%06d" % i
        ids.append(mid)
        memories.append(
            Memory(mid, "memory body %d" % i, created_at_days=rng.uniform(0.0, 720.0))
        )

        is_candidate = rng.random() < veto_rate
        failure_at = -1.0
        approved_at = -1.0
        failure_count = 0
        days_since_failure = 1e9
        if is_candidate:
            n_candidates += 1
            failure_at = rng.uniform(0.0, 100.0)
            failure_count = rng.randint(1, 3)
            days_since_failure = rng.expovariate(1.0 / 60.0)
            # Split candidates between the two veto triggers.
            if rng.random() < 0.5:
                eval_failed = True
                eval_score = rng.uniform(0.0, 0.4)
                negative = rng.randint(0, veto_negative_threshold - 1)
            else:
                eval_failed = False
                eval_score = rng.uniform(0.2, 0.8)
                negative = rng.randint(veto_negative_threshold, veto_negative_threshold + 4)
            if rng.random() < approval_rate:
                # Approval must postdate the failure to count.
                approved_at = failure_at + rng.uniform(0.1, 10.0)
                approved.append(mid)
                n_approved += 1
        else:
            eval_failed = False
            eval_score = rng.uniform(0.55, 1.0)
            negative = rng.randint(0, max(0, veto_negative_threshold - 1))

        signals.append(
            Signals(
                memory_id=mid,
                eval_score=eval_score,
                eval_failed=eval_failed,
                validations=rng.randint(0, 40),
                retrievals=rng.randint(0, 800),
                last_used_days=rng.expovariate(1.0 / 25.0),
                negative_feedback=negative,
                source_trust=rng.uniform(0.1, 1.0),
                last_failure_at=failure_at,
                approved_at=approved_at,
                failure_count=failure_count,
                days_since_failure=days_since_failure,
            )
        )

    policy = VetoPolicy(negative_threshold=veto_negative_threshold)
    for s in signals:
        policy.apply(s)

    return Corpus(
        MemoryStore(memories, rtt_us=memory_rtt_us),
        SignalStore(signals, rtt_us=signal_rtt_us),
        ApprovalStore(approved, rtt_us=approval_rtt_us),
        ids,
        veto_rate,
        approval_rate,
        n_candidates,
        n_approved,
    )


def build_quality_corpus(size=20000, seed=4242, veto_negative_threshold=3):
    """Signals generated from a latent quality the scorers never see.

    The generative story, and the bias each signal carries:

      true_quality  latent, uniform-ish via a Beta(2,2) shape
      popularity    how much traffic the memory gets; only weakly tied to
                    quality, which is the confound under test
      eval_score    quality plus measurement noise -- the cleanest signal
      validations   Poisson in quality -- good memories get confirmed more
      retrievals    driven by popularity, heavy tailed and unbounded
      last_used_days recency, a function of popularity rather than quality
      negative_feedback Poisson in (1 - quality)
      source_trust  quality plus heavy noise -- a weak signal
    """
    rng = random.Random(seed)

    signals = []
    ids = []
    true_quality = {}
    popularity = {}

    for i in range(size):
        mid = "q%06d" % i
        ids.append(mid)

        q = rng.betavariate(2.0, 2.0)
        # Popularity leans on quality but is not bounded by it: the noise is
        # Gaussian rather than a convex mixture, so a genuinely bad memory can
        # still be extremely popular. A mixture like 0.25*q + 0.75*rand would
        # make top popularity arithmetically imply high quality, which quietly
        # rigs the comparison in favour of any scorer that chases traffic.
        pop = _clamp01(rng.gauss(0.30 + 0.35 * q, 0.30))
        true_quality[mid] = q
        popularity[mid] = pop

        eval_score = _clamp01(q + rng.gauss(0.0, 0.12))
        eval_failed = q < 0.25 and rng.random() < 0.6
        negative = _poisson(rng, 4.0 * (1.0 - q))

        failure_at = -1.0
        approved_at = -1.0
        failure_count = 0
        days_since_failure = 1e9
        if eval_failed or negative >= veto_negative_threshold:
            failure_at = rng.uniform(0.0, 100.0)
            failure_count = rng.randint(1, 3)
            days_since_failure = rng.expovariate(1.0 / 60.0)
            # A minority of failures have been reviewed and cleared. Those are
            # reinstated and retrieved normally; only their score carries the mark.
            if rng.random() < 0.2:
                approved_at = failure_at + rng.uniform(0.1, 10.0)

        signals.append(
            Signals(
                memory_id=mid,
                eval_score=eval_score,
                eval_failed=eval_failed,
                validations=_poisson(rng, 8.0 * q),
                retrievals=int(1200.0 * (pop ** 3)),
                last_used_days=rng.expovariate(1.0 / (60.0 * (1.0 - pop) + 1.0)),
                negative_feedback=negative,
                source_trust=_clamp01(q + rng.gauss(0.0, 0.28)),
                last_failure_at=failure_at,
                approved_at=approved_at,
                failure_count=failure_count,
                days_since_failure=days_since_failure,
            )
        )

    policy = VetoPolicy(negative_threshold=veto_negative_threshold)
    for s in signals:
        policy.apply(s)

    return QualityCorpus(signals, ids, true_quality, popularity)


def sample_batches(corpus, batch_size, n_batches, seed=99, ids=None):
    """Pre-sample id batches so batch construction is outside the timed region."""
    rng = random.Random(seed)
    pool = ids if ids is not None else corpus.ids
    if batch_size > len(pool):
        raise ValueError("batch_size %d exceeds pool size %d" % (batch_size, len(pool)))
    return [rng.sample(pool, batch_size) for _ in range(n_batches)]
