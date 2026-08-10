"""The two competing strength calculations.

Both implement the same two-phase interface so the projection module stays
generic:

    veto_candidates(signals)          -> ids needing an approval lookup
    score_many(signals, approved_ids) -> list of StrengthResult

Approach A (RawAdditive) never nominates veto candidates, so the projection
skips the approval round trip entirely. Approach B (NormalizedScorecard) does,
which is exactly the cost difference the benchmark is there to measure.
"""

import math

from .models import StrengthResult, ACTIVE, VETOED

_NO_APPROVALS = frozenset()


class Strategy(object):
    __slots__ = ()
    name = "base"

    def veto_candidates(self, signals):
        return ()

    def score_many(self, signals, approved_ids=_NO_APPROVALS, explain=False):
        raise NotImplementedError


class NoOp(Strategy):
    """Control arm: isolates the projection module's own overhead."""

    __slots__ = ()
    name = "noop"

    def score_many(self, signals, approved_ids=_NO_APPROVALS, explain=False):
        return [StrengthResult(s.memory_id, 0.0, ACTIVE) for s in signals]


class RawAdditive(Strategy):
    """Approach A: weighted sum over raw signal values.

    Unbounded in both directions and uncapped by design; a single very hot or
    very old memory can dominate the range. No veto path: negative feedback is
    just another negative term.
    """

    name = "additive"

    __slots__ = (
        "w_eval",
        "w_validation",
        "w_retrieval",
        "w_recency",
        "w_trust",
        "w_negative",
        "w_failure",
    )

    def __init__(
        self,
        w_eval=10.0,
        w_validation=2.0,
        w_retrieval=1.0,
        w_recency=-0.1,
        w_trust=3.0,
        w_negative=-5.0,
        w_failure=-6.0,
    ):
        self.w_eval = w_eval
        self.w_validation = w_validation
        self.w_retrieval = w_retrieval
        self.w_recency = w_recency
        self.w_trust = w_trust
        self.w_negative = w_negative
        self.w_failure = w_failure

    def score_many(self, signals, approved_ids=_NO_APPROVALS, explain=False):
        w_eval = self.w_eval
        w_val = self.w_validation
        w_ret = self.w_retrieval
        w_rec = self.w_recency
        w_tru = self.w_trust
        w_neg = self.w_negative
        w_fail = self.w_failure

        out = []
        append = out.append
        for s in signals:
            value = (
                w_eval * s.eval_score
                + w_val * s.validations
                + w_ret * s.retrievals
                + w_rec * s.last_used_days
                + w_tru * s.source_trust
                + w_neg * s.negative_feedback
                + w_fail * s.failure_count
            )

            if explain:
                components = {
                    "eval": w_eval * s.eval_score,
                    "validation": w_val * s.validations,
                    "retrieval": w_ret * s.retrievals,
                    "recency": w_rec * s.last_used_days,
                    "trust": w_tru * s.source_trust,
                    "negative": w_neg * s.negative_feedback,
                    "history": w_fail * s.failure_count,
                }
                append(StrengthResult(s.memory_id, value, ACTIVE, components))
            else:
                append(StrengthResult(s.memory_id, value, ACTIVE))
        return out


class NormalizedScorecard(Strategy):
    """Approach B: every signal normalized to [0, 1], capped, weighted to 0-100.

    Caps are applied in two places: count-style inputs are clamped before
    normalization so one very hot memory cannot saturate its component, and the
    negative-feedback penalty is clamped so it degrades rather than annihilates
    a score.

    Veto: a hard eval failure or negative feedback at or above the threshold
    forces the memory to 0/VETOED until an entry appears in the approval ledger.
    The veto is evaluated first and short-circuits the arithmetic.
    """

    name = "scorecard"

    __slots__ = (
        "w_eval",
        "w_validation",
        "w_retrieval",
        "w_recency",
        "w_trust",
        "validation_k",
        "retrieval_k",
        "validation_cap",
        "retrieval_cap",
        "recency_lambda",
        "negative_step",
        "negative_cap",
        "history_step",
        "history_cap",
        "history_lambda",
        "veto_negative_threshold",
        "scale",
    )

    def __init__(
        self,
        w_eval=0.35,
        w_validation=0.20,
        w_retrieval=0.15,
        w_recency=0.20,
        w_trust=0.10,
        validation_k=3.0,
        retrieval_k=10.0,
        validation_cap=25.0,
        retrieval_cap=500.0,
        recency_half_life_days=30.0,
        negative_step=0.15,
        negative_cap=0.45,
        history_step=0.12,
        history_cap=0.36,
        history_half_life_days=45.0,
        veto_negative_threshold=3,
        scale=100.0,
    ):
        self.w_eval = w_eval
        self.w_validation = w_validation
        self.w_retrieval = w_retrieval
        self.w_recency = w_recency
        self.w_trust = w_trust
        self.validation_k = validation_k
        self.retrieval_k = retrieval_k
        self.validation_cap = validation_cap
        self.retrieval_cap = retrieval_cap
        self.recency_lambda = math.log(2.0) / recency_half_life_days
        self.negative_step = negative_step
        self.negative_cap = negative_cap
        self.history_step = history_step
        self.history_cap = history_cap
        self.history_lambda = math.log(2.0) / history_half_life_days
        self.veto_negative_threshold = veto_negative_threshold
        self.scale = scale

    def is_veto_candidate(self, s):
        return s.eval_failed or s.negative_feedback >= self.veto_negative_threshold

    def veto_candidates(self, signals):
        threshold = self.veto_negative_threshold
        return [
            s.memory_id
            for s in signals
            if s.eval_failed or s.negative_feedback >= threshold
        ]

    def score_many(self, signals, approved_ids=_NO_APPROVALS, explain=False):
        w_eval = self.w_eval
        w_val = self.w_validation
        w_ret = self.w_retrieval
        w_rec = self.w_recency
        w_tru = self.w_trust
        val_k = self.validation_k
        ret_k = self.retrieval_k
        val_cap = self.validation_cap
        ret_cap = self.retrieval_cap
        lam = self.recency_lambda
        neg_step = self.negative_step
        neg_cap = self.negative_cap
        hist_step = self.history_step
        hist_cap = self.history_cap
        hist_lam = self.history_lambda
        threshold = self.veto_negative_threshold
        scale = self.scale
        exp = math.exp

        out = []
        append = out.append
        for s in signals:
            # Veto first: short-circuits all of the arithmetic below.
            if s.eval_failed or s.negative_feedback >= threshold:
                if s.memory_id not in approved_ids:
                    append(StrengthResult(s.memory_id, 0.0, VETOED))
                    continue

            v = s.validations
            if v > val_cap:
                v = val_cap
            r = s.retrievals
            if r > ret_cap:
                r = ret_cap

            c_eval = s.eval_score
            c_val = v / (v + val_k)
            c_ret = r / (r + ret_k)
            c_rec = exp(-lam * s.last_used_days)
            c_tru = s.source_trust

            penalty = neg_step * s.negative_feedback
            if penalty > neg_cap:
                penalty = neg_cap

            # A reinstated memory is retrieved like any other; what is left of
            # its history is this penalty, which fades as it stays clean.
            if s.failure_count:
                hist = hist_step * s.failure_count
                if hist > hist_cap:
                    hist = hist_cap
                penalty += hist * exp(-hist_lam * s.days_since_failure)

            score = (
                w_eval * c_eval
                + w_val * c_val
                + w_ret * c_ret
                + w_rec * c_rec
                + w_tru * c_tru
            ) - penalty
            if score < 0.0:
                score = 0.0
            elif score > 1.0:
                score = 1.0

            if explain:
                components = {
                    "eval": w_eval * c_eval,
                    "validation": w_val * c_val,
                    "retrieval": w_ret * c_ret,
                    "recency": w_rec * c_rec,
                    "trust": w_tru * c_tru,
                    "penalty": -penalty,
                }
                append(
                    StrengthResult(s.memory_id, score * scale, ACTIVE, components)
                )
            else:
                append(StrengthResult(s.memory_id, score * scale, ACTIVE))
        return out


class ScorecardInline(NormalizedScorecard):
    """Approach B, with veto standing read off the signal record.

    Identical arithmetic to NormalizedScorecard. The only difference is where
    the veto decision comes from: `s.vetoed`, stamped at write time by
    VetoPolicy, instead of a condition evaluated here against an approval set
    fetched from a second store. So it nominates no veto candidates and the
    projection makes no approval round trip.

    Under a PartitionedMemoryStore the `s.vetoed` branch should never fire,
    because vetoed memories are not in the active table to begin with. It is
    kept as a defence in depth: if the partition ever drifts, this fails closed
    rather than serving a vetoed memory. One boolean test per record.

    The scoring loop is duplicated from the parent rather than factored into a
    shared helper on purpose -- a per-record function call would show up in the
    benchmark and would penalise both scorecard arms against the additive one.
    """

    __slots__ = ()
    name = "scorecard_inline"

    def veto_candidates(self, signals):
        return ()

    def score_many(self, signals, approved_ids=_NO_APPROVALS, explain=False):
        w_eval = self.w_eval
        w_val = self.w_validation
        w_ret = self.w_retrieval
        w_rec = self.w_recency
        w_tru = self.w_trust
        val_k = self.validation_k
        ret_k = self.retrieval_k
        val_cap = self.validation_cap
        ret_cap = self.retrieval_cap
        lam = self.recency_lambda
        neg_step = self.negative_step
        neg_cap = self.negative_cap
        hist_step = self.history_step
        hist_cap = self.history_cap
        hist_lam = self.history_lambda
        scale = self.scale
        exp = math.exp

        out = []
        append = out.append
        for s in signals:
            if s.vetoed:
                append(StrengthResult(s.memory_id, 0.0, VETOED))
                continue

            v = s.validations
            if v > val_cap:
                v = val_cap
            r = s.retrievals
            if r > ret_cap:
                r = ret_cap

            c_eval = s.eval_score
            c_val = v / (v + val_k)
            c_ret = r / (r + ret_k)
            c_rec = exp(-lam * s.last_used_days)
            c_tru = s.source_trust

            penalty = neg_step * s.negative_feedback
            if penalty > neg_cap:
                penalty = neg_cap

            # A reinstated memory is retrieved like any other; what is left of
            # its history is this penalty, which fades as it stays clean.
            if s.failure_count:
                hist = hist_step * s.failure_count
                if hist > hist_cap:
                    hist = hist_cap
                penalty += hist * exp(-hist_lam * s.days_since_failure)

            score = (
                w_eval * c_eval
                + w_val * c_val
                + w_ret * c_ret
                + w_rec * c_rec
                + w_tru * c_tru
            ) - penalty
            if score < 0.0:
                score = 0.0
            elif score > 1.0:
                score = 1.0

            if explain:
                components = {
                    "eval": w_eval * c_eval,
                    "validation": w_val * c_val,
                    "retrieval": w_ret * c_ret,
                    "recency": w_rec * c_rec,
                    "trust": w_tru * c_tru,
                    "penalty": -penalty,
                }
                append(StrengthResult(s.memory_id, score * scale, ACTIVE, components))
            else:
                append(StrengthResult(s.memory_id, score * scale, ACTIVE))
        return out
