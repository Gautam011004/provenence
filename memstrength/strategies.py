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

# Retrieval frequency is capped at half whatever the other signals get, unless
# an explicit cap says otherwise.
#
# This is about proportion, not distrust. Retrieval carries real information --
# a memory that keeps getting pulled up is probably worth having. But it is the
# signal most able to swamp the others, for two reasons: it is an unbounded
# count, and it is the only one that feeds back on itself, since a high score
# surfaces a memory more often, which raises its retrieval count, which raises
# its score. An eval result does not improve because a memory was shown. So its
# ceiling is set lower to keep it in proportion, while its weight stays at
# parity and remains the thing to calibrate.
#
# Note that bench/quality.py cannot see the feedback effect at all: it scores a
# single snapshot with no loop running, so the popularity coupling it reports is
# a floor.
RETRIEVAL_CAP_RATIO = 0.5

# The ceiling frees up headroom that would otherwise shrink the reachable range.
# It goes to the two signals that speak most directly to whether a memory is
# correct, keeping the scale a true 0-100.
HEADROOM_RECIPIENTS = ("eval", "validation")


def _redistribute_headroom(weights, retrieval_cap_ratio=RETRIEVAL_CAP_RATIO,
                           recipients=HEADROOM_RECIPIENTS):
    """Reassign the budget retrieval gives up, so the budgets still total 1.0.

    Takes and returns a {signal: weight} mapping. Retrieval's weight is left
    alone -- only its ceiling is lower -- so what is redistributed is the gap
    between the two.
    """
    freed = weights["retrieval"] * (1.0 - retrieval_cap_ratio)
    share = freed / len(recipients)
    out = dict(weights)
    for name in recipients:
        out[name] += share
    return out


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
        "validation_cap",
        "retrieval_cap",
        "cap_eval",
        "cap_validation",
        "cap_retrieval",
        "cap_recency",
        "cap_trust",
        "cap_negative",
        "cap_history",
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
        validation_cap=float("inf"),
        retrieval_cap=float("inf"),
        contribution_caps=None,
    ):
        self.w_eval = w_eval
        self.w_validation = w_validation
        self.w_retrieval = w_retrieval
        self.w_recency = w_recency
        self.w_trust = w_trust
        self.w_negative = w_negative
        self.w_failure = w_failure
        # Two different ceilings, and the difference matters.
        #
        # validation_cap / retrieval_cap bound the *input*: clamp the count
        # before multiplying. That bounds the signal but does not make it
        # proportionate -- retrievals capped at 500 with weight 1.0 still
        # contributes 500 against eval's 10.
        #
        # contribution_caps bound the *contribution*: clamp |weight * value|
        # after multiplying, so the ceiling holds whatever the weight is. Every
        # signal gets one, including recency, negative feedback and failure
        # history, which are elapsed days and counts and so grow without limit
        # otherwise. Infinite by default, so the uncapped arm is unchanged.
        self.validation_cap = validation_cap
        self.retrieval_cap = retrieval_cap
        inf = float("inf")
        caps = contribution_caps or {}
        self.cap_eval = caps.get("eval", inf)
        self.cap_validation = caps.get("validation", inf)
        self.cap_retrieval = caps.get("retrieval", inf)
        self.cap_recency = caps.get("recency", inf)
        self.cap_trust = caps.get("trust", inf)
        self.cap_negative = caps.get("negative", inf)
        self.cap_history = caps.get("history", inf)

    def influence_budget(self):
        """The most each signal can contribute to a score.

        `inf` means that signal can grow without limit and drown every other
        one. Capping an input bounds it; leaving the input uncapped does not,
        no matter how small the weight.
        """
        inf = float("inf")
        return {
            "eval": min(abs(self.w_eval), self.cap_eval),
            "validation": min(abs(self.w_validation) * self.validation_cap,
                              self.cap_validation),
            "retrieval": min(abs(self.w_retrieval) * self.retrieval_cap,
                             self.cap_retrieval),
            # Elapsed days and raw counts have no ceiling of their own, so these
            # are bounded only by an explicit contribution cap.
            "recency": self.cap_recency,
            "trust": min(abs(self.w_trust), self.cap_trust),
            "negative": self.cap_negative,
            "history": self.cap_history,
        }

    def score_many(self, signals, approved_ids=_NO_APPROVALS, explain=False):
        w_eval = self.w_eval
        w_val = self.w_validation
        w_ret = self.w_retrieval
        w_rec = self.w_recency
        w_tru = self.w_trust
        w_neg = self.w_negative
        w_fail = self.w_failure
        val_cap = self.validation_cap
        ret_cap = self.retrieval_cap
        c_eval_cap = self.cap_eval
        c_val_cap = self.cap_validation
        c_ret_cap = self.cap_retrieval
        c_rec_cap = self.cap_recency
        c_tru_cap = self.cap_trust
        c_neg_cap = self.cap_negative
        c_his_cap = self.cap_history

        out = []
        append = out.append
        for s in signals:
            v = s.validations
            if v > val_cap:
                v = val_cap
            r = s.retrievals
            if r > ret_cap:
                r = ret_cap

            k_eval = w_eval * s.eval_score
            if k_eval > c_eval_cap:
                k_eval = c_eval_cap
            elif k_eval < -c_eval_cap:
                k_eval = -c_eval_cap
            k_val = w_val * v
            if k_val > c_val_cap:
                k_val = c_val_cap
            elif k_val < -c_val_cap:
                k_val = -c_val_cap
            k_ret = w_ret * r
            if k_ret > c_ret_cap:
                k_ret = c_ret_cap
            elif k_ret < -c_ret_cap:
                k_ret = -c_ret_cap
            k_rec = w_rec * s.last_used_days
            if k_rec > c_rec_cap:
                k_rec = c_rec_cap
            elif k_rec < -c_rec_cap:
                k_rec = -c_rec_cap
            k_tru = w_tru * s.source_trust
            if k_tru > c_tru_cap:
                k_tru = c_tru_cap
            elif k_tru < -c_tru_cap:
                k_tru = -c_tru_cap
            k_neg = w_neg * s.negative_feedback
            if k_neg > c_neg_cap:
                k_neg = c_neg_cap
            elif k_neg < -c_neg_cap:
                k_neg = -c_neg_cap
            k_his = w_fail * s.failure_count
            if k_his > c_his_cap:
                k_his = c_his_cap
            elif k_his < -c_his_cap:
                k_his = -c_his_cap

            value = k_eval + k_val + k_ret + k_rec + k_tru + k_neg + k_his

            if explain:
                components = {
                    "eval": k_eval,
                    "validation": k_val,
                    "retrieval": k_ret,
                    "recency": k_rec,
                    "trust": k_tru,
                    "negative": k_neg,
                    "history": k_his,
                }
                append(StrengthResult(s.memory_id, value, ACTIVE, components))
            else:
                append(StrengthResult(s.memory_id, value, ACTIVE))
        return out


class NormalizedScorecard(Strategy):
    """Approach B: every signal normalized to [0, 1], capped, weighted to 0-100.

    Caps are applied in two places: count-style inputs are clamped before
    normalization so one very hot memory cannot saturate its component, and each
    contribution is clamped after weighting so no signal can exceed its ceiling
    however the weights are later retuned.

    Penalties carry the same ceiling as the positive signals -- one fifth of the
    scale by default, matching an equal share. Negative feedback and failure
    history therefore weigh exactly as much against a memory as any single
    positive signal weighs for it, and neither can annihilate a score on its own.

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
        "cap_eval",
        "cap_validation",
        "cap_retrieval",
        "cap_recency",
        "cap_trust",
    )

    def __init__(
        self,
        # eval and validation carry the headroom retrieval gives up by being
        # capped at half its weight: 0.35 + 0.0375 and 0.20 + 0.0375. Budgets
        # therefore still total 1.0 and max_score() is a true 100.
        w_eval=0.3875,
        w_validation=0.2375,
        w_retrieval=0.15,
        w_recency=0.20,
        w_trust=0.10,
        validation_k=3.0,
        retrieval_k=10.0,
        validation_cap=25.0,
        retrieval_cap=500.0,
        recency_half_life_days=30.0,
        negative_step=0.15,
        negative_cap=0.20,
        history_step=0.12,
        history_cap=0.20,
        history_half_life_days=45.0,
        veto_negative_threshold=3,
        scale=100.0,
        contribution_caps=None,
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

        # Hard ceiling on what each signal may contribute, as a fraction of the
        # scale, applied *after* weighting. Defaults to the weight itself, which
        # is already the natural maximum. Setting one lower pins that signal's
        # influence no matter how the weights are later retuned -- a weight is a
        # preference, a cap is a guarantee.
        caps = contribution_caps or {}
        self.cap_eval = caps.get("eval", w_eval)
        self.cap_validation = caps.get("validation", w_validation)
        self.cap_retrieval = caps.get(
            "retrieval", w_retrieval * RETRIEVAL_CAP_RATIO
        )
        self.cap_recency = caps.get("recency", w_recency)
        self.cap_trust = caps.get("trust", w_trust)

    def influence_budget(self):
        """The most each signal can contribute, on the same 0-100 scale.

        Every entry is finite by construction: each component is normalised to
        0..1 before weighting, so a signal's weight *is* its ceiling. The
        positive weights sum to the full scale, which makes the budget readable
        straight off the config.
        """
        s = self.scale
        return {
            "eval": min(self.w_eval, self.cap_eval) * s,
            "validation": min(self.w_validation, self.cap_validation) * s,
            "retrieval": min(self.w_retrieval, self.cap_retrieval) * s,
            "recency": min(self.w_recency, self.cap_recency) * s,
            "trust": min(self.w_trust, self.cap_trust) * s,
            "negative": self.negative_cap * s,
            "history": self.history_cap * s,
        }

    def max_score(self):
        """The highest score actually reachable, given the ceilings.

        Capping a signal below its weight lowers this: with retrieval held to
        half its share the top of the range is 92.5, not 100. Ranking is
        unaffected, but anything that reads the number as a percentage should
        divide by this rather than assume 100.
        """
        b = self.influence_budget()
        return sum(b[k] for k in
                   ("eval", "validation", "retrieval", "recency", "trust"))

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
        cap_eval = self.cap_eval
        cap_val = self.cap_validation
        cap_ret = self.cap_retrieval
        cap_rec = self.cap_recency
        cap_tru = self.cap_trust
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

            k_eval = w_eval * c_eval
            if k_eval > cap_eval:
                k_eval = cap_eval
            k_val = w_val * c_val
            if k_val > cap_val:
                k_val = cap_val
            k_ret = w_ret * c_ret
            if k_ret > cap_ret:
                k_ret = cap_ret
            k_rec = w_rec * c_rec
            if k_rec > cap_rec:
                k_rec = cap_rec
            k_tru = w_tru * c_tru
            if k_tru > cap_tru:
                k_tru = cap_tru

            score = (k_eval + k_val + k_ret + k_rec + k_tru) - penalty
            if score < 0.0:
                score = 0.0
            elif score > 1.0:
                score = 1.0

            if explain:
                components = {
                    "eval": k_eval,
                    "validation": k_val,
                    "retrieval": k_ret,
                    "recency": k_rec,
                    "trust": k_tru,
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
        cap_eval = self.cap_eval
        cap_val = self.cap_validation
        cap_ret = self.cap_retrieval
        cap_rec = self.cap_recency
        cap_tru = self.cap_trust
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

            k_eval = w_eval * c_eval
            if k_eval > cap_eval:
                k_eval = cap_eval
            k_val = w_val * c_val
            if k_val > cap_val:
                k_val = cap_val
            k_ret = w_ret * c_ret
            if k_ret > cap_ret:
                k_ret = cap_ret
            k_rec = w_rec * c_rec
            if k_rec > cap_rec:
                k_rec = cap_rec
            k_tru = w_tru * c_tru
            if k_tru > cap_tru:
                k_tru = cap_tru

            score = (k_eval + k_val + k_ret + k_rec + k_tru) - penalty
            if score < 0.0:
                score = 0.0
            elif score > 1.0:
                score = 1.0

            if explain:
                components = {
                    "eval": k_eval,
                    "validation": k_val,
                    "retrieval": k_ret,
                    "recency": k_rec,
                    "trust": k_tru,
                    "penalty": -penalty,
                }
                append(StrengthResult(s.memory_id, score * scale, ACTIVE, components))
            else:
                append(StrengthResult(s.memory_id, score * scale, ACTIVE))
        return out


SIGNALS = ("eval", "validation", "retrieval", "recency", "trust")


def equal_weight_scorecard(cap=None, retrieval_cap=None, cls=None, **kw):
    """Every signal weighted the same, each with a contribution ceiling.

    A neutral starting point for tuning: with five signals each gets 1/5 of the
    scale, so nothing is privileged before there is evidence to privilege it.
    `cap` sets each signal's ceiling as a fraction of the scale, defaulting to
    the equal weight itself. Lower it to pin a signal below its fair share.

    Retrieval is the exception, capped at RETRIEVAL_CAP_RATIO of that ceiling so
    it cannot swamp the rest; pass `retrieval_cap` to override. Its *weight*
    stays at parity -- the cap is a safety ceiling, the weight is the
    calibration, and the weight is what you should tune against real labels. The
    headroom the ceiling frees goes to eval and validation, so the reachable
    maximum stays 100.

    Retune the weights later via keyword arguments; the caps stay in force
    regardless, so a mistuned weight cannot let one signal take over.
    """
    w = 1.0 / len(SIGNALS)
    ceiling = w if cap is None else cap
    ret_ceiling = (ceiling * RETRIEVAL_CAP_RATIO
                   if retrieval_cap is None else retrieval_cap)
    base = _redistribute_headroom({name: w for name in SIGNALS},
                                  retrieval_cap_ratio=ret_ceiling / w)
    weights = {"w_" + name: base[name] for name in SIGNALS}
    # Penalties share the ceiling, so nothing counts against a memory harder
    # than any single signal counts for it.
    weights.setdefault("negative_cap", ceiling)
    weights.setdefault("history_cap", ceiling)
    weights.update(kw)
    factory = cls if cls is not None else NormalizedScorecard
    # With no explicit `cap`, each signal is held to its own weight and the
    # redistribution keeps the reachable maximum at 100. An explicit `cap`
    # constrains every signal uniformly, which deliberately lowers that maximum
    # -- check max_score() if the number is shown to anyone.
    caps = {name: (base[name] if cap is None else cap) for name in SIGNALS}
    caps["retrieval"] = ret_ceiling
    return factory(contribution_caps=caps, **weights)
