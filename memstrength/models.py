"""Core records.

Strength is deliberately absent from Memory: it is produced by the projection
module at read time and attached to a StrengthResult, never persisted next to
the memory itself.
"""

ACTIVE = "ACTIVE"
VETOED = "VETOED"


class Memory(object):
    """The stored memory. Carries no strength and no strength signals."""

    __slots__ = ("id", "text", "created_at_days")

    def __init__(self, id, text, created_at_days):
        self.id = id
        self.text = text
        self.created_at_days = created_at_days

    def __repr__(self):
        return "Memory(id=%r)" % (self.id,)


class Signals(object):
    """Bounded signals owned by the signal module, keyed by memory id.

    Every field is bounded or monotonically counted so that both strategies see
    the same input domain:

      eval_score       float in [0, 1]   pass rate over the eval suite
      eval_failed      bool              hard failure on a blocking eval
      validations      int  >= 0         human/system confirmations
      retrievals       int  >= 0         retrieval frequency
      last_used_days   float >= 0        recency, days since last retrieval
      negative_feedback int >= 0         explicit negative feedback events
      source_trust     float in [0, 1]   provenance confidence

    Veto standing is carried here too, as just another signal, so the read path
    can decide from one record instead of joining against an approval store:

      vetoed           bool              current standing, set by VetoPolicy
      last_failure_at  float             day of the most recent veto trigger
      approved_at      float             day of the most recent re-approval

    An approval only counts if it is newer than the failure it covers, so one
    approval cannot immunize a memory against every future failure. -1.0 means
    never. The approval ledger remains the system of record; these two fields
    are the read-side projection of it.

    Failure history is graded rather than a gate:

      failure_count      int >= 0        times this memory has been quarantined
      days_since_failure float >= 0      recency of the most recent quarantine

    Once a memory is approved it is reinstated and retrieved like any other --
    there is no lingering flag that re-gates it. What remains of its history is
    a penalty on its strength, decaying as it stays clean, so a previously
    failed memory reads as weak rather than as blocked. The strength score is
    the only thing marking it, which is the point.
    """

    __slots__ = (
        "memory_id",
        "eval_score",
        "eval_failed",
        "validations",
        "retrievals",
        "last_used_days",
        "negative_feedback",
        "source_trust",
        "vetoed",
        "last_failure_at",
        "approved_at",
        "failure_count",
        "days_since_failure",
    )

    def __init__(
        self,
        memory_id,
        eval_score=0.0,
        eval_failed=False,
        validations=0,
        retrievals=0,
        last_used_days=0.0,
        negative_feedback=0,
        source_trust=0.5,
        vetoed=False,
        last_failure_at=-1.0,
        approved_at=-1.0,
        failure_count=0,
        days_since_failure=1e9,
    ):
        self.memory_id = memory_id
        self.eval_score = eval_score
        self.eval_failed = eval_failed
        self.validations = validations
        self.retrievals = retrievals
        self.last_used_days = last_used_days
        self.negative_feedback = negative_feedback
        self.source_trust = source_trust
        self.vetoed = vetoed
        self.last_failure_at = last_failure_at
        self.approved_at = approved_at
        self.failure_count = failure_count
        self.days_since_failure = days_since_failure


class StrengthResult(object):
    """What the projection returns alongside a memory."""

    __slots__ = ("memory_id", "value", "state", "components")

    def __init__(self, memory_id, value, state=ACTIVE, components=None):
        self.memory_id = memory_id
        self.value = value
        self.state = state
        self.components = components

    def __repr__(self):
        return "StrengthResult(%r, %.4f, %s)" % (self.memory_id, self.value, self.state)
