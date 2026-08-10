"""Write-time veto and the active/quarantine partition.

The veto used to be a read-time decision, which cost the read path a second
store lookup on nearly every request. Here it moves to the write path instead:
whenever a memory's signals change, VetoPolicy decides its standing and the
partition moves it between two tables.

    active      memories that may be retrieved and scored
    quarantine  vetoed memories, awaiting human approval

Retrieval only ever reads the active table, so a vetoed memory is not fetched,
not scored, and not filtered out later -- it is simply not there. The approval
queue reads the quarantine table directly, which is also the thing that made
the review workflow awkward when veto state was scattered across the corpus.

The tradeoff, stated plainly: read-time veto is always current, whereas this is
only as fresh as the last transition. A memory that fails an eval right now
stays servable until apply_signal_change (or a sweep) moves it. That is the
price of taking the lookup off the request path.
"""


class VetoPolicy(object):
    """Decides standing from a signal record alone.

    An approval only lifts a veto if it is newer than the failure it covers,
    so a memory that is approved and then fails again returns to quarantine.
    """

    __slots__ = ("negative_threshold",)

    def __init__(self, negative_threshold=3):
        self.negative_threshold = negative_threshold

    def triggered(self, s):
        """Does this record trip a veto condition, ignoring approvals?"""
        return s.eval_failed or s.negative_feedback >= self.negative_threshold

    def should_veto(self, s):
        if not (s.eval_failed or s.negative_feedback >= self.negative_threshold):
            return False
        return s.approved_at <= s.last_failure_at

    def apply(self, s):
        """Stamp current standing onto the record. Returns True if it changed."""
        was = s.vetoed
        s.vetoed = self.should_veto(s)
        return s.vetoed != was


class PartitionedMemoryStore(object):
    """Two tables, one veto policy, one invariant.

    Invariant: a memory is in exactly one table, and it is in quarantine if and
    only if its signal record says vetoed.
    """

    __slots__ = ("active", "quarantine", "signals", "policy", "transitions")

    def __init__(self, active_store, quarantine_store, signal_store, policy=None):
        self.active = active_store
        self.quarantine = quarantine_store
        self.signals = signal_store
        self.policy = policy if policy is not None else VetoPolicy()
        self.transitions = 0

    # -- read path -------------------------------------------------------

    def fetch_many(self, memory_ids):
        """Retrieval. Reads the active table only."""
        return self.active.fetch_many(memory_ids)

    def active_ids(self):
        return self.active.ids()

    # -- review queue ----------------------------------------------------

    def pending_approval(self, limit=None):
        """The approval queue: everything currently quarantined.

        This is the payoff of the split. Previously this meant scanning the
        whole corpus for records that trip a veto condition; now it is a read
        of one small table.
        """
        ids = self.quarantine.ids()
        if limit is not None:
            ids = ids[:limit]
        return self.quarantine.fetch_many(ids)

    # -- write path ------------------------------------------------------

    def apply_signal_change(self, memory_id):
        """Re-evaluate one memory's standing and move it if needed."""
        s = self.signals.fetch_many([memory_id])[0]
        self.policy.apply(s)
        return self._place(memory_id, s.vetoed)

    def record_failure(self, memory_id, at_day):
        """A blocking eval failed, or negative feedback crossed the threshold."""
        s = self.signals.fetch_many([memory_id])[0]
        s.last_failure_at = at_day
        s.days_since_failure = 0.0
        s.failure_count += 1
        return self.apply_signal_change(memory_id)

    def approve(self, memory_id, at_day):
        """Human re-approval. Only lifts the veto if newer than the failure.

        The memory is reinstated into the active table and retrieved like any
        other from then on. Its history is not a flag that re-gates it; it
        survives only as a decaying penalty on its strength, so a weak memory
        reads as weak through the score alone.
        """
        s = self.signals.fetch_many([memory_id])[0]
        s.approved_at = at_day
        return self.apply_signal_change(memory_id)

    def _place(self, memory_id, vetoed):
        src, dst = (self.active, self.quarantine) if vetoed else (self.quarantine, self.active)
        memory = src.pop(memory_id)
        if memory is None:
            return False  # already on the correct side
        dst.put(memory)
        self.transitions += 1
        return True

    def resweep(self):
        """Full re-evaluation. The safety net for drift, not the hot path."""
        moved = 0
        for mid in list(self.active.ids()) + list(self.quarantine.ids()):
            if self.apply_signal_change(mid):
                moved += 1
        return moved

    def counts(self):
        return {"active": len(self.active), "quarantine": len(self.quarantine)}
