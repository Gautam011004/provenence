"""Stores.

The point of the split: memories live in one store, their strength signals in
another, and the veto/approval ledger in a third. The projection module is the
only thing that joins them, so strength never has to be written back next to
the memory.

Each store can simulate a per-call round trip so the benchmark can model a
networked signal service, not just in-process dict lookups. The delay is a
busy-wait rather than time.sleep because sleep resolution (~1ms on darwin) is
far coarser than the effects being measured.
"""

import time


def _spin(nanos):
    if nanos <= 0:
        return
    end = time.perf_counter_ns() + nanos
    while time.perf_counter_ns() < end:
        pass


class MemoryStore(object):
    """Where the memories themselves live."""

    __slots__ = ("_by_id", "_rtt_ns")

    def __init__(self, memories=(), rtt_us=0.0):
        self._by_id = {}
        for m in memories:
            self._by_id[m.id] = m
        self._rtt_ns = int(rtt_us * 1000)

    def put(self, memory):
        self._by_id[memory.id] = memory

    def pop(self, memory_id):
        """Remove and return a memory, or None if this table does not hold it."""
        return self._by_id.pop(memory_id, None)

    def ids(self):
        return list(self._by_id.keys())

    def fetch_many(self, memory_ids):
        """One batched round trip, as a real retrieval call would be."""
        _spin(self._rtt_ns)
        by_id = self._by_id
        return [by_id[mid] for mid in memory_ids]

    def __contains__(self, memory_id):
        return memory_id in self._by_id

    def __len__(self):
        return len(self._by_id)


class SignalStore(object):
    """The separate strength-signal module. Always hit once per projection."""

    __slots__ = ("_by_id", "_rtt_ns", "calls")

    def __init__(self, signals=(), rtt_us=0.0):
        self._by_id = {}
        for s in signals:
            self._by_id[s.memory_id] = s
        self._rtt_ns = int(rtt_us * 1000)
        self.calls = 0

    def put(self, signals):
        self._by_id[signals.memory_id] = signals

    def fetch_many(self, memory_ids):
        self.calls += 1
        _spin(self._rtt_ns)
        by_id = self._by_id
        return [by_id[mid] for mid in memory_ids]

    def __len__(self):
        return len(self._by_id)


class ApprovalStore(object):
    """Ledger of manual re-approvals that lift a veto.

    Only the scorecard strategy needs this, and only for the subset of memories
    that actually trip a veto condition. That conditional second round trip is a
    real cost of the veto design and the benchmark measures it.
    """

    __slots__ = ("_approved", "_rtt_ns", "calls")

    def __init__(self, approved_ids=(), rtt_us=0.0):
        self._approved = set(approved_ids)
        self._rtt_ns = int(rtt_us * 1000)
        self.calls = 0

    def approve(self, memory_id):
        self._approved.add(memory_id)

    def revoke(self, memory_id):
        self._approved.discard(memory_id)

    def fetch_many(self, memory_ids):
        """Returns the subset of the given ids that carry a live approval."""
        self.calls += 1
        _spin(self._rtt_ns)
        approved = self._approved
        return set([mid for mid in memory_ids if mid in approved])
