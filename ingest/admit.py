"""Candidates -> Memory + Signals records.

Two jobs: deduplicate, and decide what a memory's signals look like on day one.

The cold start is the interesting part. Every signal the strength model uses is
evidence accumulated from use: evals that have run, validations people have
given, times it was retrieved, complaints received. A memory extracted five
seconds ago has none of that. It is not weak, it is unmeasured, and those are
different things the score cannot currently tell apart.

So ingest writes an explicit prior and marks it. `eval_score` is a placeholder
until the eval suite actually runs, not a measurement, and `provisional` says
so. Anything reading strength on freshly ingested memories is reading the prior
back, and should treat the ranking among them as arbitrary until real signals
land.
"""

import hashlib
import re

from memstrength.models import Memory, Signals

# A brand manual published by the rights holder is authoritative about its own
# rules, so provenance confidence starts high. This is about the source, not
# about whether any particular extraction is correct.
DEFAULT_SOURCE_TRUST = 0.9

# Neither good nor bad: no eval has run. See the module docstring.
DEFAULT_EVAL_PRIOR = 0.5


class Admitted(object):
    __slots__ = ("memory", "signals", "candidate", "provisional")

    def __init__(self, memory, signals, candidate, provisional=True):
        self.memory = memory
        self.signals = signals
        self.candidate = candidate
        self.provisional = provisional

    def as_dict(self):
        c = self.candidate
        return {
            "id": self.memory.id,
            "text": self.memory.text,
            "kind": c.kind,
            "section": c.section,
            "page": c.page,
            "heading": c.heading,
            "provisional": self.provisional,
            "signals": {
                "eval_score": self.signals.eval_score,
                "source_trust": self.signals.source_trust,
                "validations": self.signals.validations,
                "retrievals": self.signals.retrievals,
                "negative_feedback": self.signals.negative_feedback,
            },
        }


def memory_id(source, candidate):
    """Stable across runs: same document and same text gives the same id.

    Keyed on the text rather than the page so that re-paginating the source in a
    later edition does not orphan every memory and re-admit it as new.
    """
    norm = re.sub(r"\s+", " ", candidate.text.strip().lower())
    digest = hashlib.sha1(("%s\x00%s" % (source, norm)).encode("utf-8")).hexdigest()
    return "%s-%s" % (_slug(source), digest[:12])


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24]


def dedupe(candidates):
    """Drop repeats, keeping the first occurrence.

    The same instruction is restated across sections in a manual like this
    ('Don't stretch or otherwise distort ...' appears under several logos).
    Keeping one copy per distinct wording means retrieval frequency accrues to
    one record rather than being split across near-identical ones.
    """
    seen = {}
    for c in candidates:
        key = re.sub(r"\s+", " ", c.text.strip().lower())
        if key in seen:
            continue
        seen[key] = c
        yield c


def admit(
    candidates,
    source,
    source_trust=DEFAULT_SOURCE_TRUST,
    eval_prior=DEFAULT_EVAL_PRIOR,
):
    """Turn candidates into (Memory, Signals) pairs with day-one signals."""
    for c in dedupe(candidates):
        mid = memory_id(source, c)
        memory = Memory(mid, c.text, created_at_days=0.0)
        # Extraction confidence discounts provenance confidence. A colour whose
        # values are certain but whose name could not be attributed is a less
        # trustworthy record than a cleanly parsed sentence, and the score
        # should reflect that rather than treating both as equally solid.
        trust = source_trust * getattr(c, "confidence", 1.0)
        signals = Signals(
            memory_id=mid,
            eval_score=eval_prior,
            eval_failed=False,
            validations=0,
            retrievals=0,
            last_used_days=0.0,
            negative_feedback=0,
            source_trust=trust,
            vetoed=False,
            failure_count=0,
        )
        yield Admitted(memory, signals, c)


def load_stores(admitted, memory_store, signal_store):
    """Write admitted records into the live stores. Returns the count."""
    n = 0
    for a in admitted:
        memory_store.put(a.memory)
        signal_store.put(a.signals)
        n += 1
    return n
