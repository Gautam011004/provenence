"""How each strategy responds when a memory weakens.

Runs on the memories actually extracted from FIFA-2026.pdf, so the texts and
their provenance are real. Usage history is not: freshly ingested memories all
carry the same prior and therefore all score the same, so there is nothing for a
ranking to discriminate on. `mature_corpus` invents a plausible history for the
other 585 memories to give the subject something to be ranked against. That
history is synthetic and seeded; only the memories are real.

What is being compared is how each strategy answers one question: when a memory
starts attracting complaints, how fast does it stop being surfaced?

  additive             raw weighted sum, uncapped, no veto
  additive_all_capped  same sum with every contribution bounded
  scorecard_equal      normalised, equal weights, retrieval capped at half
  scorecard_tuned      normalised, tuned weights

Run:
    python3 bench/feedback.py
    python3 bench/feedback.py --scenario popular_but_complained
    python3 bench/feedback.py --json results/feedback.json
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memstrength.models import Signals, VETOED  # noqa: E402
from memstrength.strategies import (  # noqa: E402
    RawAdditive,
    ScorecardInline,
    equal_weight_scorecard,
)
from memstrength.veto import VetoPolicy  # noqa: E402

DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fifa_memories.json",
)

SUM_CAPS = {k: 12.5 for k in ("eval", "validation", "retrieval", "recency",
                              "trust", "negative", "history")}
SUM_CAPS["retrieval"] = 6.25


def make_arms(veto_threshold=3):
    return [
        ("additive", RawAdditive()),
        ("additive_all_capped",
         RawAdditive(w_retrieval=0.01, w_validation=0.5,
                     validation_cap=25.0, retrieval_cap=500.0,
                     contribution_caps=SUM_CAPS)),
        ("scorecard_equal",
         equal_weight_scorecard(cls=ScorecardInline,
                                veto_negative_threshold=veto_threshold)),
        ("scorecard_tuned", ScorecardInline(veto_negative_threshold=veto_threshold)),
    ]


# ------------------------------------------------------------------ loading


def load_extracted(path):
    """Read the ingest output back as Signals, keyed by memory id."""
    with open(path) as fh:
        payload = json.load(fh)
    records = []
    for m in payload["memories"]:
        s = m["signals"]
        records.append((
            m["id"],
            m["text"],
            Signals(
                memory_id=m["id"],
                eval_score=s["eval_score"],
                validations=s["validations"],
                retrievals=s["retrievals"],
                negative_feedback=s["negative_feedback"],
                source_trust=s["source_trust"],
                last_used_days=0.0,
            ),
        ))
    return records


def mature_corpus(records, seed=17):
    """Give the corpus a plausible usage history. Synthetic, seeded.

    Without this every record is identical and rank is meaningless. Nothing here
    claims to model real FIFA usage; it exists so the subject memory has a
    distribution to move through.
    """
    rng = random.Random(seed)
    for _id, _text, s in records:
        s.eval_score = min(1.0, max(0.0, rng.gauss(0.78, 0.14)))
        s.validations = int(abs(rng.gauss(6, 5)))
        s.retrievals = int(abs(rng.gauss(120, 160)))
        s.last_used_days = abs(rng.gauss(20, 25))
        s.negative_feedback = 1 if rng.random() < 0.18 else 0
    return records


# ------------------------------------------------------------------ events


def complaint(s, n=1):
    s.negative_feedback += n


def eval_fails(s, day):
    s.eval_failed = True
    s.eval_score = 0.2
    s.last_failure_at = day
    s.days_since_failure = 0.0
    s.failure_count += 1


def validated(s, n=1):
    s.validations += n


def retrieved(s, n=1):
    s.retrievals += n
    s.last_used_days = 0.0


def approved(s, day):
    s.approved_at = day


def time_passes(s, days):
    s.last_used_days += days
    if s.failure_count:
        s.days_since_failure += days


SCENARIOS = {
    "single_complaint": [
        ("baseline", lambda s, d: None),
        ("one complaint", lambda s, d: complaint(s, 1)),
    ],
    "mounting_complaints": [
        ("baseline", lambda s, d: None),
        ("1 complaint", lambda s, d: complaint(s, 1)),
        ("2 complaints", lambda s, d: complaint(s, 1)),
        ("3 complaints", lambda s, d: complaint(s, 1)),
        ("4 complaints", lambda s, d: complaint(s, 1)),
    ],
    "eval_failure": [
        ("baseline", lambda s, d: None),
        ("eval fails", lambda s, d: eval_fails(s, d)),
        ("approved back", lambda s, d: approved(s, d + 1)),
        ("90 days clean", lambda s, d: time_passes(s, 90)),
    ],
    "popular_but_complained": [
        ("baseline", lambda s, d: None),
        ("heavily retrieved", lambda s, d: retrieved(s, 900)),
        ("1 complaint", lambda s, d: complaint(s, 1)),
        ("2 complaints", lambda s, d: complaint(s, 1)),
    ],
    "recovery": [
        ("baseline", lambda s, d: None),
        ("2 complaints", lambda s, d: complaint(s, 2)),
        ("re-validated", lambda s, d: validated(s, 8)),
        ("evals pass", lambda s, d: setattr(s, "eval_score", 0.97)),
    ],
    "neglect": [
        ("baseline", lambda s, d: None),
        ("60 days unused", lambda s, d: time_passes(s, 60)),
        ("180 days unused", lambda s, d: time_passes(s, 120)),
        ("2 years unused", lambda s, d: time_passes(s, 550)),
    ],
}


# ------------------------------------------------------------------ running


def clone(s):
    return Signals(
        memory_id=s.memory_id, eval_score=s.eval_score, eval_failed=s.eval_failed,
        validations=s.validations, retrievals=s.retrievals,
        last_used_days=s.last_used_days, negative_feedback=s.negative_feedback,
        source_trust=s.source_trust, vetoed=s.vetoed,
        last_failure_at=s.last_failure_at, approved_at=s.approved_at,
        failure_count=s.failure_count, days_since_failure=s.days_since_failure,
    )


def rank_of(strategy, subject, others, policy):
    """Where the subject lands among the corpus. 1 is strongest."""
    policy.apply(subject)
    scored = strategy.score_many([subject] + others)
    subject_result = scored[0]
    if subject_result.state == VETOED:
        return None, subject_result
    better = sum(1 for r in scored[1:]
                 if r.state != VETOED and r.value > subject_result.value)
    return better + 1, subject_result


def run_scenario(name, steps, records, veto_threshold=3):
    policy = VetoPolicy(negative_threshold=veto_threshold)
    subject_id, subject_text, subject_base = records[0]
    others = [clone(s) for _id, _t, s in records[1:]]
    for o in others:
        policy.apply(o)

    rows = []
    for arm_name, strategy in make_arms(veto_threshold):
        s = clone(subject_base)
        day = 100.0
        for label, event in steps:
            event(s, day)
            day += 1
            rank, result = rank_of(strategy, s, others, policy)
            rows.append({
                "scenario": name, "arm": arm_name, "step": label,
                "value": result.value, "state": result.state,
                "rank": rank, "of": len(others) + 1,
            })
    return rows, subject_text


def print_scenario(name, rows, subject_text, total):
    print("\n" + "=" * 78)
    print("SCENARIO: %s" % name)
    print("  subject: %s" % subject_text[:70])
    print("=" * 78)

    steps = []
    for r in rows:
        if r["step"] not in steps:
            steps.append(r["step"])

    header = "%-22s" % "" + "".join("%17s" % s[:16] for s in steps)
    print(header)
    print("-" * len(header))
    for arm, _ in make_arms():
        cells = []
        for step in steps:
            r = next(x for x in rows if x["arm"] == arm and x["step"] == step)
            if r["state"] == VETOED:
                cells.append("%17s" % "VETOED")
            elif r["rank"] is None:
                cells.append("%17s" % "-")
            else:
                cells.append("%17s" % ("%.1f  #%d" % (r["value"], r["rank"])))
        print("%-22s%s" % (arm, "".join(cells)))
    print("  cell = strength, then rank among %d memories (#1 strongest)" % total)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--scenario", default=None, help="run one scenario by name")
    ap.add_argument("--subject", type=int, default=0, help="index of the memory to degrade")
    ap.add_argument("--veto-threshold", type=int, default=3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args(argv)

    if not os.path.exists(args.input):
        print("no extracted memories at %s\n"
              "run: .venv/bin/python -m ingest FIFA-2026.pdf --out %s"
              % (args.input, args.input))
        return 1

    records = mature_corpus(load_extracted(args.input), seed=args.seed)
    records = [records[args.subject]] + records[:args.subject] + records[args.subject + 1:]

    print("\nstrength response to negative feedback")
    print("  %d memories extracted from the brand manual (texts real)" % len(records))
    print("  usage history synthetic, seed %d -- see module docstring" % args.seed)

    names = [args.scenario] if args.scenario else list(SCENARIOS)
    all_rows = []
    for name in names:
        if name not in SCENARIOS:
            print("unknown scenario %r; known: %s" % (name, ", ".join(SCENARIOS)))
            return 1
        rows, text = run_scenario(name, SCENARIOS[name], records, args.veto_threshold)
        print_scenario(name, rows, text, len(records))
        all_rows.extend(rows)

    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump({"rows": all_rows, "corpus": len(records), "seed": args.seed},
                      fh, indent=2)
        print("\nwrote %s" % args.json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
