"""Quality benchmark: does the strength score rank the right memories first?

Runs on the active table only -- under the partitioned design, vetoed memories
are not in it, so ranking quality is a question about everything that survives
the veto. That makes this an honest head-to-head of the arithmetic, with the
veto held constant.

Ground truth comes from the generator: every memory has a latent `true_quality`
no scorer sees, and the signals are noisy, biased views of it. A good scoring
method recovers the latent ranking; a bad one recovers something else, and the
most tempting something-else here is popularity.

Metrics:
  spearman_quality    rank correlation with true quality. Higher is better.
  ndcg@k              gain-weighted ranking quality at the top. Higher better.
  precision@k         share of the top k that are genuinely good. Higher better.
  spearman_popularity rank correlation with popularity. LOWER is better -- a
                      high number means the score is largely a traffic proxy.
  junk@k              count of top-k entries whose true quality is below median.
                      The concrete failure: bad memories ranked highly.
  stability@k         overlap of the top k before and after a small random
                      perturbation of the signals. Higher is better.

Arms:
  additive        approach A with the weights from strategies.RawAdditive
  additive_tuned  approach A with the retrieval weight dialled down, so the
                  comparison is not against a strawman
  scorecard       approach B (ScorecardInline; identical maths to the
                  read-time-veto version, veto read off the record)

Run:
    python3 bench/quality.py
    python3 bench/quality.py --size 50000 --k 100 --seed 7
"""

import argparse
import copy
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memstrength.dataset import build_quality_corpus  # noqa: E402
from memstrength.strategies import (  # noqa: E402
    RawAdditive,
    ScorecardInline,
    equal_weight_scorecard,
)
from memstrength.veto import VetoPolicy  # noqa: E402


# ---------------------------------------------------------------- statistics


def _ranks(values):
    """Average ranks, ties shared."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def pearson(a, b):
    n = len(a)
    if n == 0:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = 0.0
    da = 0.0
    db = 0.0
    for i in range(n):
        x = a[i] - ma
        y = b[i] - mb
        num += x * y
        da += x * x
        db += y * y
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return num / math.sqrt(da * db)


def spearman(a, b):
    return pearson(_ranks(a), _ranks(b))


def top_k_indices(scores, k):
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


def ndcg_at_k(scores, gains, k):
    idx = top_k_indices(scores, k)
    dcg = 0.0
    for rank, i in enumerate(idx):
        dcg += gains[i] / math.log(rank + 2, 2)
    ideal = sorted(gains, reverse=True)[:k]
    idcg = 0.0
    for rank, g in enumerate(ideal):
        idcg += g / math.log(rank + 2, 2)
    return (dcg / idcg) if idcg > 0.0 else 0.0


def precision_at_k(scores, truth, k, threshold):
    idx = top_k_indices(scores, k)
    return sum(1 for i in idx if truth[i] >= threshold) / float(k)


def junk_at_k(scores, truth, k, median):
    idx = top_k_indices(scores, k)
    return sum(1 for i in idx if truth[i] < median)


def jaccard(a, b):
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    return (len(sa & sb) / float(union)) if union else 1.0


# ---------------------------------------------------------------- harness


def perturb(signals, rng, scale=0.05):
    """Small random jitter on every signal, to test ranking stability."""
    out = []
    for s in signals:
        c = copy.copy(s)
        c.eval_score = min(1.0, max(0.0, s.eval_score + rng.gauss(0.0, scale)))
        c.source_trust = min(1.0, max(0.0, s.source_trust + rng.gauss(0.0, scale)))
        c.validations = max(0, s.validations + rng.randint(-1, 1))
        c.retrievals = max(0, int(s.retrievals * (1.0 + rng.gauss(0.0, scale))))
        c.last_used_days = max(0.0, s.last_used_days * (1.0 + rng.gauss(0.0, scale)))
        out.append(c)
    return out


# Same ceilings the scorecard uses, so capping is compared like for like.
VAL_CAP = 25.0
RET_CAP = 500.0
TUNED = dict(w_retrieval=0.01, w_validation=0.5)


def make_arms(veto_threshold):
    """A 2x2 on weights x caps, plus the scorecard.

    The point of the grid is to separate two things that are easy to conflate:
    picking better weights, and bounding the inputs those weights multiply.
    """
    return [
        ("additive", RawAdditive()),
        ("additive_capped", RawAdditive(validation_cap=VAL_CAP, retrieval_cap=RET_CAP)),
        ("additive_tuned", RawAdditive(**TUNED)),
        ("additive_tuned_capped",
         RawAdditive(validation_cap=VAL_CAP, retrieval_cap=RET_CAP, **TUNED)),
        # Every signal capped, including the three that are elapsed days or raw
        # counts. 12.5 is the largest contribution any signal could already
        # reach, so this levels them rather than inventing a new scale.
        ("additive_all_capped",
         RawAdditive(validation_cap=VAL_CAP, retrieval_cap=RET_CAP,
                     contribution_caps=dict(
                         {k: 12.5 for k in
                          ("eval", "validation", "retrieval",
                           "recency", "trust", "negative", "history")},
                         retrieval=6.25),
                     **TUNED)),
        ("scorecard_equal", equal_weight_scorecard(
            cls=ScorecardInline, veto_negative_threshold=veto_threshold)),
        ("scorecard_tuned", ScorecardInline(veto_negative_threshold=veto_threshold)),
    ]


def evaluate(args):
    corpus = build_quality_corpus(
        size=args.size, seed=args.seed, veto_negative_threshold=args.veto_threshold
    )
    policy = VetoPolicy(negative_threshold=args.veto_threshold)
    active = corpus.active_signals(policy)

    truth = [corpus.true_quality[s.memory_id] for s in active]
    pop = [corpus.popularity[s.memory_id] for s in active]
    median = sorted(truth)[len(truth) // 2]

    rng = random.Random(args.seed + 1)
    jittered = perturb(active, rng, args.perturb)

    ks = sorted(set(min(int(x), len(active)) for x in args.k.split(",") if x.strip()))
    rows = []
    for name, strategy in make_arms(args.veto_threshold):
        scores = [r.value for r in strategy.score_many(active)]
        scores_j = [r.value for r in strategy.score_many(jittered)]
        row = {
            "arm": name,
            "spearman_quality": spearman(scores, truth),
            "spearman_popularity": spearman(scores, pop),
            "at_k": {},
        }
        for k in ks:
            top = top_k_indices(scores, k)
            row["at_k"][k] = {
                "ndcg": ndcg_at_k(scores, truth, k),
                "precision": precision_at_k(scores, truth, k, args.good_threshold),
                "junk": junk_at_k(scores, truth, k, median),
                "stability": jaccard(top, top_k_indices(scores_j, k)),
                "mean_quality": sum(truth[i] for i in top) / float(k),
            }
        rows.append(row)

    meta = {
        "size": args.size,
        "active": len(active),
        "quarantined": args.size - len(active),
        "ks": ks,
        "good_threshold": args.good_threshold,
        "median_quality": median,
        "perturb": args.perturb,
        "veto_threshold": args.veto_threshold,
        "seed": args.seed,
    }
    return rows, meta, (corpus, active, truth, pop, ks)


def print_report(rows, meta, detail):
    corpus, active, truth, pop, ks = detail

    print("")
    print("strength ranking quality, active table only")
    print(
        "  corpus=%d active=%d quarantined=%d (%.1f%% vetoed at write time)"
        % (
            meta["size"],
            meta["active"],
            meta["quarantined"],
            100.0 * meta["quarantined"] / meta["size"],
        )
    )
    print(
        "  good means true_quality >= %.2f  median quality %.3f"
        % (meta["good_threshold"], meta["median_quality"])
    )
    print("")

    print("whole-ranking correlation")
    print("%-22s %12s %12s" % ("arm", "vs quality", "vs popularity"))
    print("-" * 42)
    for r in rows:
        print(
            "%-22s %12.3f %12.3f"
            % (r["arm"], r["spearman_quality"], r["spearman_popularity"])
        )
    print("  vs quality: higher is better. vs popularity: LOWER is better -- a high")
    print("  value means the score is mostly measuring traffic, not quality.")

    for k in ks:
        print("")
        header = (
            "%-22s %8s %9s %9s %9s %9s"
            % ("arm @k=%d" % k, "ndcg", "prec", "mean q", "junk", "stability")
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            m = r["at_k"][k]
            print(
                "%-22s %8.3f %9.3f %9.3f %9d %9.3f"
                % (r["arm"], m["ndcg"], m["precision"], m["mean_quality"],
                   m["junk"], m["stability"])
            )
    print("")
    print("  junk: entries in the top k whose true quality is below median.")

    _print_worst_case(rows, active, truth, pop, meta, max(ks))


def _print_worst_case(rows, active, truth, pop, meta, k):
    """Show the single most over-ranked memory under each arm."""
    print("")
    print("worst memory appearing in the top %d, per arm" % k)
    print(
        "%-22s %10s %8s %8s %10s %6s %5s"
        % ("arm", "rank", "quality", "popular", "retrievals", "evals", "neg")
    )
    print("-" * 70)
    for name, strategy in make_arms(meta["veto_threshold"]):
        scores = [r.value for r in strategy.score_many(active)]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        worst_i, worst_rank = None, None
        for rank, i in enumerate(order[:k]):
            if worst_i is None or truth[i] < truth[worst_i]:
                worst_i, worst_rank = i, rank + 1
        if worst_i is None:
            continue
        s = active[worst_i]
        print(
            "%-22s %10d %8.3f %8.3f %10d %6.2f %5d"
            % (
                name,
                worst_rank,
                truth[worst_i],
                pop[worst_i],
                s.retrievals,
                s.eval_score,
                s.negative_feedback,
            )
        )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--size", type=int, default=20000)
    ap.add_argument("--k", default="100,1000,5000",
                    help="comma separated top-k depths for the @k metrics")
    ap.add_argument("--good-threshold", type=float, default=0.70)
    ap.add_argument("--perturb", type=float, default=0.05)
    ap.add_argument("--veto-threshold", type=int, default=3)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args(argv)

    rows, meta, detail = evaluate(args)
    print_report(rows, meta, detail)

    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump({"meta": meta, "rows": rows}, fh, indent=2)
        print("")
        print("wrote %s" % args.json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
