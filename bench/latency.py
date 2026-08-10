"""Latency benchmark: raw additive sum vs normalized scorecard with veto,
each measured through the strength projection module.

Three arms per configuration:

  noop       control. Retrieve + join with a do-nothing scorer, so the
             projection module's own overhead can be subtracted out.
  additive   approach A. Weighted sum over raw signals, no caps, no veto.
  scorecard  approach B. Normalized + capped components, plus a veto that
             short-circuits scoring and costs a conditional approval lookup.

Two modes:

  e2e       full read path: memory fetch -> signal fetch -> approval fetch
            -> score. This is what a user-facing request actually pays.
  compute   strategy arithmetic over pre-fetched signals. Isolates the part
            that genuinely differs between the two approaches.

Run:
    python3 bench/latency.py
    python3 bench/latency.py --batch-sizes 1,10,100,1000 --veto-rates 0,0.15,0.4
    python3 bench/latency.py --signal-rtt-us 250 --approval-rtt-us 250
    python3 bench/latency.py --json results.json
"""

import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memstrength.dataset import build_corpus, sample_batches  # noqa: E402
from memstrength.projection import StrengthProjection  # noqa: E402
from memstrength.strategies import (  # noqa: E402
    NoOp,
    RawAdditive,
    NormalizedScorecard,
    ScorecardInline,
)

ARMS = ("noop", "additive", "scorecard", "scorecard_inline")


def make_strategy(arm, veto_negative_threshold):
    if arm == "noop":
        return NoOp()
    if arm == "additive":
        return RawAdditive()
    if arm == "scorecard":
        return NormalizedScorecard(veto_negative_threshold=veto_negative_threshold)
    if arm == "scorecard_inline":
        return ScorecardInline(veto_negative_threshold=veto_negative_threshold)
    raise ValueError("unknown arm %r" % (arm,))


def percentile(sorted_samples, q):
    """Nearest-rank percentile. q in [0, 100]."""
    if not sorted_samples:
        return 0.0
    k = int(round(q / 100.0 * len(sorted_samples) + 0.5)) - 1
    if k < 0:
        k = 0
    elif k >= len(sorted_samples):
        k = len(sorted_samples) - 1
    return float(sorted_samples[k])


def summarize(samples_ns, batch_size):
    s = sorted(samples_ns)
    total = float(sum(s))
    n = len(s)
    mean = total / n
    return {
        "n": n,
        "mean_us": mean / 1000.0,
        "p50_us": percentile(s, 50) / 1000.0,
        "p95_us": percentile(s, 95) / 1000.0,
        "p99_us": percentile(s, 99) / 1000.0,
        "min_us": s[0] / 1000.0,
        "max_us": s[-1] / 1000.0,
        "per_memory_us": (mean / 1000.0) / batch_size,
        "throughput_per_s": (batch_size * 1e9 / mean) if mean > 0 else 0.0,
    }


def time_calls(fn, payloads, reps, warmup):
    """Time `fn(payload)` over cycling payloads. Returns list of ns per call."""
    n_payloads = len(payloads)
    for i in range(warmup):
        fn(payloads[i % n_payloads])

    samples = []
    append = samples.append
    clock = time.perf_counter_ns
    for i in range(reps):
        payload = payloads[i % n_payloads]
        t0 = clock()
        fn(payload)
        t1 = clock()
        append(t1 - t0)
    return samples


def reps_for(batch_size, budget):
    """Keep total scored memories per configuration roughly constant."""
    r = budget // max(1, batch_size)
    if r < 40:
        r = 40
    elif r > 3000:
        r = 3000
    return int(r)


def attribute_phases(projection, batches, reps):
    """Average per-phase cost over a small instrumented pass."""
    n = len(batches)
    acc = [0, 0, 0, 0]
    for i in range(reps):
        p = projection.project_timed(batches[i % n])
        acc[0] += p.t_retrieve_ns
        acc[1] += p.t_signals_ns
        acc[2] += p.t_approvals_ns
        acc[3] += p.t_score_ns
    return {
        "retrieve_us": acc[0] / reps / 1000.0,
        "signals_us": acc[1] / reps / 1000.0,
        "approvals_us": acc[2] / reps / 1000.0,
        "score_us": acc[3] / reps / 1000.0,
    }


def run(args):
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    veto_rates = [float(x) for x in args.veto_rates.split(",") if x.strip()]

    rows = []
    for veto_rate in veto_rates:
        corpus = build_corpus(
            size=args.corpus_size,
            veto_rate=veto_rate,
            approval_rate=args.approval_rate,
            veto_negative_threshold=args.veto_threshold,
            seed=args.seed,
            memory_rtt_us=args.memory_rtt_us,
            signal_rtt_us=args.signal_rtt_us,
            approval_rtt_us=args.approval_rtt_us,
        )

        if args.partitioned:
            partition = corpus.partition(rtt_us=args.memory_rtt_us)
            read_store = partition
            id_pool = partition.active_ids()
        else:
            partition = None
            read_store = corpus.memories
            id_pool = corpus.ids

        for batch_size in batch_sizes:
            batches = sample_batches(
                corpus, batch_size, args.distinct_batches, seed=args.seed + 1, ids=id_pool
            )
            # Pre-fetched signals for the compute-only mode, one list per batch.
            signal_batches = [corpus.signals.fetch_many(b) for b in batches]
            reps = reps_for(batch_size, args.work_budget)

            # How often does a batch contain at least one veto candidate? That
            # is how often the scorecard pays the extra approval round trip,
            # and it saturates to 1.0 quickly as batch size grows.
            # Measured over the timed batches themselves, so the reported rate
            # describes exactly the workload that produced the latencies.
            probe = NormalizedScorecard(veto_negative_threshold=args.veto_threshold)
            hits = sum(1 for sb in signal_batches if probe.veto_candidates(sb))
            approval_hit_rate = hits / float(len(signal_batches))

            for arm in ARMS:
                strategy = make_strategy(arm, args.veto_threshold)
                proj = StrengthProjection(
                    read_store, corpus.signals, strategy, corpus.approvals
                )

                if args.mode in ("e2e", "both"):
                    fn = lambda ids, p=proj: p.project(ids, args.explain)
                    samples = time_calls(fn, batches, reps, args.warmup)
                    row = summarize(samples, batch_size)
                    row.update(
                        mode="e2e", arm=arm, batch_size=batch_size, veto_rate=veto_rate,
                        approval_hit_rate=approval_hit_rate,
                    )
                    row["phases"] = attribute_phases(
                        proj, batches, min(reps, args.attribution_reps)
                    )
                    rows.append(row)

                if args.mode in ("compute", "both"):
                    fn = lambda sig, p=proj: p.compute_only(sig, args.explain)
                    samples = time_calls(fn, signal_batches, reps, args.warmup)
                    row = summarize(samples, batch_size)
                    row.update(
                        mode="compute", arm=arm, batch_size=batch_size, veto_rate=veto_rate,
                        approval_hit_rate=approval_hit_rate,
                    )
                    rows.append(row)

    return rows, {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "corpus_size": args.corpus_size,
        "approval_rate": args.approval_rate,
        "veto_threshold": args.veto_threshold,
        "explain": args.explain,
        "memory_rtt_us": args.memory_rtt_us,
        "signal_rtt_us": args.signal_rtt_us,
        "approval_rtt_us": args.approval_rtt_us,
        "work_budget": args.work_budget,
        "warmup": args.warmup,
        "seed": args.seed,
        "partitioned": args.partitioned,
    }


HEADER = (
    "%-8s %-17s %7s %6s %10s %10s %10s %10s %12s"
    % ("mode", "arm", "batch", "veto", "p50 us", "p95 us", "p99 us", "us/mem", "mem/s")
)


def print_table(rows, meta):
    print("")
    print("strength projection latency")
    print("  python %s / %s / %s" % (meta["python"], meta["machine"], meta["platform"]))
    print(
        "  corpus=%d approval_rate=%.2f veto_threshold=%d explain=%s"
        % (meta["corpus_size"], meta["approval_rate"], meta["veto_threshold"], meta["explain"])
    )
    print(
        "  store rtt us: memory=%.1f signal=%.1f approval=%.1f"
        % (meta["memory_rtt_us"], meta["signal_rtt_us"], meta["approval_rtt_us"])
    )
    print("  retrieval: %s"
          % ("active table only (vetoed quarantined at write time)"
             if meta.get("partitioned") else "single table (veto decided at read time)"))
    print("")

    ordered = sorted(
        rows,
        key=lambda r: (r["mode"], r["veto_rate"], r["batch_size"], ARMS.index(r["arm"])),
    )

    last_group = None
    for r in ordered:
        group = (r["mode"], r["veto_rate"], r["batch_size"])
        if group != last_group:
            if last_group is not None:
                print("")
            print(HEADER)
            print("-" * len(HEADER))
            last_group = group
        print(
            "%-8s %-17s %7d %6.2f %10.2f %10.2f %10.2f %10.4f %12.0f"
            % (
                r["mode"],
                r["arm"],
                r["batch_size"],
                r["veto_rate"],
                r["p50_us"],
                r["p95_us"],
                r["p99_us"],
                r["per_memory_us"],
                r["throughput_per_s"],
            )
        )

    _print_deltas(rows)
    _print_phases(rows)


def _index(rows):
    idx = {}
    for r in rows:
        idx[(r["mode"], r["batch_size"], r["veto_rate"], r["arm"])] = r
    return idx


def _print_deltas(rows):
    idx = _index(rows)
    keys = sorted(set((r["mode"], r["batch_size"], r["veto_rate"]) for r in rows))

    print("")
    print("scoring cost per arm: p50 minus the noop control, in us")
    header = (
        "%-8s %7s %6s %10s %10s %10s %9s"
        % ("mode", "batch", "veto", "additive", "scorecard", "inline", "appr hit")
    )
    print(header)
    print("-" * len(header))
    for mode, batch, veto in keys:
        n = idx.get((mode, batch, veto, "noop"))
        if not n:
            continue
        base = n["p50_us"]
        cells = []
        for arm in ("additive", "scorecard", "scorecard_inline"):
            r = idx.get((mode, batch, veto, arm))
            cells.append("%10.2f" % (r["p50_us"] - base) if r else "%10s" % "-")
        hit = idx.get((mode, batch, veto, "scorecard"), {}).get("approval_hit_rate", 0.0)
        print("%-8s %7d %6.2f %s %8.0f%%" % (mode, batch, veto, " ".join(cells), 100.0 * hit))
    print("  appr hit is the share of batches containing at least one veto candidate,")
    print("  i.e. how often the read-time scorecard pays the extra approval round trip.")
    print("  inline reads veto standing off the signal record, so it never pays it.")


def _print_phases(rows):
    e2e = [r for r in rows if r["mode"] == "e2e" and "phases" in r]
    if not e2e:
        return
    print("")
    print("e2e phase attribution (mean us per call)")
    print(
        "%-17s %7s %6s %10s %10s %11s %10s"
        % ("arm", "batch", "veto", "retrieve", "signals", "approvals", "score")
    )
    print("-" * 75)
    e2e = sorted(
        e2e, key=lambda r: (r["veto_rate"], r["batch_size"], ARMS.index(r["arm"]))
    )
    for r in e2e:
        p = r["phases"]
        print(
            "%-17s %7d %6.2f %10.2f %10.2f %11.2f %10.2f"
            % (
                r["arm"],
                r["batch_size"],
                r["veto_rate"],
                p["retrieve_us"],
                p["signals_us"],
                p["approvals_us"],
                p["score_us"],
            )
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-sizes", default="1,10,100,1000")
    ap.add_argument("--veto-rates", default="0.0,0.15,0.40")
    ap.add_argument("--approval-rate", type=float, default=0.30,
                    help="fraction of veto candidates already re-approved")
    ap.add_argument("--veto-threshold", type=int, default=3,
                    help="negative feedback count that triggers a veto")
    ap.add_argument("--corpus-size", type=int, default=20000)
    ap.add_argument("--mode", choices=("e2e", "compute", "both"), default="both")
    ap.add_argument("--explain", action="store_true",
                    help="build the per-component breakdown on every score")
    ap.add_argument("--work-budget", type=int, default=200000,
                    help="target scored memories per configuration; sets rep count")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--attribution-reps", type=int, default=200)
    ap.add_argument("--distinct-batches", type=int, default=16)
    ap.add_argument("--partitioned", action="store_true",
                    help="retrieve from the active table only, with vetoed memories "
                         "moved to quarantine at write time")
    ap.add_argument("--memory-rtt-us", type=float, default=0.0)
    ap.add_argument("--signal-rtt-us", type=float, default=0.0,
                    help="simulated round trip to the signal service, busy-waited")
    ap.add_argument("--approval-rtt-us", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args(argv)

    rows, meta = run(args)
    print_table(rows, meta)

    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump({"meta": meta, "rows": rows}, fh, indent=2)
        print("")
        print("wrote %s" % args.json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
