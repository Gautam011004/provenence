"""CLI: .venv/bin/python -m ingest FIFA-2026.pdf --out results/fifa_memories.json"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.pipeline import run, write_json  # noqa: E402
from ingest.admit import DEFAULT_EVAL_PRIOR, DEFAULT_SOURCE_TRUST  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf")
    ap.add_argument("--out", default=None, help="write extracted memories as JSON")
    ap.add_argument("--source", default=None, help="source name used in memory ids")
    ap.add_argument("--sample", type=int, default=0, help="print N extracted memories")
    ap.add_argument("--kind", default=None, help="restrict --sample to one kind")
    ap.add_argument("--source-trust", type=float, default=DEFAULT_SOURCE_TRUST)
    ap.add_argument("--eval-prior", type=float, default=DEFAULT_EVAL_PRIOR)
    args = ap.parse_args(argv)

    admitted, _stores, report = run(
        args.pdf,
        source=args.source,
        source_trust=args.source_trust,
        eval_prior=args.eval_prior,
    )
    print(report.summary())

    if args.sample:
        pool = [a for a in admitted
                if args.kind is None or a.candidate.kind == args.kind]
        print("\n--- sample of %d %s memories ---"
              % (min(args.sample, len(pool)), args.kind or "extracted"))
        step = max(1, len(pool) // args.sample)
        for a in pool[::step][:args.sample]:
            c = a.candidate
            print("\n  [%s] %s p%d" % (c.kind, c.section, c.page))
            print("  %s" % c.text[:300])

    if args.out:
        write_json(admitted, args.out, args.source or args.pdf, report)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
