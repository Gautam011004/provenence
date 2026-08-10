# memory strength: additive sum vs normalized scorecard

Test harness for choosing between two strength calculations, both running as a
read-time projection: strength is never stored next to a memory, it is computed
when memories are retrieved and attached on the way out.

    python3 tests/test_strategies.py     # 58 correctness guards
    python3 bench/latency.py --help      # latency
    python3 bench/quality.py --help      # ranking quality

## Layout

    memstrength/
      models.py       Memory (carries no strength), Signals, StrengthResult
      store.py        memory / signal / approval tables, optional simulated RTT
      veto.py         VetoPolicy + PartitionedMemoryStore (active / quarantine)
      strategies.py   RawAdditive, NormalizedScorecard, ScorecardInline, NoOp
      projection.py   the read-time join: ids -> memories + signals -> strength
      dataset.py      deterministic corpora, latency and quality flavours
    bench/
      latency.py      four arms x two modes, with phase attribution
      quality.py      ranking quality against a latent ground truth
    results/          saved runs

## The three designs measured

| | veto decided | approval state | read path |
|---|---|---|---|
| `additive` | never | n/a | 1 memory fetch + 1 signal fetch |
| `scorecard` | at read time | separate store | + a conditional approval fetch |
| `scorecard_inline` | at write time | on the signal record | same as additive |

`scorecard` and `scorecard_inline` compute identical scores — there is a test
asserting it. They differ only in where veto standing comes from.

## Result 1: the arithmetic difference is negligible

In-process, no network. Scoring cost only (p50 minus the no-op control):

    batch 1000, veto 0.15    additive 141us    scorecard 306us

Roughly 2x, driven by the `exp` recency decay and the saturating normalizers.
In absolute terms it is ~0.16us per memory, below the projection module's own
join overhead. Not a basis for a decision.

## Result 2: the approval round trip is the whole story

With 250us stores, `scorecard` pays one extra round trip whenever a batch
contains at least one veto candidate. That probability is `1 - (1-r)^n`, so it
saturates almost immediately:

    batch    veto 0.05    veto 0.15     effect when paid
       10          39%          80%     +250us on a ~500us request
       25          71%          98%
      100         100%         100%

Above roughly 25 memories per request it is not occasional, it is every
request: a flat ~50% latency increase.

## Result 3: moving the veto to write time removes it

`veto.py` decides standing when signals change and moves the memory between an
active and a quarantine table. Retrieval reads the active table only, so a
vetoed memory is never fetched, scored, or filtered. Standing also lives on the
signal record, so no second store is consulted.

    batch 100, veto 0.15, 250us stores    scoring cost above the no-op control
      additive                              12.1us
      scorecard          (read-time veto)  287.5us
      scorecard_inline   (write-time veto)  32.8us

The veto becomes affordable. Partitioning alone is not what does it: the saving
comes from the read path no longer consulting a second store. `scorecard` still
pays, because with approvals in a separate store it cannot tell a reinstated
memory from a blocked one without asking.

Two things this buys beyond latency:

- the approval queue is a read of one small table, not a scan of the corpus
- the read path has no veto branch to get wrong

And one thing it costs: read-time veto is always current, write-time veto is
only as fresh as the last transition. `PartitionedMemoryStore.resweep()` is the
drift repair, and revocations should be applied eagerly while approvals may lag
— the asymmetry matters, since a stale revocation serves a memory that should
be suppressed while a stale approval merely keeps suppressing a cleared one.

### Reinstatement leaves no gate behind

Once approved, a memory is moved back to the active table and retrieved like any
other. Nothing re-checks it, nothing re-flags it, and no second lookup happens on
its account. What survives is a penalty on its strength:

    identical memories, scorecard        strength
      never failed                          86.3
      failed once, reinstated today         74.3
      failed once, clean for a year         86.3

The penalty is bounded (a repeat offender cannot be driven to zero by history
alone) and decays with a 45 day half life, so a reinstated memory earns its way
back through clean evals and validations rather than being permanently marked.
Approach A carries the same history term but has no decay: under the raw
additive sum a single old failure marks a memory forever. That is a difference
between the two, recorded as such in `TestReinstatement`.

`Signals.failure_count` and `days_since_failure` carry this. `eval_failed` is
current state, not history -- it is an input to the veto decision, not a
permanent mark.

### Approvals must be scoped

`Signals` carries `approved_at` and `last_failure_at` rather than a bare
`approved` flag. An approval only lifts a veto if it postdates the failure it
covers, otherwise one approval immunizes a memory against every future failure.
Tested in `TestVetoPolicy.test_approval_before_failure_does_not_lift_veto`.

## Result 4: quality, on the active table

`bench/quality.py` generates memories with a latent `true_quality` no scorer
sees. Signals are noisy, biased views of it — critically, retrieval frequency is
driven mostly by an independent `popularity` term, so traffic is a confound
rather than a proxy for quality.

    arm               vs quality   vs popularity   prec@1000   junk@1000
    additive               0.283           0.989       0.457         323
    additive_tuned         0.664           0.536       0.809          52
    scorecard              0.751           0.397       0.928           9

`additive_tuned` is the same additive sum with the retrieval weight dialled
down, included so the comparison is not against a strawman.

Raw additive is essentially a popularity ranker (0.99 correlation with traffic,
0.28 with quality). Of its top 1000, 323 are below-median quality; it placed a
memory of true quality 0.018 high in its ranking purely on retrieval count. It
is also
unstable: perturbing signals by 5% churns 91% of its top 100, because an
unbounded count dominates the sum.

Tuning the weights recovers most of that, which is worth saying plainly — the
gap is mostly about unbounded counts, not about addition itself. But tuned
additive still trails on every metric, and it stays fragile: the weights are
correct only for the current range of the counts, and counts only grow.

## Caveats

- Latency numbers are single-machine, single-process CPython 3.9 on arm64.
  Ratios travel; absolute microseconds do not.
- Simulated store RTT is a busy-wait, chosen because `sleep` resolution (~1ms)
  is coarser than the effects measured. It burns CPU while waiting.
- `--distinct-batches` defaults to 16; the `appr hit` column is measured over
  exactly those timed batches, so it is coarse at small batch sizes. Pass a
  larger value for a tighter estimate.
- At batch 1000 the p99 column picks up allocator noise (occasional ~4ms
  outliers across all arms, including the no-op control). p50 and p95 are the
  trustworthy columns there.
- Quality results depend on the generative model in `build_quality_corpus`,
  particularly how strongly popularity is tied to quality. That coupling is
  deliberately Gaussian rather than a convex mixture; a mixture would make high
  popularity arithmetically imply high quality and quietly rig the comparison
  toward traffic-chasing scorers.
