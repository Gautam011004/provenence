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

    arm                     vs quality   vs popularity   prec@1000   junk@1000
    additive                     0.283           0.989       0.457         323
    additive_capped              0.309           0.981       0.798          49
    additive_tuned               0.664           0.536       0.809          52
    additive_tuned_capped        0.689           0.472       0.889          13
    additive_all_capped          0.698           0.477       0.889          13
    scorecard_equal              0.786           0.380       0.912          10
    scorecard_tuned              0.812           0.364       0.934           9

The grid separates two things that are easy to conflate. Caps fix the top of the
ranking and barely move the overall order: they only bind above the ceiling, and
most memories sit below it — at k=5000 capping alone changes junk not at all
(1963 either way). Weights fix the overall order but leave junk near the top.
You need both.

`scorecard_equal` is the untuned starting point: every signal weighted 1/5, each
capped at its share. It already matches a hand-tuned capped sum without anyone
choosing a weight, which is the case for the structure rather than for any
particular calibration.

## Per-signal contribution caps

`influence_budget()` reports the most each signal can contribute to one score:

                         eval  validation  retrieval  recency  negative  history
    sum (plain)         10.00         inf        inf      inf       inf      inf
    sum (input caps)    10.00       12.50       5.00      inf       inf      inf
    sum (all capped)    10.00       12.50       5.00    12.50     12.50    12.50
    scorecard_equal     25.00       25.00      10.00    20.00     20.00    20.00

Two things this makes visible. Capping the count inputs of a raw sum still
leaves recency, negative feedback and failure history unbounded — they are
counts and elapsed days, and they grow forever too. And a cap bounds a signal
without making it proportionate: cap retrievals at 500 while the weight is 1.0
and retrieval can still contribute 500 against eval's 10.

Caps are applied after weighting and are independent of the weights, so retuning
a weight later cannot lift a signal above its ceiling. A weight is a preference;
a cap is a guarantee. `TestContributionCaps` pins this.

### Retrieval is capped below everything else

Retrieval frequency is held to half the ceiling the other signals get. This is
about proportion, not distrust — retrieval carries real information, and a
memory that keeps getting pulled up is probably worth having. But it is the
signal most able to swamp the rest: it is an unbounded count, and it is the only
one that feeds back on itself, since a high score surfaces a memory more often,
which raises its retrieval count, which raises its score. An eval result does
not improve because a memory was shown.

The headroom the lower ceiling frees goes to eval and validation, the two
signals that speak most directly to whether a memory is correct. Budgets still
total 1.0, so `max_score()` stays a true 100 and the scale remains readable.

`bench/quality.py` cannot see this at all — it scores one snapshot with no loop
running, so every popularity number it reports is a floor.

Its *weight* stays at parity. The cap is the safety ceiling that survives
retuning; the weight is the calibration, and the weight is what to tune against
real labels. A sweep over both is in the history of this file: lowering the
weight ranks slightly better than clamping the cap at the same value, because
clamping ties every heavily-retrieved memory at the ceiling and discards the
ordering among them. Use both — weight for quality, cap for safety.

Passing an explicit uniform `cap` does still lower the reachable maximum —
constraining every signal cannot preserve the range. `max_score()` reports it,
and anything presenting strength as a percentage should divide by that rather
than assume 100.

### Penalty ceilings

Penalties carry the same ceiling as the positive signals: negative feedback and
failure history weigh exactly as much against a memory as any single signal
weighs for it, and neither can annihilate a score alone.

Levelling them (down from 45 and 36) *improved* correlation with quality, 0.751
to 0.783. The oversized penalties had been flooring 8.6% of the active corpus at
zero, collapsing that whole slice into a tie and throwing away the ordering
within it; at the levelled caps only 2.0% floors. The cost is a slightly
stronger pull toward popularity, 0.397 to 0.463, since weaker penalties leave
the positive signals relatively more say. Junk in the top 1000 was unchanged at
9 either way.

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

## Ingest: PDF to memories

`ingest/` turns a source document into `Memory` + `Signals` records and loads
them into the stores. Built and exercised against `FIFA-2026.pdf`, the 184-page
FIFA World Cup 26 Assets Brand Manual.

    .venv/bin/pip install pypdf cryptography
    .venv/bin/python -m ingest FIFA-2026.pdf --out results/fifa_memories.json
    python3 tests/test_ingest.py     # 38 tests, stdlib only, no PDF needed

    pdf.py       PDF -> RawPage        the only module needing a dependency
    segment.py   RawPage -> Candidate  where the judgement lives
    admit.py     Candidate -> Memory + Signals, deduped, cold-start priors
    pipeline.py  wiring and a report

Segmentation and admission are stdlib-only and tested against hand-written page
fixtures, so the judgement calls are pinned without a 43MB fixture in the repo.

### What comes out

    pages read        184
    candidates found  707
    duplicates merged 121
    memories admitted 586

    guideline  520     prose: "The Legal Notice may be reduced proportionally..."
    rule        56     explicit do/don't: "Don't omit the Legal Notice."
    spec        10     colour tables, kept whole

Three kinds because they need different handling. Prose is hard-wrapped across
lines and has to be rejoined into sentences. Rules arrive atomic. Colour specs
are tables — split per line they become a dozen useless fragments, since
"HEX #FFFFFF" means nothing without the "WHITE" above it.

Dropped: headings, image captions, layout labels, and the placeholder Lorem
Ipsum used to demo typography. The test for admission is whether the text is
something someone could later be right or wrong about.

### Where it is weak

Colour specs on multi-column pages. The extractor reads every swatch name first
and every value block after, so when the counts match they can be paired
positionally, and when they don't the attribution is a guess. Those records are
emitted with `confidence=0.4`, which flows through to `source_trust` — the
values are real, it is the naming that is shaky, so they are admitted at reduced
trust rather than dropped or asserted. 12 of 586 records are affected. Fixing it
properly needs a layout-aware read of character coordinates, or a vision pass.

### The cold start is the real limitation

Every extracted memory scores **exactly the same**. Across all 586, there are
two distinct strengths: 50.5 for cleanly extracted records and 39.7 for the
low-confidence specs. Nothing else separates them.

That is not a bug. Every signal the model reads is accumulated evidence — evals
that ran, validations given, retrievals counted, complaints received — and a
memory extracted five seconds ago has none of it. Ingest writes a marked prior
(`eval_score` 0.5, `provisional=True`) rather than a measurement.

So strength does no work at ingest time. It starts discriminating once evals run
and usage accrues; until then, ranking freshly extracted memories is ranking
noise. `TestColdStartCarriesNoInformation` pins this so it is not mistaken for a
score.

The gap this exposes: the model cannot currently distinguish *weak* from
*unmeasured*. Both read as a middling number. If that distinction matters — and
for a system that decides what to surface, it probably does — it wants either a
separate confidence dimension alongside strength, or an explicit unmeasured
state that retrieval treats differently from a low score.

## Weakening: how each strategy responds to negative feedback

`bench/feedback.py` runs scenarios against the memories extracted from the brand
manual. The texts and provenance are real; usage history is synthetic and seeded,
because freshly ingested memories all carry the same prior and a ranking needs
something to discriminate on.

    python3 bench/feedback.py
    python3 bench/feedback.py --scenario popular_but_complained
    python3 tests/test_feedback.py     # 13 tests pinning these findings

Each cell is strength, then rank among 586 memories, #1 strongest.

### Complaints accumulating

                          baseline   1 complaint  2 complaints  3 complaints
    additive              262.5 #125  257.5 #131   252.5 #139    247.5 #147
    additive_all_capped    11.0 #334    6.0 #547     1.0 #586     -1.5 #586
    scorecard_equal        69.7 #346   54.7 #546    49.7 #569        VETOED

The raw sum barely reacts. Four complaints move it from #125 to #152 — still in
the top quarter of the corpus. The capped sum bottoms out immediately. The
scorecard bottoms out and then removes it at the threshold.

### The one that matters: popular and complained about

                          baseline  heavily used   1 complaint  2 complaints
    additive              262.5 #125 1164.0   #1   1159.0  #1   1154.0   #1
    additive_all_capped    11.0 #334   15.0  #99     10.0 #391      5.0 #558
    scorecard_equal        69.7 #346   75.5 #211     60.5 #502     55.5 #542

Under the uncapped sum, retrieval volume buys immunity. A memory with two
complaints sits at **rank 1 of 586** because it is heavily used, and would be the
first thing surfaced on every query. Capping retrieval's contribution is what
removes the immunity — the capped sum and the scorecard both push it to the
bottom fifth.

This is the same failure the synthetic quality benchmark measured as popularity
coupling, here on real memories with a concrete consequence.

### Eval failure

                          baseline    eval fails  approved back  90 days on
    additive              262.5 #125  253.2 #138   253.2   #138  244.2 #152
    scorecard_equal        69.7 #346      VETOED    49.5   #570   46.1 #581

The scorecard removes it on the spot. The raw sum moves it thirteen places: a
failed eval is one bounded term among several unbounded ones, so it barely
registers.

### Recovery works

                          baseline  2 complaints  re-validated  evals pass
    scorecard_tuned        64.7 #402   44.7 #578    50.2  #559   67.3 #361

Validations and passing evals pull a memory back, and a reinstated one climbs as
the history penalty fades. Weakening is not one-way.

### Two things worth deciding

**Capping demotes; only the veto removes.** Under a capped sum a memory with
forty complaints or a failed eval sits at #586 — last, but still retrievable, and
still returned if a query matches nothing better. If "never surface this again"
is a state you need, the sum alone cannot express it.

**Scorecard decay saturates.** Six months unused and two years unused score
within 0.3 points of each other, because recency is a decaying credit that
approaches zero rather than a penalty that grows. The raw sum keeps sliding
indefinitely. Whether an ancient memory should keep falling or should settle at a
floor is a policy question this harness does not answer — `TestNeglect` pins the
current behaviour either way.
