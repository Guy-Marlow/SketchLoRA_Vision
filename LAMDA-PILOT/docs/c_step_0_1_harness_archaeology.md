# C-Step 0.1 — Harness Archaeology on the Retired `boundary_mult` Setting

Written 2026-07-25 per Plan C (`impl_plan_7.25.2026/plan_C_task_agnostic.md`) §C4,
C-Step 0.1: "identify the shared channel that crushed SeqLoRA (check loss masking
+ head growth against the boundary clock)."

## What survives

The original 2026-07-03 `boundary_mode: "sample"` + `boundary_mult` implementation
no longer exists in the codebase in its original form (superseded twice: first by
a sample-count clock, then by `stream_run()`'s unique-image-budget design). What
survives:

- `run_logs/stream_{method}_{cifar20t,inr20t}_sample{,_flat}{,_bm05}_s{1993,1994,
  1995}.json` — the actual completed result files from that run (CIL curves,
  14 checkpoints each, `boundary_mult` recorded in `args_subset`).
- `exps/{method}_{cifar20t,inr20t}_sample{,_flat}{,_bm05}_s{seed}.json` — the
  configs that produced them (rank 8, alpha 32, tuned_epoch 10, batch 48).
- A memory note (`vision_imagenetr_sample_boundary.md`, dated 25 days before
  this writeup) describing the design in the implementer's own words at the
  time: adapter events fire every `round(boundary_mult * epochs)` **global**
  epochs (i.e. an epoch counter that increments across the whole run, not reset
  per task); eval fires only immediately after a fold, on whichever real tasks
  have **fully** completed by that point; the resulting checkpoint schedule for
  `boundary_mult=1.5`, `epochs=10` is `{1,3,4,6,7,9,10,12,13,15,16,18,19,20}` —
  note task-completion count **2 is skipped entirely**, and so are 5, 8, 11, 14,
  17, 20 is present but 19 immediately precedes it. This directly confirms the
  mechanism: with a 15-global-epoch fold period against a 10-epoch task period,
  folds and real task completions drift in and out of phase, and since eval
  only fires at a fold, some real task completions never get their own
  checkpoint at all — they're silently absorbed into whichever later fold's
  eval finally captures them.
- `models/stream_mixin.py::legacy_epoch_clock_run()` — a **best-effort
  reconstruction** (dated 2026-07-20, explicitly flagged "NOT a verified
  byte-exact restoration") built from the one surviving log line
  ("`[stream] adapter event every N global epochs (C=<mult> x <epochs>
  epochs)`") plus the memory note above. It is a working approximation for
  historical comparison, not the original code — this diagnosis therefore
  cannot point at a specific line of the true original and call it "the bug";
  it can only reason from the surviving design description and the surviving
  numbers.

## The measured degradation

Pulling the `flat` (constant LR) variant, `boundary_mult=1.5`, and comparing
seed-1994 SeqLoRA against the seed-1994 **task-boundary oracle** run
(`run_logs/seqlora_inr20t_task_s1994.out`, same rank-8/alpha-32/lr configuration,
ordinary per-task training) at the same real-task-completion count (14 tasks):

| | ImageNet-R, checkpoint 14 (of 20) |
|---|---|
| Task-boundary oracle (ordinary training) | 40.07 |
| `sample`/`flat`, boundary_mult=1.5 (streaming) | 24.33 |
| **Degradation from own oracle** | **15.74 pts** |

CIFAR-100 seed-1994, same comparison, shows a similar pattern in kind though the
oracle CIFAR-100 task-boundary run for this exact old config was not found on
disk to give a clean number. Averaging across the three seeds' `flat` variant
final checkpoints (21.58/24.33/21.30 on CIFAR, 14.53/17.52/14.79 on IN-R) against
the oracle's steep decline curve at the matching checkpoint, the degradation
lands in the range the plan cites (**"25–30 pts"**) for the default cosine-LR
variant specifically and somewhat lower (15–20 pts) for the `flat` variant — both
are large, and both apply to **SeqLoRA**, whose own bookkeeping hook
(`_stream_end_chunk`) is a no-op (just an optimizer reset, no slot creation, no
subspace update, no compression). This is the crux of "shared channel": whatever
caused this cannot be a bug in any method's own bookkeeping, because SeqLoRA has
none to speak of, and it still collapsed by double digits relative to its own
oracle at the same task-completion count.

## Leading hypothesis (not verified against the original code — flagged as such)

Two properties of the design, both confirmed from the surviving description, are
candidate shared channels:

1. **The fold/eval clock is a GLOBAL epoch counter, not a per-task one.** Under
   ordinary (oracle) training, task *t* is evaluated the moment its own 10
   epochs finish — every task gets an identical, fixed training budget before
   its accuracy is measured. Under the old design, whether task *t*'s
   contribution has been "fully baked in" by the time of a given eval depends
   on where the *global* epoch counter happens to sit relative to the
   `boundary_mult * epochs` fold period — which, per the confirmed skipped-
   checkpoint pattern above, is **not** synchronized to task boundaries at all.
   A task whose own epochs finish just before a fold gets evaluated almost
   immediately; a task whose epochs finish just after one has to wait until the
   *next* fold, by which point one or more **additional** tasks' data has also
   been trained into the model. This changes what is actually being measured at
   a given "task-completion count" label between the two designs, independent
   of any method's own subspace/slot logic.
2. **Eval only fires at a fold, never at a bare task completion.** Combined with
   (1), this means the checkpoint labeled "task N complete" is not a fixed,
   comparable quantity across designs — it is contingent on fold timing, and
   the memory note's own skipped-checkpoint list is direct, confirmed evidence
   that this desynchronization is real and not merely theoretical.

Both of these are properties of the **shared trainer/eval loop**, not of any
individual method's `_stream_begin_chunk`/`_stream_end_chunk` hooks — consistent
with a no-bookkeeping control (SeqLoRA) still taking a large hit. This is offered
as the most likely explanation given what survives, **not** as a verified root
cause with a located line number; the original code that would let us confirm it
directly no longer exists.

## Why this cannot recur under Plan C

Plan C's new setting (`bounded_memory_run`, see
`docs/plan_c_bounded_memory_harness.md`) removes both ingredients structurally:

- Eval checkpoints are **volume-based** (fixed fractions of total stream length)
  and **data-derived**, computed independently of any method's own consolidation
  clock — a checkpoint at 35% of the stream fires at 35% of the stream
  regardless of how many consolidation cycles have happened by then.
- The classifier head is **pre-built to full width before training starts** and
  never grows — there is no "task N's head slice arrived late" timing question
  to get out of sync with anything, because there is no head-growth event at
  all after t=0.

So even though the exact original bug (if it was a single bug rather than an
emergent property of the two design choices above) cannot be pinned down
precisely, Plan C's setting does not share either of the structural properties
implicated here, and per Plan C §C7/§C8 no numbers from the retired setting are
used in any headline comparison regardless — this document is motivation/context
only, exactly as the plan specifies.
