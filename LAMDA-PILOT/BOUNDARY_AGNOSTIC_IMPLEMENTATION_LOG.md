# Boundary-Agnostic Streaming Implementation Log

Started 2026-07-18, autonomous session (user stepped away). Goal: extend
`stream_mixin.py`'s sample-driven, boundary-agnostic streaming to the full
budget-mode-eligible roster (seqlora, olora, inflora, sketchlora, treelora,
hidelora, rainbowprompt, progprompt), with the interval length now determined by
a genuine memory/sample-count constraint (not the historical epoch-count
approximation), and replacing `BudgetStreamManager` (class-contiguous byte
chunking) as the memory-constrained training design going forward.

## Design recap (confirmed with user before they stepped away)

- Data delivery stays completely ordinary: real tasks, in order, clean batches,
  classifier head grows only at real task boundaries. This is deliberate — it
  isolates the experiment to ONE variable (does the adapter's own bookkeeping
  mechanism need to know where task boundaries are), not conflated with
  classifier-head-growth timing.
- What's decoupled: each method's own "boundary action" (SeqLoRA's optimizer
  reset, O-LoRA's new-slot + orthogonality, InfLoRA's DualGPM + new slot,
  SketchLoRA's compression, and the 4 new ones below) fires on an independent
  clock, NOT synced to real task boundaries.
- The clock itself: previously (validated, produced the SVDLoRA/InfLoRA/O-LoRA/
  SeqLoRA table) an EPOCH-count clock (`boundary_mult * epochs`, e.g. 15 global
  epochs = "1.5 tasks' worth"). User confirmed (2026-07-18) this changes to a
  genuine MEMORY-CONSTRAINT-derived SAMPLE COUNT clock instead — resolves the
  approximation gap that existed for datasets with uneven per-task sample
  counts (imagenet-r class sizes range 41-344, ~8x spread; CIFAR-100 is exactly
  balanced so the old epoch-based approximation was exact there by coincidence).
  Boundary-check GRANULARITY: per-epoch (option "b" — track actual samples
  consumed in each completed epoch, check against the sample threshold after
  each epoch; do NOT interrupt mid-epoch/mid-batch). Simpler than per-batch
  checking, user confirmed this is fine.
- Confirmed: it IS possible (guaranteed, in fact) for two different adapter
  slots to train on the exact same task's images, if the sample-count boundary
  happens to fall in the middle of that task's own epoch range. This is not a
  bug -- it's the mechanism that stresses each method's bookkeeping-timing
  robustness.
- EASE/TUNA/CL-LoRA: explicitly OUT OF SCOPE, left untouched. Their disposable
  per-task proxy head has no generic reusable "slot" the way the other 8
  methods do -- decoupling their bookkeeping from real task boundaries would
  require inventing a "proxy head per chunk" mechanism, which reintroduces the
  same duplicate-class/no-faithful-merge problem that already excluded them
  from BudgetStreamManager. Root cause is architectural (task-scoped proxy
  head), not specific to which streaming mechanism triggered it.
- `BudgetStreamManager` (byte-budget, class-contiguous chunking): superseded.
  Not deleting the file (per user's "don't delete any files" instruction) but
  no longer the memory-constrained design going forward.

## Decisions made autonomously (flag for review when user returns)

1. **Task splits** (20 tasks minimum, 100 for omnibenchmark-1k, per instruction). Verified exact
   class/image counts directly from each dataset (not assumed from memory):
   - cifar224: 100 classes, 50000 train images, balanced (500/class). init_cls=5, increment=5 (20
     tasks, 5 classes each) -- already the existing convention, unchanged.
   - imagenetr: 200 classes, 24076 train images, UNBALANCED (41-344/class). init_cls=10,
     increment=10 (20 tasks, 10 classes each) -- already the existing convention, unchanged.
   - omnibenchmark1k: 1000 classes, 168718 train images (100-403/class). init_cls=10, increment=10
     (100 tasks, 10 classes each) -- NEW, matches the "100 tasks" instruction and keeps
     classes/task=10 consistent with imagenetr.
   - sun397: 397 classes, 19850 train images, balanced (50/class). Does NOT divide evenly into 20
     tasks. Chose init_cls=17, increment=20 (17 + 19*20 = 397 exactly, 20 tasks). AUTONOMOUS CHOICE
     -- flagging for review, an alternative even split wasn't available given 397 isn't a multiple
     of 20 or close round numbers.
   - food101: 101 classes, 75750 train images, balanced (750/class). Does NOT divide evenly into 20
     tasks either. Chose init_cls=6, increment=5 (6 + 19*5 = 101 exactly, 20 tasks). AUTONOMOUS
     CHOICE -- same caveat as sun397.

2. **Memory constraint multiplier.** User's instruction ("determine each memory constraint... in a
   way that makes sense" + "strictly by a memory constraint") specified the MECHANISM (sample-count
   clock, not epoch-count) but not an exact multiplier value. Used the historically-validated
   `1.5 x mean-samples-per-task` (matches the "3k/2" convention from the table that motivated this
   whole redesign) as the default, computed per-dataset since mean-samples/task varies a lot
   (sun397 992/task lowest, food101 3788/task highest, a 3.8x spread -- consistent with the
   earlier-recorded samples/task-equalization note in the plan doc). Resulting per-dataset memory
   constraints: cifar224=538.3MB, imagenetr=259.2MB, omnibenchmark1k=363.3MB, sun397=213.7MB,
   food101=815.6MB. **FLAGGING FOR REVIEW**: only one multiplier (1.5x) was implemented; the user
   may want additional severities (e.g. 1.0x/2.0x) swept later, analogous to the old two-tier
   250/500MB approach -- did not build those without a specific value to target, per the "leave
   gaps for genuinely-missing decisions" instruction.

3. **`_stream_new_optimizer`'s cosine-schedule `T_max` (shared helper used by O-LoRA/InfLoRA/
   SketchLoRA/SeqLoRA/TreeLoRA/HiDeLoRA/ProgPrompt).** Previously `self._stream_boundary_every`
   (a fixed epoch-count under the old epoch-clock design, removed by the sample-count rewrite --
   this was a real bug I introduced and caught before it shipped: every one of these methods'
   `_stream_new_optimizer()` call would have raised AttributeError). Chunks now have VARIABLE
   epoch length (depends on which task's data is active when the sample threshold fires), so
   there's no single correct "epochs per chunk" anymore. Used `self.epochs` (the standard
   per-task epoch count) as the T_max proxy. AUTONOMOUS APPROXIMATION -- flagging for review; a
   more precise alternative would track each chunk's actual realized epoch-length and rebuild the
   scheduler with that as T_max, but that's circular (you don't know a chunk's length until it's
   already fired), so some approximation is unavoidable here regardless.

## Bug found + fixed (pre-existing, not introduced by today's redesign)

**TIL evaluation under streaming was routing to the wrong adapter slot for O-LoRA and
InfLoRA (not SeqLoRA or SketchLoRA, which already override the relevant hook).**
`stream_mixin.py::_stream_eval`'s TIL loop iterates REAL task indices (`for t, (lo,hi)
in enumerate(ranges)`) and calls `self._forward_task(inputs, t)`. The default
`_forward_task` (`models/lora.py`) routes via `self._eval_adapter(task)`, whose default
implementation was simply `return task` -- i.e. "route to the LoRA slot numbered the
same as the real task index." That's correct for ordinary (non-streaming) training,
where task index IS slot index. Under streaming, though, adapter slots are indexed by
CHUNK (advanced on the decoupled bookkeeping clock), and chunk count generically
diverges from real task count (the whole point of the design) -- so for any real task
whose index exceeds the number of chunks that have actually fired so far, TIL eval
would route to a slot that's either the WRONG slot or literally never trained
(Kaiming-random A / zero B, never touched by `_stream_begin_chunk`/`_stream_end_chunk`).
CIL eval was unaffected (`_stream_cil_forward`'s default already routes via
`self._stream_slot()`, which IS correctly chunk-indexed). SeqLoRA (pins slot 0 always)
and SketchLoRA (always routes TIL to the compressed sketch slot) already override
`_eval_adapter` themselves and were unaffected.

Fixed: `stream_run()` now tracks `self._stream_task_to_chunk[ct] = self._stream_chunk`
at the end of each real task's own epoch loop (added the `_stream_end_task(ct)` hook at
the same point, needed anyway for HiDeLoRA below); `models/lora.py::Learner._eval_adapter`
now consults this map when present, falling back to the identity mapping otherwise (so
ordinary non-streaming runs are completely unaffected). This means **the historical
InfLoRA/O-LoRA TIL numbers in the table that motivated this whole redesign may be
unreliable for later-task checkpoints** -- flagging clearly rather than asserting either
way, since I have not re-run that historical comparison to check by how much (if at all)
the corrected numbers differ. CIL numbers in that table are NOT affected by this bug.

## Bug found + fixed (introduced by today's redesign, caught live during smoke testing)

**Duplicate/conflicting eval checkpoints when a chunk is smaller than one task's own
epoch range (now common -- chunk size is fixed samples, task size varies).**
`completed = global_epoch // epochs` only advances by 1 once `global_epoch` crosses a
NEW multiple of `epochs`. Under the OLD epoch-count clock, boundaries fired at REGULAR
multiples of a period tied to `epochs` itself, so this quotient was always distinct at
each firing. Under the new sample-count clock, MULTIPLE adapter-boundary events can
fire while still inside the SAME task's own epoch range (e.g. cifar224: task volume
2500 samples/epoch, memory constraint 3750 samples -- a boundary fires mid-way through
almost every task). Caught live: ProgPrompt/cifar224 smoke logged `completed=1` TWICE,
first at 98.4% CIL then at 36.2% CIL, from two different boundaries both firing before
task 1 had actually finished -- silently overwriting/duplicating that checkpoint's
result with an unrelated, less-trained state. Fixed: added a `last_eval_completed`
dedupe guard, only firing `_stream_eval` when `completed` is a genuinely NEW value.
Killed and restarted all in-flight smoke tests after this fix (their results predate
the fix and are invalid).

## Progress

**cifar224 smoke (5 tasks, 3 GPU lanes: GPU0=hidelora, GPU1=progprompt/seqlora/rainbowprompt,
GPU2=olora/inflora/sketchlora/treelora) -- restarted after the `last_eval_completed` fix above.**
Confirmed fix working live: progprompt now logs `completed 1` then `completed 2` as genuinely
distinct checkpoints (98.40/98.00 CIL/TIL, then 18.70/35.80 -- normal-looking CIL forgetting +
TIL-per-task curve, no duplicate/overwritten values), olora logs a clean `completed 1` (100.0/100.0,
expected for task 0's own held-out set at rank-8 capacity). All 3 lanes still running as of last
check, no Tracebacks/errors in any of the 8 logs.

**imagenetr smoke (5 tasks, added as extra validation on GPU4 -- genuinely idle 40GB card, 0 other
compute processes, confirmed via `nvidia-smi --query-compute-apps` immediately before launch) --
started as a 4th, independent parallel lane** (not touching the cifar224 lanes' queues) since
imagenetr's uneven per-class counts (41-344/class) are the actual motivating case for the
epoch-count -> sample-count clock redesign; cifar224 being perfectly balanced can't by itself
distinguish the old approximation from the new exact mechanism. Runs all 8 methods sequentially
(seqlora/olora/inflora/sketchlora/hidelora/treelora/rainbowprompt/progprompt) against
`imagenetr_{method}.json` (init_cls=10, increment=10, stream_budget_mb=259.22). `gen_stream_smoke_
configs.py` extended with a `DEVICE_OVERRIDE` dict so per-dataset device pins don't require hand-
editing every generated config. First method (seqlora) confirmed launched cleanly (device
resolved to cuda:4, config values loaded correctly, no errors in first log lines).

Will continue checking all 4 lanes to completion before finalizing this section further.

## BLOCKING ARCHITECTURAL GAP FOUND (2026-07-19) -- stopped here, needs user input

**Every one of the 8 methods preallocates exactly `nb_tasks` adapter/prompt slots at
network-construction time, and the new sample-count clock breaks that invariant.**

Root cause: the shared LoRA scaffold (`backbone/vit_lora.py:55-58`) builds
`lora_A_q`/`lora_B_q`/`lora_A_v`/`lora_B_v` as `nn.ModuleList([... for _ in
range(n_tasks)])` where `n_tasks = data_manager.nb_tasks` (20 for cifar224/imagenetr).
This is used by ALL SIX LoRA-scaffold methods (seqlora, olora, inflora, sketchlora,
hidelora, treelora). RainbowPrompt/ProgPrompt and TreeLoRA's `KD_LoRA_Tree` follow the
identical pattern with their own per-task `ParameterList`/tree structure, also sized
from `nb_tasks` at construction (confirmed by reading their allocation sites, not
assumed). Under the OLD `boundary_mult*epochs` clock, total folds over a full run
were intrinsically bounded to roughly `nb_tasks / boundary_mult` (folds only ever
advanced roughly in step with real tasks), so this preallocation was always safe.
Under the NEW clock, folds are driven by CUMULATIVE SAMPLES-INCLUDING-EPOCH-REPEATS
(confirmed correct per the user's own description: "two adapters CAN and WILL end up
training on the exact same real task's images" -- multiple folds within a single
task's own epoch range is intended, not a bug). This means total folds over a full
run scales with `total_tasks * epochs_per_task * samples_per_task / boundary_every_
samples`, which generically EXCEEDS `nb_tasks` for any non-trivial epoch count --
there is no reason it should stay bounded to `nb_tasks`, and empirically it does not.

**Direct evidence (cifar224 smoke, 538.33MB constraint = 3750-sample threshold,
5-class/2500-sample tasks):**
- **HiDeLoRA** (tuned_epoch=50): crashed with `IndexError: index out of range` on
  `A_list[t]` inside `_warm_start()`, chunk index having climbed past 19 WITHIN just
  the first real task's own 50-epoch loop (folds every ~1.5 epochs -> ~33 folds from
  task 0 alone).
- **O-LoRA** (tuned_epoch=10): crashed identically (`IndexError: index 20 is out of
  range`, `backbone/vit_lora.py:272` `mlist[task]`) after 3 completed real tasks --
  slower to hit the wall (lower epoch count) but hit it regardless.
- **ProgPrompt** (tuned_epoch=5): completed the 5-task SMOKE run cleanly (finished
  exit 0, FINAL CIL 6.44/TIL 49.08) -- but only because 5 real tasks x 5 epochs x
  2500 samples / 3750 = ~16.6 folds, just under the 20-slot cap. Extrapolating to
  the FULL 20-task run: ~250,000 sample-units / 3750 = ~66 folds -- this WOULD crash
  partway through a full run (around real task ~6 of 20). Its smoke-test "pass" is
  a truncation artifact, not evidence of immunity.
- **SeqLoRA**: immune by construction (always trains/routes to a single fixed slot 0,
  never advances a per-task index) -- confirmed genuinely safe regardless of fold
  count, on both cifar224 and imagenetr smoke.
- InfLoRA/SketchLoRA/TreeLoRA/RainbowPrompt: not directly crashed yet (killed
  pre-emptively once the pattern was clear, to avoid burning further GPU-time on
  guaranteed-eventual crashes) but share the identical `nb_tasks`-sized preallocation
  and WILL hit the same wall in a full run, by the same arithmetic.

**Action taken:** killed all 4 in-flight GPU lanes (cifar224 x3 + imagenetr x1) and
their queue-driver scripts once the pattern was confirmed across 2 independent
crashes + 1 explained near-miss, rather than let the remaining queued methods burn
GPU-hours toward the same guaranteed failure. GPUs 0/1/2/4 are now idle (verified via
`nvidia-smi`). No files deleted, no other users affected (GPU3, the 4GB DGX card,
was never touched; all 4 GPUs used were confirmed zero-other-process before every
launch).

**Why this needs user input rather than an autonomous fix:** the correct remediation
touches the SHARED backbone (`backbone/vit_lora.py`) used by every LoRA method in
BOTH streaming and non-streaming (regular task-incremental) modes, plus TreeLoRA's
`KD_LoRA_Tree` and RainbowPrompt/ProgPrompt's own prompt pools -- a high-blast-radius
change across the whole method roster, not a local one-file fix. It also requires a
judgment call this log shouldn't make unilaterally:
1. **Confirm the intended fold-count regime.** Does "memory constraint" streaming
   really mean folds can outnumber real tasks by 3x-10x+ over a full run (consistent
   with a genuine byte-budget/streaming framing, decoupled from task semantics
   entirely), or was something closer to "roughly one fold per task, just not
   synced to boundaries" intended (which would instead call for a MUCH larger
   `stream_budget_mb` multiplier than the 1.5x currently used, keeping fold count
   near `nb_tasks`)? These lead to very different fixes.
2. **If many-fold-than-tasks is intended**, slot storage needs to become either (a)
   dynamically growing (append a fresh `nn.Linear`/`nn.ParameterList` entry per
   fold instead of preallocating, straightforward for the LoRA scaffold since LoRA
   adapters are tiny, but changes a widely-shared file used by every method's
   non-streaming path too -- needs care to leave that path's behavior unchanged),
   or (b) preallocated to a computed worst-case (`ceil(total_run_samples *
   epochs_per_task / boundary_every_samples) + margin`) instead of `nb_tasks`, only
   when `boundary_mode=="sample"`.
3. **TreeLoRA's `KD_LoRA_Tree`** and **RainbowPrompt/ProgPrompt's prompt pools** would
   each need an equivalent, but structurally different, fix (tree-shaped vs flat
   list) -- worth deciding the general strategy once, rather than re-deriving it per
   method.

Not attempting any of the above without direction, per the explicit "leave gaps for
genuinely-missing decisions" instruction -- this is squarely that kind of gap.
Everything else in this log (the redesigned clock semantics themselves, the 8
methods' hook wiring, the TIL-routing fix, the duplicate-checkpoint fix, the task
splits/memory constants) is unaffected by this finding and remains in place; only the
SMOKE-TEST VALIDATION of the full pipeline is blocked pending this decision, since
every method will eventually hit this wall on any run long enough to matter.

Stopping autonomous work here per the original instruction ("once all methods have
been adjusted... or you can safely do nothing else, you can stop and wait").

## RESOLVED 2026-07-19: dynamic slot growth (user decision + implementation)

**User decision**: dynamically add one slot per fold, rather than preallocate any
upper bound -- "Why don't we dynamically add a slot at each sketching boundary,
rather than allocating them all to begin with?" Confirmed one-at-a-time append
(no batched headroom) is fine. This answers gap-question 1/2 above: many-folds-
than-tasks IS the intended regime; slot storage must grow on demand.

**Memory-constraint convention finalized**: flat **250MB and 500MB** across every
dataset (user-confirmed), superseding the earlier per-dataset 1.5x-mean-samples/
task values (538.33MB/259.22MB/etc, which were only ever a smoke-test placeholder
computed before this question was settled). Matches the same 250/500MB convention
already settled for the superseded BudgetStreamManager design.

**Implementation** (all only active under `boundary_mode=="sample"`; the regular
task-incremental path is completely untouched -- still preallocates exactly
`nb_tasks` slots at construction, never calls any of the new growth methods):
- `backbone/vit_lora.py`: `Attention_LoRA.add_task_slot()` / `VisionTransformer.
  add_task_slot()` -- appends one fresh (A,B) pair per q/v, matching the
  constructor's device/dtype/init convention (kaiming A, zero B) exactly.
  Passthrough added to `utils/inc_net.py::LoRAVitNet.add_task_slot()`.
- `utils/inc_net.py::get_backbone`: `_lora`/`_progprompt`/`_rainbowprompt`
  branches now start with **1 slot** (not `nb_tasks`) when `boundary_mode==
  "sample"` (still respects an explicit `lora_n_slots` override, e.g.
  sketchlora's own small fixed pool, unaffected).
- `backbone/vit_progprompt.py::VisionTransformer.add_task_slot()` -- appends one
  fresh per-task prompt Parameter (`nn.init.normal_(std=0.02)`, matching ctor).
- `backbone/rainbowprompt_module.py::RainbowPromptModule.add_task_slot()` --
  RainbowPrompt's per-task storage is DENSE tensors indexed by task (not a
  ModuleList of separate params), so growth here concatenates a fresh row onto
  `base_knowledge[layer]`/`base_key`/`stored_prompts` (uniform_() / zeros,
  matching ctor) rather than appending a new module.
- `utils/kd_tree.py::KD_LoRA_Tree`: `all_accumulate_grads` (a plain list) now
  grows on demand in `end_task` instead of assuming the `num_tasks` preallocation
  is an upper bound; `num_of_selected` (rebuilt fresh every epoch anyway) now
  sized to the actual `task_id` rather than the fixed `num_tasks`, removing the
  dependency entirely.
- Each of olora/inflora/hidelora/treelora's `_stream_begin_chunk` now calls
  `self._network.add_task_slot()` (guarded `if self._stream_chunk > 0`, since
  slot 0 already exists from construction); rainbowprompt/progprompt call their
  own backbone's `add_task_slot()` at the equivalent point.
- Config generator (`scripts/gen_stream_smoke_configs.py`) updated to the flat
  250/500MB convention (`BUDGETS_MB = [250, 500]`, applied uniformly instead of
  per-dataset).

**Verified**: targeted 1-task cifar224 re-run of HiDeLoRA (the fastest-crashing
method) and O-LoRA at 250MB (harder threshold than the original 538.33MB crash) --
both completed with NO crash, confirming slot growth past the old 20-slot ceiling
works (O-LoRA cleanly, HiDeLoRA revealed a SEPARATE bug, see next section).

## Bug found + fixed (unmasked by the slot-growth fix): `_stream_end_task` ordering

HiDeLoRA's targeted re-verify above did not crash from the slot bug, but crashed
with a NEW error: `RuntimeError: stack expects a non-empty TensorList` in
`_predict_task_ids` (`models/hidelora.py:257`). Root cause: `stream_mixin.py`'s
intra-loop eval check (`completed = global_epoch // epochs`) can reach `completed
== ct+1` (the just-finished real task) WHILE STILL INSIDE that task's own `for _ep
in range(epochs)` loop -- specifically when a fold boundary happens to land on the
task's very LAST epoch. The post-loop `self._stream_end_task(ct)` call (which
computes HiDeLoRA's per-class centroids) only fires AFTER that loop fully exits,
so the intra-loop eval could fire BEFORE any centroids existed at all for a
never-yet-`_stream_end_task`'d task -- `all_means` was empty, `torch.stack([])`
crashed. Every other method's `_stream_cil_forward` is chunk-scoped only (routes
via `_stream_slot()`, no dependency on `_stream_end_task`'s side effects), so this
was invisible everywhere else. Was always latent in the redesign but never
triggered before today, since HiDeLoRA always hit the (now-fixed) slot-overflow
crash first, before any eval ever got the chance to reach this path.

Fixed in `stream_mixin.py::stream_run`: added a `last_task_ended` guard tracking
the highest real-task index that has had `_stream_end_task` called. Since
`completed` always equals `(just-finished real task index) + 1` exactly (every
real task runs precisely `self.epochs` epochs regardless of fold timing), the
intra-loop eval branch now calls `_stream_end_task(completed - 1)` itself first
if it hasn't already fired, before evaluating; the original post-loop call site
is now guarded (`if ct > last_task_ended`) so it doesn't double-call. Re-verified:
HiDeLoRA's 1-task re-run no longer crashes.

## HiDeLoRA is genuinely broken under this streaming design (not a bug in today's fixes)

User asked to evaluate HiDeLoRA (and the rest of the roster) under a REALISTIC
setting rather than the artificial 1-task/immediate-fold stress config used to
verify the crash fixes: launched all 8 methods on the real ImageNet-R 20-task
split at 250MB (stop_after_tasks=5, matching the historical seqlora_inr precedent),
across 4 idle GPUs (0/1/2/4, 2 methods/lane).

**6 of 8 methods produced normal, healthy CIL/TIL curves** (SeqLoRA, O-LoRA,
InfLoRA, SketchLoRA, RainbowPrompt all finished; TreeLoRA/ProgPrompt still running
as of this writing) -- e.g. SeqLoRA CIL [94.3,84.2,81.1,63.2,61.8], InfLoRA CIL
[94.6,87.8,82.7,72.2,74.5] (lower forgetting, as expected), RainbowPrompt CIL
[92.1,89.0,84.3,82.2,79.0] (low forgetting). All qualitatively consistent with
normal continual-learning behavior.

**HiDeLoRA collapsed completely**: CIL [9.5, 2.2, 2.3, 2.4, 2.7], TIL/task
[9.5, 11.8, 15.3, 11.2, 12.3] -- i.e. it never learned even a SINGLE task (10-way
chance is ~10%, and TIL/task -- each task's own masked accuracy, independent of
CIL routing -- never rises meaningfully above chance across all 5 checkpoints).
This rules out the initial hypothesis that the 1-task synthetic verify test was
an unrepresentative extreme (user pushed back correctly: "we fold at the end of
the interval, like we always have; HiDeLoRA isn't any different" -- the fold
TIMING mechanism is identical for every method, confirmed by 6/8 methods sharing
it and working fine).

**Likely cause** (not yet root-caused with a targeted diagnostic -- flagging as
hypothesis, not confirmed fact): HiDeLoRA is the only method in the roster with
`tuned_epoch=50` (vs. 10 for the rest of the LoRA-scaffold family), a real,
previously-validated gain in the ORIGINAL task-incremental (non-streaming) setting
(isolation-tested, see plan doc §12: +6.09pp at task 1). Under this streaming
design at 250MB, a fold fires roughly every 1-2 epochs relative to task volume --
so a 10-epoch method experiences ~5-10 folds per real task, while HiDeLoRA's
50-epoch budget produces ~30+ warm-start+momentum-blend cycles WITHIN a single
real task, before that task's training is even done. HiDeLoRA's warm-start
(slot t <- copy of slot t-1) + momentum-blend (`(1-m)*trained + m*mean(all prior
slots)`, m=0.1) was designed assuming folds roughly track real task boundaries
(as in the original non-streaming design and the old epoch-count clock) -- ~30
successive blend-toward-history operations before a task's own training has
converged even once plausibly self-reinforces toward an uninformative average,
which would explain never-above-chance TIL (a training-quality failure, not a
CIL-routing/forgetting failure).

**Status**: confirmed as a real, reproducible finding (not an artifact of any bug
introduced this session) via 2 independent runs (1-task synthetic + 5-task real
ImageNet-R) both showing the same collapse, contrasted against 6 other methods
sharing the identical mechanism running normally.

## Ablation series (2026-07-19, isolating HiDeLoRA's collapse)

**RainbowPrompt epoch=10 ablation (control)**: re-ran RainbowPrompt on the same
ImageNet-R/250MB setup with `tuned_epoch` overridden from its own 20 down to 10
(matching the O-LoRA/InfLoRA/SeqLoRA/SketchLoRA family), to check whether elevated
epoch count alone (independent of any cross-slot mechanism) could explain
degradation under frequent folding. Result: CIL [93.4,87.4,79.2,77.5,74.5] vs. the
epoch=20 run's [92.1,89.0,84.3,82.2,79.0] -- both healthy, normal forgetting
curves, the epoch=10 version only slightly worse (as expected from less training
per task). **Rules out epoch count as the driver.** Combined with a direct code
read confirming `models/rainbowprompt.py` has no `warm_start`/`momentum`-style
mechanism at all (each new slot is an independently-initialized, independently-
trained dense-tensor row, no copying/blending from history), this pointed
specifically at HiDeLoRA's warm-start+momentum-blend chain as the likely cause,
not fold frequency or epoch count in general.

**HiDeLoRA momentum-blend-disabled ablation** (`lora_momentum: 0.0`, already a
supported config value, no code change needed -- both `_momentum_blend()` call
sites were already gated `if self.lora_momentum > 0`): checkpoint 1 result CIL/TIL
11.71 -- **still collapsed**, barely different from the un-ablated 9.49. Momentum-
blend alone is NOT sufficient to explain the collapse.

**Diagnostic code change (flagged, small/reversible)**: added a `self.lora_
warmstart = args.get("lora_warmstart", True)` gate to `models/hidelora.py`
(mirrors the existing `lora_momentum` pattern exactly), and wrapped both
`_warm_start()` call sites (`incremental_train`'s non-streaming path, line ~130;
`_stream_begin_chunk`'s streaming path, line ~364) with `if self.lora_warmstart:`.
Defaults to `True` everywhere -- every existing config, the non-streaming path,
and every other method are completely unaffected unless a config explicitly sets
`"lora_warmstart": false`. Purpose: isolate warm-start from momentum-blend, since
disabling momentum alone didn't fix the collapse and warm-start is the only other
HiDeLoRA-unique (vs. the rest of the roster) bookkeeping mechanism.

**HiDeLoRA fully-cold ablation** (`lora_momentum: 0.0` AND `lora_warmstart:
false` together -- launched, in progress as of this writing) -- tests whether a
version with NO cross-slot state transfer at all (each new slot trained purely
independently, matching RainbowPrompt/O-LoRA/etc's own successful design pattern)
can learn at least one task. Result pending.

**HiDeLoRA fully-cold ablation -- result: WORSE, not better.** FINAL (5 tasks)
CIL 2.07 / TIL 8.20 -- the worst of the three variants tested (default: CIL 2.69/
TIL 11.95; momentum-off only: CIL 2.63/TIL 12.89; fully cold: CIL 2.07/TIL 8.20).
This falsifies the working hypothesis that warm-start or momentum-blend were
*causing* the collapse -- removing warm-start entirely made things worse, meaning
it was providing a genuinely useful head-start, just not enough of one.

## Root cause identified (not a bug -- an architectural mismatch)

User pushback ("why does each slot only survive one to two epochs? what resource
is being consumed?") forced a more precise re-examination, since the "HiDeLoRA's
50 epochs vs. everyone else's 10" framing didn't actually explain anything on its
own -- **every** LoRA-scaffold method's `_stream_begin_chunk` calls `freeze_to_
task(new_slot)`, which freezes every previously-created slot (including ones made
earlier within the SAME real task) and hands trainability to a fresh one with a
rebuilt optimizer. This fragmentation is IDENTICAL for O-LoRA/InfLoRA/SketchLoRA/
TreeLoRA, and none of them collapse. So "a slot only gets 1-2 epochs" cannot be
the root cause by itself.

**The actual distinguishing factor is what each method's DEPLOYED forward does
with that pile of frozen, briefly-trained slots**:
- O-LoRA/InfLoRA/TreeLoRA (`merge=True`): CIL forward **sums every frozen slot's
  contribution**. Even though any single slot only received 1-2 epochs of
  gradient descent, the additive combination of many such briefly-trained slots
  -- each nudging correctly on a properly-labeled subset of that task's own data
  -- still converges to a well-trained overall function. Fragmentation is
  harmless because nothing is ever thrown away; it's structurally equivalent to
  training one adapter for many epochs, just spread across summed low-rank
  pieces.
- HiDeLoRA's deployed forward (`_predict_task_ids` + per-sample task-routed
  `merge=False`) **selects a single slot**, never sums. Whichever one slot ends
  up "owning" a real task at eval time benefits ONLY from the handful of epochs
  it personally received -- the training that happened on that task's other,
  earlier, now-frozen-and-abandoned slots is never recombined into it. Warm-start
  is the only channel that could carry any of that forward (confirmed: removing
  it made things measurably worse), and momentum-blend averages toward an
  increasingly diluted pool of similarly under-trained history (confirmed:
  removing it alone didn't help either) -- neither is a substitute for actually
  summing the accumulated training the way O-LoRA's forward naturally does.

**Verified this isn't threshold-size-dependent either**: re-ran the full 8-method
matrix at 500MB (3 tasks, 2x the 250MB threshold). HiDeLoRA: CIL [10.1, 3.9, 2.5],
TIL/task staying at 9-14% (chance) throughout -- IDENTICAL collapse pattern to
250MB. This is expected under the above explanation: doubling the byte threshold
only doubles how many of a task's own epochs elapse before the FIRST fold (still
a handful, still far short of the 50 HiDeLoRA needs), so the same slot-swap
fragmentation dynamic recurs regardless of budget size in any practically
reasonable range. All other 7 methods produced healthy curves at 500MB too
(better than at 250MB, as expected -- less frequent folding, more forgetting
resistance), consistent across both budgets.

**Conclusion**: this is not a bug introduced by today's redesign, and not fixable
via a config knob (momentum/warmstart on or off, or budget size) -- it is a
genuine architectural mismatch between HiDeLoRA's per-sample task-routed
(summation-free) inference and ANY streaming design where slot-training gets
fragmented across many brief windows. The one lever that would actually target
this (decoupling HiDeLoRA's slot-swap from the sample clock entirely and keeping
it task-scoped -- one slot trained fully per real task, matching how it works in
the original non-streaming design) would reintroduce real-task-boundary
awareness, which defeats the purpose of testing it under this regime at all --
functionally the same category of exclusion already applied to EASE/TUNA/CL-LoRA
for a related reason. Flagging this for the user's decision rather than
implementing it unilaterally: is HiDeLoRA excluded from boundary-agnostic
streaming outright (parallel to EASE/TUNA/CL-LoRA), or is there a middle ground
worth exploring (e.g. a HiDeLoRA-specific minimum per-slot epoch floor before a
fold is allowed to swap it out, at the cost of no longer being purely sample-
count-driven for this one method)? Not attempted without direction -- this is a
design decision, not a bug fix.

## ImageNet-R smoke matrix results (2026-07-19), 250MB and 500MB, all 8 methods

All results below use the flat 250MB/500MB convention, real ImageNet-R 20-task
split (init_cls=10, increment=10), stop_after_tasks=5 (250MB run) / 3 (500MB run,
per user request to see behavior with more data per interval).

**250MB (5 tasks) final CIL / TIL:**
- InfLoRA: 74.47 / 88.49 (lowest forgetting of the "normal" methods, as expected)
- O-LoRA: 77.35 / 90.86
- SketchLoRA: 71.96 / 89.99
- RainbowPrompt (own epoch=20): 79.04 / 92.62
- RainbowPrompt (ablation, epoch=10): 74.47 / 89.24 -- control confirming epoch
  count isn't what protects it (see ablation section above)
- SeqLoRA: 61.76 / 84.17
- TreeLoRA: 71.59 / 89.67
- **ProgPrompt: 6.95 / 55.88 -- severe CIL-specific collapse, see below**
- **HiDeLoRA: 2.69 / 11.95 -- total collapse, see above**

**500MB (3 tasks) final CIL / TIL:**
- SeqLoRA: 76.79 / 90.41, O-LoRA: 77.40 / 89.30, InfLoRA: 76.79 / 88.80,
  SketchLoRA: 83.65 / 93.44, TreeLoRA: 80.73 / 92.13 -- all healthy, all higher
  than their 250MB counterparts at the same checkpoint count (less frequent
  folding -> less forgetting, exactly as expected).
- **HiDeLoRA: 2.52 / 11.60 -- identical collapse pattern to 250MB** (see root-
  cause section above for why budget size doesn't matter here).
- RainbowPrompt/ProgPrompt: still running as of this writing. **Minor anomaly
  spotted, not investigated**: RainbowPrompt's 500MB checkpoint 1 shows CIL 91.77
  but TIL only 65.82 -- TIL is normally >= CIL (it's the easier, masked-to-
  known-classes eval), and at checkpoint 1 with only one task's classes in play
  they'd normally be identical or very close (compare the 250MB run's checkpoint
  1: CIL 92.09 == TIL 93.04, both essentially equal, as expected). Worth a look
  later -- possibly related to `stored_prompts` (used by TIL's `known_task` path)
  only being refreshed while its OWN chunk is actively training, going stale
  once folding moves on, analogous in spirit to HiDeLoRA's fragmentation but
  seemingly not fatal for RainbowPrompt's CIL path specifically. Not chased
  tonight -- flagged for follow-up, separate from both investigations above.

## ProgPrompt: a second, DIFFERENT failure mode -- CONFIRMED (2026-07-19)

ProgPrompt's 250MB result is NOT the same kind of failure as HiDeLoRA's. TIL/task
shows it genuinely learns each task reasonably well when trained (90.82, 76.2,
40.2, 38.18, 26.71 -- well above the ~10% chance floor for every task, including
later ones), but CIL collapses far faster than any other method (90.82 -> 6.95
over 5 checkpoints, vs. e.g. TreeLoRA's 84.18 -> 71.59 or InfLoRA's 94.62 ->
74.47). This pattern -- learns fine per-task, catastrophic CIL-specific failure
-- is architecturally distinct from HiDeLoRA's never-learns-anything collapse.

**Hypothesis confirmed via direct diagnostic** (not just code inspection):
`backbone/vit_progprompt.py::forward_features` concatenates `self.prompts[t] for
t in range(task, -1, -1)` -- i.e. EVERY slot from 0 up through whatever `task`
index is passed. `_stream_cil_forward` (`models/progprompt.py`) always passes
`task=self._stream_chunk` -- the CURRENT, latest CHUNK index (grows on the fold
clock) -- for every sample in every eval, regardless of which real task it
belongs to. TIL doesn't have this problem: `_forward_task` uses the SAME chunk
index a given real task's slot was originally trained with (via `_stream_task_
to_chunk`), so its concatenation length always matches training-time exactly.

Added a diagnostic-only log line (`models/stream_mixin.py::_stream_eval`, prints
`self._stream_slot()` alongside `completed` -- additive only, zero behavior
change, safe for every method) and re-ran ProgPrompt on the same ImageNet-R/
250MB setup:

| completed (real tasks) | chunk (CIL concat length) | CIL | TIL |
|---|---|---|---|
| 1 | 2 | 90.8 | 90.8 |
| 2 | 7 | 55.9 | 82.9 |
| 3 | 10 | 15.3 | 69.9 |
| 4 | 14 | 8.9 | 62.0 |
| 5 | 17 | 7.0 | 55.9 |

Chunk count diverges from real task count immediately and grows to 3.4x by the
end (17 slots concatenated for a 5-task run), tracking the CIL collapse curve
almost exactly while TIL stays well above chance throughout. **Confirmed, not
just hypothesized**: ProgPrompt's CIL forward feeds the transformer an
ever-lengthening prompt sequence it was never trained to process as a whole
(each individual slot was only ever trained with its OWN chunk-index's
concatenation length, never the final, much-longer one used at CIL eval time).
Same general "fold clock vs. real-task clock" divergence problem underlying the
HiDeLoRA investigation above, but a structurally different failure mode
(sequence-length train/eval mismatch vs. summation-free per-slot isolation) and
a different fix would be needed (e.g. capping CIL's concatenation to the chunks
associated with real tasks seen so far via `_stream_task_to_chunk`'s recorded
values, rather than always using the single latest chunk) -- not implemented,
flagging for a decision same as HiDeLoRA, since this also touches the deployed-
inference definition rather than being a pure bug fix.

**Excluded from the SUN397/Food101 workable-method smoke below**, alongside
HiDeLoRA, pending a decision on either method.

**RESOLVED 2026-07-19 (user decision): HiDeLoRA is excluded from this design
outright** -- "I don't think that's going to be used in either of the final
evaluations." No epoch-floor-gate repair (the mechanism proposed earlier in
this log to fix both HiDeLoRA and ProgPrompt) will be built for HiDeLoRA.
ProgPrompt's status is still open/undecided as of this writing.

## SUN397/Food101 workable-method smoke (2026-07-19), 250MB and 500MB

All 6 workable methods (SeqLoRA, O-LoRA, InfLoRA, SketchLoRA, TreeLoRA,
RainbowPrompt), both datasets, both budgets -- all healthy, no crashes.

**250MB final CIL/TIL:**
| method | SUN397 | Food101 |
|---|---|---|
| SeqLoRA | 52.14/87.30 | 63.83/93.20 |
| O-LoRA | 66.66/91.67 | 75.03/97.26 |
| InfLoRA | 66.70/91.38 | 72.72/94.82 |
| SketchLoRA | 50.76/82.54 | 73.62/96.69 |
| TreeLoRA | 65.03/89.79 | 77.31/96.82 |
| RainbowPrompt | 69.42/93.67 | 83.94/97.88 |

RainbowPrompt best on both (lowest forgetting, consistent with its ImageNet-R
profile). SUN397 CIL is uniformly lower than Food101's across every method --
consistent with SUN397 being the lowest-density dataset (50 img/class vs
Food101's 750) and thus the hardest per-task learning problem.

**Finding: Food101's 250MB and 500MB results are IDENTICAL (not a bug).**
Food101's per-task epoch volume (task 0: 6 classes x 750 img/class = 4500;
later tasks: 5 x 750 = 3750) exceeds BOTH thresholds (250MB=1741 samples,
500MB=3483 samples) -- so a fold fires every single epoch regardless of which
budget is configured, for every method's own tuned_epoch count. Verified
directly: O-LoRA/TreeLoRA's chunk-index diagnostic shows IDENTICAL chunk counts
at every checkpoint (9,19,29,39,49) for both 250MB and 500MB runs, and
byte-identical CIL/TIL curves as a result. SUN397, by contrast, shows the
expected budget sensitivity (O-LoRA: chunk 4 at 250MB vs chunk 2 at 500MB for
completed=1, roughly halved as expected) since its much lower per-class density
(50 img/class, task volumes 850-1000) stays under both thresholds. **Practical
implication**: for a dense dataset like Food101, distinguishing 250MB from
500MB (or likely any budget in the low-GB range) would require testing with a
method that has a low enough epoch count that its per-epoch volume doesn't
already exceed the larger threshold on its own -- or accepting that budget
choice is effectively moot for this dataset/method combination at these values.

## CIL-only made the real default (2026-07-19, user-confirmed)

Reiterating a decision already established earlier for `BudgetStreamManager`
(TIL is meaningless when the whole premise is that task boundaries/counts
aren't available) -- now enforced structurally for `stream_mixin.py` too,
rather than just mentally ignored in the output. `_stream_eval` previously
computed TIL (an extra `_forward_task` pass, plus per-task masked argmax) on
EVERY checkpoint unconditionally; now gated behind `args.get("stream_til",
False)`, off by default. When off: only the CIL forward pass runs (halving the
per-checkpoint eval cost), the log line is a plain `CIL {:.2f}` (no TIL/TIL-
per-task fields), and `trainer.py::_run_stream`'s summary/FINAL lines skip TIL
output entirely when the result dict's `til` field is `None`. The underlying
TIL computation is NOT deleted (matches this codebase's standing convention of
keeping-but-defaulting-off rather than ripping out capability) -- any config
that explicitly sets `"stream_til": true` still gets the full TIL computation
exactly as before. Verified live (cifar224/O-LoRA 1-task smoke): clean
single-line `CIL 100.00` output, no crash, no TIL fields present.

**Not retroactively applied to already-completed runs** -- every result in this
log up through this point (ImageNet-R, SUN397, Food101 matrices) was captured
before this change and still includes TIL numbers (harmless to have, just no
longer computed going forward). Any NEW streaming run from this point on will
be CIL-only unless `stream_til` is explicitly set.

## Summary of code changes this session (all flagged, all small/reversible)

1. `backbone/vit_lora.py`: `Attention_LoRA.add_task_slot()` / `VisionTransformer.
   add_task_slot()` -- new methods, additive only, never called outside streaming.
2. `utils/inc_net.py`: `LoRAVitNet.add_task_slot()` passthrough (additive); `get_
   backbone`'s `_lora`/`_progprompt`/`_rainbowprompt` branches start at 1 slot
   instead of `nb_tasks` ONLY when `boundary_mode=="sample"` (non-streaming
   completely unaffected).
3. `backbone/vit_progprompt.py`: `VisionTransformer.add_task_slot()` -- new,
   additive.
4. `backbone/rainbowprompt_module.py`: `RainbowPromptModule.add_task_slot()` --
   new, additive.
5. `utils/kd_tree.py`: `KD_LoRA_Tree.end_task`/`tree_search` -- `all_accumulate_
   grads` grows on demand instead of assuming `num_tasks` is an upper bound;
   `num_of_selected` sized to actual `task_id` instead of fixed `num_tasks`.
   Behavior-preserving for every existing (non-overflowing) call pattern.
6. `models/olora.py`/`models/inflora.py`/`models/hidelora.py`/`models/treelora.py`/
   `models/rainbowprompt.py`/`models/progprompt.py`: each `_stream_begin_chunk`
   now calls the appropriate `add_task_slot()` before referencing a new slot.
   Only active under streaming.
7. `models/stream_mixin.py`: added `last_task_ended` guard so `_stream_end_task`
   fires before any eval that needs its side effects (fixes the HiDeLoRA
   centroid-crash found after fix #1-6 above unmasked it), guarded to never
   double-call.
8. `models/hidelora.py`: added `self.lora_warmstart` diagnostic gate (default
   `True`, mirrors the existing `lora_momentum` pattern) -- **diagnostic-only,
   used for tonight's ablation series, not otherwise load-bearing**. Every
   existing config and the non-streaming path are unaffected unless a config
   explicitly sets `"lora_warmstart": false`.
9. `scripts/gen_stream_smoke_configs.py`: flat `BUDGETS_MB = [250, 500]`
   convention, replacing the earlier per-dataset 1.5x-mean-samples/task values.

None of these touch the non-streaming (regular task-incremental) path's
behavior for any method -- every change is gated behind `boundary_mode==
"sample"` or an explicit new opt-in config key with a default that preserves
prior behavior exactly.

## Scope changes, 2026-07-19 (user decisions, after the ablation series above)

- **HiDeLoRA excluded from the final evaluations outright** -- "I don't think
  that's going to be used in either of the final evaluations." No epoch-floor-
  gate repair will be built for it (the mechanism proposed to fix both
  HiDeLoRA and ProgPrompt's collapses, see the ablation-series section above).
  ProgPrompt's status is still open.
- **OmniBenchmark-1k dropped from the final evaluation entirely** -- "I don't
  know that it's studied well enough yet, and it's the longest evaluation by
  far." Only CIFAR-100, ImageNet-R, Food101, SUN397 remain in scope.
- **Rank-8 adapters enforced for EASE and TUNA** (previously their own papers'
  r=16/ffn_num=64) to match the roster-wide convention already used by the
  LoRA-scaffold family.
- **Batch=128, lr=0.0005 enforced for every adapter-based method** (SeqLoRA,
  O-LoRA, InfLoRA, SketchLoRA, TreeLoRA, EASE, TUNA) across every dataset/split
  -- resolves a real batch/LR mismatch found during a literature cross-check:
  our LoRA-scaffold family had been running InfLoRA's own tuned LR (0.0005) at
  batch=48, but InfLoRA's authors tuned that LR specifically for batch=128 (a
  ~2.7x mismatch, unquantified but plausibly non-trivial for AdamW's effective
  step size). Applied to 17 config files (`exps/review/task_incremental_imr5t/
  *.json` HP sources + `exps/final/*.json` run configs), then all streaming
  smoke configs regenerated from the updated sources. **EASE/TUNA's optimizer
  was also switched from their native `sgd` to `adamw`** -- not literally
  requested, but flagged clearly at the time: pairing an Adam-scale LR
  (0.0005) with SGD (previously tuned at 0.02-0.05 for these two methods)
  would have undertrained them by 40-100x; reverted if this reading is wrong.
  CL-LoRA was NOT touched (not mentioned, not part of the current active
  roster) -- flagged as an omission for the user to confirm either way.
- Two Food101 500MB jobs (RainbowPrompt, SketchLoRA) that were already running
  when the batch=128 change landed were left to finish rather than killed --
  RainbowPrompt is unaffected (not adapter-based), but SketchLoRA's result
  predates the new convention and would need a re-run if it matters.

## Single-task sanity check (2026-07-19): does anything look catastrophically undertrained?

User request: train every current method (the 6 workable streaming methods +
ProgPrompt + EASE + TUNA, 9 total; HiDeLoRA excluded per above) on just task 0
of each of CIFAR-100/ImageNet-R/Food101/SUN397's 20-task split (regular,
non-streaming `_train` path, `stop_after_tasks=1`), to catch any regression
from tonight's HP changes before running anything longer. 36 jobs total,
configs in `exps/review/single_task_check/`, 3 GPU lanes (12 jobs each).
CIFAR-100 results so far, all under the new batch=128/lr=0.0005: SeqLoRA
100.0, O-LoRA 100.0, InfLoRA 100.0, SketchLoRA 99.4, TreeLoRA 100.0 -- no signs
of undertraining. Full results pending as the remaining lanes finish.

## Data-delivery redesign #2, 2026-07-19: unique-image memory budget (supersedes the epoch-repeat sample-count clock)

User identified a deeper problem with the "cumulative epoch-repeated samples"
clock (the one used for every result recorded above, in this section and
every section before it): training 10 epochs over the same 1000 images counts
as 10,000 toward the budget, even though nothing new was actually shown to the
model. This makes the "memory budget" a proxy for compute/time elapsed, not
for how much DISTINCT data the model has ever had access to -- and it's
exactly why Food101 produced identical results at 250MB and 500MB (its
per-epoch task volume alone exceeded both thresholds, so folding fired every
epoch regardless of the configured budget).

**New scheme** (user-specified, confirmed via restatement before implementing):
the budget now bounds how many UNIQUE images the model has ever seen, as a
genuine bounded-memory streaming simulation:
- Each real task gets ONE fixed random permutation over its own images (mixing
  its own classes together -- an "arrival order," distinct from ordinary
  per-epoch minibatch shuffling, which still happens normally).
- All tasks' permuted streams are concatenated task-by-task into ONE fixed,
  global sequence, then sliced into chunks of exactly `stream_budget_mb` worth
  of images. No image ever appears in two chunks; a chunk landing mid-task
  carries that task's remaining images into the very next chunk untouched (no
  repeats, no reshuffling, no omissions).
- A chunk trains for the method's own `epochs` epochs. Repeated epochs over a
  chunk's fixed image set no longer consume additional budget -- the budget
  now counts distinct images once, resolving the Food101 problem directly:
  fold count over a full run is now `total_images / budget`, a property of
  the DATA, completely independent of any method's epoch count.
- Classifier head size and CE-loss range are derived directly from whichever
  classes are actually present in the current chunk (`max(chunk labels)+1`),
  which can include a next task's classes the moment any of its images land
  in the current chunk -- this is still the one deliberately-unhidden
  confound (task/class identity fully visible via loss mask + head size,
  exactly as before); only each method's own adapter bookkeeping is decoupled.
- A real task becomes eligible for its own CIL checkpoint the moment a
  chunk's cumulative image count first reaches that task's own cumulative
  image count -- multiple tasks can complete within one large chunk (if
  budget spans several tasks), or one task can span many small chunks, with
  no dedupe/special-casing needed either way (each chunk advances the
  "highest completed task" pointer monotonically, at most through each task
  once, by construction).

**Structural decisions flagged to the user before/while implementing**:
1. The outer loop had to become CHUNK-major rather than task-major -- a chunk
   combining data from two tasks can't be trained until BOTH tasks' data is
   known, so `stream_run()` now precomputes the entire chunk schedule up
   front and iterates over chunks, deriving task-boundary bookkeeping (head
   growth, task-completion, eval checkpoints) from each chunk's own contents.
2. Classifier head sizing is computed directly as `max(chunk_targets)+1`
   rather than by reasoning about "which tasks does this chunk touch" --
   equivalent outcome, simpler and less error-prone to implement.
3. CE-loss range for a chunk spanning multiple tasks reuses the EXACT same
   `[min(targets), max(targets)+1)` pattern already built and validated for
   `utils/budget_stream.py::ce_range()` (the now-superseded BudgetStreamManager
   design hit the identical problem -- a carryover chunk with two different
   classes' worth of labels needing a CE range wider than the naive
   `[known,total)` growth-based slice). Low-risk reuse of already-proven logic.
4. Per-task shuffle granularity: ONE flat permutation over a task's full image
   set (mixing classes), not an exact per-class quota -- the user's own
   example's "100 images/class" was described as the expected OUTCOME of a
   uniform shuffle, not a strict requirement; a flat shuffle is more faithful
   to "an agnostic stream with no visible structure" than an engineered exact
   quota would be. Flagged as an interpretive choice, not the only valid one.
5. **Confirmed via direct code inspection that none of the 8 methods' own
   hook overrides (`_stream_begin_chunk`/`_stream_end_chunk`/`_stream_train_
   epoch`/`_stream_cil_forward`/`_stream_end_task`) needed to change at all**
   -- every one of them already takes the chunk's loader and an explicit
   `(lo, hi)` CE range as arguments, agnostic to how chunk boundaries were
   computed. The entire rewrite is confined to `stream_run()`'s own data-
   preparation and outer-loop logic in `models/stream_mixin.py`.
6. Data fetching reuses `DataManager.get_dataset()`'s existing `appendent`
   parameter (raw `(data, targets)` arrays, with an empty `indices` list)
   rather than adding a new DataManager method.

**Implementation**: `models/stream_mixin.py::stream_run()` rewritten in full
(module docstring and `trainer.py`'s adjacent comment block updated to match).
**Verified offline** (no GPU needed for this part -- pure data-partitioning
logic against a real `DataManager`, cifar224, 5 tasks, 3750-image budget):
no-overlap/full-coverage assertion passes across the whole 12,500-image
stream; the chunk sequence exactly reproduces the user's own example shape
(chunk 0 = task 0 full + task 1 partial; chunk 1 picks up task 1's *other*
half + all of task 2, correctly completing two real tasks in one chunk).
**Live GPU smoke test: DONE, clean pass** (O-LoRA, cifar224, 250MB, 5 tasks,
GPU4 the moment it freed up). Confirmed at every checkpoint against hand
computation: `[stream] unique-image budget 1741 images (250.0MB); 12500 total
images, 5 tasks -> 8 chunks` (5 tasks x 2500 images = 12500, matches). Task 0
(2500 images) correctly does NOT complete within chunk 0 alone (1741 < 2500),
completing instead at chunk 1 (`completed 1 | chunk 1`, CIL 100.0) -- exactly
the "full task + partial next task" shape from the user's own example. Chunk 3
(index 3) correctly produced NO eval line at all, since it didn't complete any
new real task (task 2 needed chunk 4 to finish) -- confirms chunks that don't
complete a task are silently skipped for eval purposes, no wasted/duplicate
checkpoints. Full run: `completed [1,2,3,4,5] | chunk [1,2,4,5,7] | CIL
[100.0, 97.6, 96.07, 93.4, 93.64]` -- a normal, healthy forgetting curve, no
crashes, no errors, matches hand-computed chunk/task-completion arithmetic at
every single checkpoint. **Redesign #2 is fully implemented and verified**,
both offline (pure partition logic) and live (full GPU training+eval cycle).
