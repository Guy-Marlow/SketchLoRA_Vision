# Plan C — Bounded-Working-Memory Boundary-Free Streaming: Implementation

Implements `impl_plan_7.25.2026/plan_C_task_agnostic.md` §C1. New, additive
codepath: `boundary_mode: "bounded_memory"` in `trainer.py` dispatches to
`models/bounded_memory_mixin.py::BoundedMemoryMixin.bounded_memory_run()`,
mixed into `models/lora.py::Learner` alongside the existing `StreamMixin`.
**No existing file's behavior changes** for any other `boundary_mode` value —
`models/stream_mixin.py` (sample/sample_legacy), `utils/budget_stream.py`
(budget), and the plain per-task loop in `trainer.py` are untouched.

## Config keys

| Key | Meaning | Required |
|---|---|---|
| `boundary_mode` | `"bounded_memory"` | yes |
| `bm_budget_mb` | Working-memory size B, in MB, same 224x224x3 bytes/image accounting as `stream_budget_mb`/`budget_mb`. **USER OVERRIDE 2026-07-25**: flat MB values (closest analogs to Plan C §C2's `{0.2x,0.5x,1x,2x}`-of-mean-latent-task-size spec are **{50, 75, 100, 200} MB**), not a fraction — supersedes the plan's own framing. | yes |
| `stop_after_tasks` | Real latent tasks included in the stream (still governs how much of the dataset is used); unrelated to cycle count. | no (defaults to full dataset) |
| `tuned_epoch` | E, epochs trained per memory cycle (Plan C: E=20, the campaign budget) | yes (existing key) |

SketchLoRA frozen-variant keys (`sketchlora_admission`, `sketchlora_rank_cap`,
`sketchlora_lora_wd`) compose with this exactly as with any other
`boundary_mode` — see `docs/sketchlora_frozen_variant.md`.

## What happens, concretely, in one cycle

1. Take the next `cycle_images = round(bm_budget_mb * 1024*1024 / (224*224*3))`
   images off the precomputed stream (task-major concatenation: each real
   task's own images in one fixed seeded permutation, tasks laid end to end —
   identical construction to `stream_run()`'s, copied not imported, so this
   file has zero import-time coupling to `stream_mixin.py`).
2. `_stream_begin_chunk(loader)` — method's own hook (O-LoRA/InfLoRA/TreeLoRA
   advance a slot + do their per-boundary setup; SeqLoRA/SketchLoRA use their
   existing overrides/defaults). Unmodified from `stream_mixin.py`'s version of
   these hooks.
3. Train `E` epochs of shuffled minibatches over that fixed cycle content
   (`_bounded_train_epoch`), **full-logit** cross-entropy — no `[lo,hi)` local
   class-window slicing anywhere, unlike every other training path in this
   codebase (plain per-task, budget-mode, and `stream_run()` all slice to a
   local window; this is Plan C's explicit departure, §C1: "Training loss
   computed over full logits").
4. `_stream_end_chunk(loader)` — the method's consolidation action (SketchLoRA
   merge, O-LoRA new-slot orthogonality snapshot, InfLoRA DualGPM recompute,
   TreeLoRA tree update). Unmodified hook.
5. Check whether cumulative images crossed one or more pre-computed volume
   checkpoints (5% steps / 20 points for short datasets, 10% steps / 10 points
   for Omni-1K, always ending at 100%); if so, run `_bounded_eval` for each
   newly-crossed checkpoint and write results incrementally to disk.

## Classifier head

Built **once**, to `data_manager.nb_classes` (the full, final label space),
immediately before the cycle loop starts. Never touched again — no `update_fc`
call anywhere else in this file. This is the one piece with no direct analog
to reuse from `stream_run()`, which grows the head incrementally as new classes
enter each chunk.

## Evaluation

CIL only (Plan C §C1: "TIL is not computed... no task identities exist").
`_bounded_eval` masks logits to `hi_total = 1 + max(class index among the first
N stream images)`, where `N` is the checkpoint's own image position — purely a
function of stream position, recomputed fresh from `all_targets` each time,
never read from any task/chunk counter.

**Per-task forgetting curve (added 2026-07-25, analysis-only):** `_bounded_eval`
now also returns a per-latent-task accuracy breakdown (accuracy on each real
task's own classes, at this checkpoint) alongside the pooled CIL number,
stored in each result dict as `per_task_acc`. This is computed purely by
slicing the SAME predictions the CIL forward pass already produces by which
task each test class belongs to (`task_class_cumends`, a cumulative CLASS
count per real task) — no new forward pass, no change to what the model sees,
no change to which checkpoint fires. Fixing this required correcting a real
bug caught in the same pass: the "nearest latent task" analysis label was
comparing a stream IMAGE-count position against an array of cumulative CLASS
counts (both had been stored in one variable, `task_sizes`, actually holding
class counts) — silently saturating to the last task index at every single
checkpoint (visible in the earlier CIFAR-100 smoke logs, which all reported
"~4" for a 4-task run regardless of real progress). Now split into
`task_image_cumends` (image-position lookups, "nearest latent task") and
`task_class_cumends` (class-index lookups, per-task eval breakdown). This was
an analysis/logging-only bug — it never affected training, the loss, head
growth, or any reported CIL accuracy number, only the informational label and
the (until now nonexistent) per-task breakdown. Verified via a quick CIFAR-100
smoke after the fix (not re-run against tonight's actual production numbers,
which remain valid and unaffected).

## InfLoRA's T concession (Plan C §C1)

InfLoRA's DualGPM threshold ramp (`models/inflora.py::update_DualGPM`) needs a
total-session denominator. Normally `self.total_sessions = args["nb_tasks"]`
(the real task count, set in `__init__`). Under bounded-memory streaming there
is no equivalently-meaningful real task count for this ramp (the whole point of
the setting is that consolidation events don't line up with real tasks), so the
harness computes `T = ceil(total_stream_images / cycle_images)` — a quantity
computable *a priori* from the budget and the dataset, per the plan's explicit
allowance — and sets it once via a new hook, `_bounded_set_total_sessions(T)`,
before the cycle loop starts. Default implementation (on `BoundedMemoryMixin`)
is a no-op; only `models/inflora.py` overrides it. No other method needs this
concession per Plan C §C1 ("No other concessions").

## SketchLoRA `lazy_merge` (Plan C §C1, non-default, §C8 sign-off gated)

Implemented in `models/sketchlora.py` (`args["lazy_merge"] = true`,
`lazy_merge_frac` default 0.9). When set, `_stream_end_chunk` skips
`_compress()` (no fold, no residual reset) unless `_lazy_should_fold()`
returns true, so the residual keeps training across multiple cycles instead
of being folded and reset every cycle. The saturation statistic: the
residual's own occupied rank (measured by the same energy-threshold rule used
for the main compression, applied to `B_r @ A_r` alone) as a fraction of its
allocated budget (`lora_rank`); folds once the mean fraction across all
q/v/block modules reaches `lazy_merge_frac`. This is an internal,
boundary-blind statistic — no data volume, cycle count, or real-task
information is read by it (leak-audit consistent). Not signed off per §C8, so
it never runs by default — only when a config explicitly sets `lazy_merge:
true`, and any such run should be labeled as the experimental arm, not folded
into headline numbers, per the plan's own gating. Not exhaustively tuned
(`lazy_merge_frac=0.9` is a reasonable starting default, not a swept value);
also does not force a final fold at the very end of a run (eval correctness is
unaffected either way since `_stream_cil_forward` always sums sketch+residual
regardless of fold timing — see SketchLoRA's `_stream_train_merge`/
`train_merge=True` — so this is a memory-footprint completeness detail only,
not a correctness issue).

## Leak audit (Plan C §C1, blocking)

Grepped `models/bounded_memory_mixin.py` for every read of `self._cur_task`,
`self._stream_chunk`, or any task/chunk index, and traced what it feeds:

- `self._cur_task = -1` / `self._stream_chunk = -1`: initialized once before
  the loop; **incremented only inside each method's own `_stream_begin_chunk`/
  `_stream_end_chunk` hooks** (e.g. InfLoRA's `self._stream_chunk += 1`), which
  is *cycle*-indexed bookkeeping the method is allowed to know about itself
  (which of its own consolidation events has fired) — this is not knowledge of
  real task identity or boundaries, and is unchanged from `stream_run()`'s
  already-accepted convention.
- `_bounded_train_epoch`: reads `self._stream_slot()` (which slot to route
  through — method-internal state) and computes loss via plain
  `F.cross_entropy(logits, targets)` over the full head width. No task index
  read anywhere in the loss path.
- `_bounded_eval`: `hi_total` is derived exclusively from `all_targets[:cum_
  images]`, i.e. stream position — never from `self._cur_task`/`_stream_chunk`.
- `task_cumends` / `_nearest_latent_task`: computed and stored in the results
  dict for **offline analysis figures only**; never read back by anything that
  affects training, the loss, head growth, or eval-checkpoint timing.
  `checkpoint_images` (what actually gates when eval fires) is computed purely
  from `total_images` and the fixed 5%/10% fractions, before the loop starts,
  and never touches `task_cumends`.
- `self._bounded_set_total_sessions(T)`: the one explicit, plan-sanctioned
  concession (InfLoRA only); `T` is derived from the cycle/budget structure,
  not from real task count, so it does not smuggle real task-boundary
  information into the method.

Eval-routing identity (Plan A §A4.3, extended to volume checkpoints per Plan C
§C1's last bullet): `_bounded_eval` always calls `self._stream_cil_forward`,
the same deployed-adapter forward every other streaming eval path uses — no
separate/ad-hoc forward call was introduced for this harness.

## Verified

End-to-end smoke (`exps/planc_smoke/seqlora_smoke.json`, SeqLoRA, CIFAR-100,
`bm_budget_mb=50`, `stop_after_tasks=4`, 2 epochs/cycle for speed): harness
correctly computed cycle=348 images from 50MB, 58 cycles, 20 eval checkpoints;
first checkpoint (5% of the 20,000-image stream) fired at cycle index 2 (cum
images 1044 >= 1000), reporting `classes_seen=10` (still within real task 0's
image span, as expected) and a sane CIL number. No crash, no assertion failure.
Full C-Step 1 (all 4 methods, B=0.5x-equivalent, CIFAR-100) is the actual
harness-verification gate per Plan C §C4 and is run separately (see
`docs/plan_c_execution_log.md`).
