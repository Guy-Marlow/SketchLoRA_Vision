# Bounded Bank Memory — Implementation Changelog

Itemized record of every change made to implement the "bounded bank memory"
setting (a byte-budget cap on each method's persistent, non-classifier-head
storage — distinct from `bounded_memory_mixin.py`'s existing sample-*volume*
budget). Design finalized and triple-audited before implementation began; see
the design doc discussion this session for the full rationale behind every
decision referenced below.

**Ground rule enforced throughout**: every change here is either (a) new,
additive code behind the `bank_cap_mb` config key (a complete no-op when
unset — the default), or (b) a change to `persistent_state()`'s byte
accounting (needed so the cap has something accurate to check against,
already decided as prerequisite work). Nothing else in any method's existing
behavior is touched. Pre-existing inefficiencies not directly caused by this
feature are left alone, even where a prior audit pass flagged them (e.g.
InfLoRA's extra wasted forward pass after freezing IS in scope, since it's
part of correctly scoping this feature's own gate — see below — but nothing
about InfLoRA's un-capped behavior is touched).

Config surface: one new key, `bank_cap_mb` (float, MB). Unset/`None` = 
feature off. One shared budget per method (no per-structure sub-budgets, per
decision).

---

## Shared infrastructure

### `models/bank_cap_mixin.py` (new file)
`BankCapMixin` — minimal shared plumbing, matching `stream_mixin.py`'s
shared-driver/per-method-hook convention:
- `_bank_cap_init(self, args)`: reads `args.get("bank_cap_mb")` into
  `self._bank_cap_bytes` (`None` if unset — feature off), initializes
  `self._bank_frozen = False`.
- `_bank_bytes(self)`: abstract (raises `NotImplementedError`) — each
  participating Learner must override with its own current non-head bank
  byte count. Deliberately not defaulted to 0, so a method that mixes this
  in but forgets to override fails loudly instead of silently never
  freezing.
- `_bank_admit(self, candidate_bytes)`: the sole admission check. Returns
  `True` (admit) if the feature is off, or if `_bank_bytes() +
  candidate_bytes <= cap`. Returns `False` and permanently sets
  `self._bank_frozen = True` the first time growth would exceed the cap —
  a strict one-way latch; every call after that returns `False`
  unconditionally, regardless of what `_bank_bytes()` reports later
  (required for InfLoRA's monotonicity guarantee — see plan doc).
No per-method fallback behavior lives here — this file only ever answers
"may I grow?"; each method's own file decides what happens when the
answer is no.

---

## Per-method changes

### TreeLoRA (`models/treelora.py`, `utils/kd_tree.py`)
`Learner(BankCapMixin, LoRALearner)`; `_bank_cap_init(args)` called in
`__init__`. TreeLoRA's single LoRA adapter is pinned to slot 0 (never
banked, per the module's own 2026-08-11 architecture correction) — the only
genuinely growing persistent structure is the tree's gradient-snapshot store,
`KD_LoRA_Tree.all_accumulate_grads`, grown once per task in `end_task()`.
- `utils/kd_tree.py::KD_LoRA_Tree.end_task(self, task_id, admit=True)`: new
  `admit` param, checked FIRST — when `False`, the method is a complete
  no-op: `all_accumulate_grads` stops growing AND `kd_tree_root` stops being
  rebuilt. Everything else that reads tree state (`tree_search`/
  `insert_grad`/`get_loss`, all unconditional on `self.reg > 0`, not on task
  boundaries) keeps running against whatever the last successful `end_task()`
  call left behind — regularizes against the frozen snapshot forever, never
  disabled (matches the project-wide KD/regularizer decision). The
  already-shipped `None`-filtering fix in `tree_search`/`end_task` (commit
  `691080d`, prerequisite work done earlier this session) means no further
  change was needed there — a frozen, gapped `all_accumulate_grads` was
  already handled correctly by that fix.
- `models/treelora.py::_train()`: new admission check right before the
  existing `end_task()` call, only inside the pre-existing `if self.reg > 0:`
  guard. `candidate_bytes` = `self.tree.current_grad.numel() * 4` — mirrors
  `persistent_state()`'s own per-snapshot byte cost exactly (it's literally
  the same tensor `end_task()` is about to store if admitted).
- New `_bank_bytes()` hook: sums `all_accumulate_grads` exactly as
  `persistent_state()` already does (no separate accounting needed — this
  structure's cost was already fully counted, unlike TUNA/CL-LoRA's gaps).
- **Streaming path (`_stream_end_chunk`) deliberately left untouched** — this
  feature is oracle-CIL only (explicit user decision). `end_task()`'s new
  `admit` parameter defaults to `True`, so the un-migrated streaming call
  site (`self.tree.end_task(self._stream_chunk)`, no `admit` argument) is a
  byte-identical no-op change.

### InfLoRA (`models/inflora.py`)
`Learner(BankCapMixin, LoRALearner)`; `_bank_cap_init(args)` called in
`__init__`. Only `feature_list`/`feature_mat` (the DualGPM subspace bases,
grown/rebuilt inside `_update_dualgpm`) genuinely grow — `frozen_delta_q/v`
(fixed `[dim,dim]` size, `register_buffer`s allocated at full size from
construction — see the "empty bank" flagged entry below for why this
matters) and the current, not-yet-folded task's own live slot stay
fixed-size every task regardless of the cap (folding keeps them bounded),
so nothing about `add_task_slot()`/`free_folded_slots()`/`_train` needs
gating.
- New `_bank_bytes()`: `frozen_delta` + DualGPM bases (mirrors
  `persistent_state()`'s existing breakdown, minus `current_slot` — the live
  slot, never counted, same convention as every other method's live-slot
  treatment — and minus `fc`, always exempt). Because `frozen_delta_q/v` are
  always-allocated (not lazily grown by folding), this is ~54MB from task 0
  onward, not 0 — an extreme cap smaller than that refuses admission
  immediately, even at task 0 (see the flagged entry below for the
  `_init_lora_A` fix this required).
- `incremental_train()`: ONE coarse gate around the WHOLE `_update_dualgpm()`
  call (not a fine-grained per-column clip inside `update_DualGPM` — round-2
  finding confirmed `feature_mat`'s rebuild commits its full cost as a single
  all-or-nothing step regardless of how many columns get admitted, so
  fine-grained clipping is provably powerless). `candidate_bytes=0` — not a
  predicted growth amount (infeasible to predict cheaply without running the
  SVDs `_update_dualgpm` itself performs) but a check of whether the
  PREVIOUS cycle's growth already put `_bank_bytes()` at/over budget; once
  true, every later cycle's `_update_dualgpm` (including its own extra ViT
  forward pass and the `feature_mat` rebuild — the round-3 scope-widening
  finding, not just `update_DualGPM`'s inner column-admission step) is
  skipped in full, one-way-latched.
- `_init_lora_A()`: it reads whatever `self.feature_mat`/`self.project_type`
  currently hold to analytically set each new task's A — once frozen, those
  simply stop changing, so every future task's A is set orthogonal to the
  same frozen subspace forever, for free (matches "never disabled, just
  stale"). **This section originally said "no change needed" — WRONG, see
  the flagged entry below.** A required fix: the branch that assumed
  `self.feature_mat` always has an entry for every block once `_cur_task >
  0` crashes when an extreme cap refuses admission before `_update_dualgpm`
  ever runs once.
- Streaming hooks (`_stream_slot`, `_stream_begin_chunk`, `_stream_end_chunk`,
  `_bounded_set_total_sessions`) deliberately untouched — oracle-CIL only.

### O-LoRA (`models/olora.py`)
`Learner(BankCapMixin, LoRALearner)`; `_bank_cap_init(args)` called in
`__init__`. Discovery during implementation: `models/lora.py`'s shared
`incremental_train()` (inherited, unmodified) ALREADY routes every slot
decision through `self._train_adapter()` (default_task assignment, the
`add_task_slot()` growth guard, and `freeze_to_task()`) rather than
`self._cur_task` directly — a pre-existing hook (used today by SeqLoRA to
pin slot 0). This meant only ONE override was needed to make the entire
existing slot-growth/training pipeline bank-cap-aware, not a new codepath.
(Note: `backbone/vit_lora.py::add_task_slot()`'s own docstring claims it's
"only ever called from streaming" — that's a STALE comment; `utils/inc_net.py`
confirms slots are grown one-at-a-time from oracle `incremental_train()` too,
verified directly by reading both files. Left as-is, not in scope.)
- New `Learner._train_adapter()` override: returns `self._cur_task`
  (no-op) unless a prior task's slot-growth admission was refused, in which
  case it returns the pinned last-admitted slot forever after. The check
  (`_bank_admit(_olora_slot_bytes())`) only fires when a new slot is actually
  about to be needed (`self._cur_task >= n_tasks`) — once `add_task_slot()`
  actually runs (if admitted), that condition goes false for the rest of the
  task, so the check doesn't re-fire on every training step despite
  `_train_adapter()` being called there too. Once frozen,
  `add_task_slot()`'s own existing guard permanently stops firing (the
  pinned slot always already exists) and `freeze_to_task(pinned_slot)`'s own
  existing unconditional `requires_grad` re-enable keeps the pinned slot
  fine-tuning continuously every task — both "stop growing" and "never
  reinitialize the post-cap slot" fall out for free from hooks that already
  existed, no separate re-arm code needed (unlike TUNA/CL-LoRA).
- New `_olora_slot_bytes()` / `_bank_bytes()`: independent, correct
  byte-unit accounting (mirrors `persistent_state()`'s top-level `total_bytes`
  computation, which is correct) — NOT reusing `persistent_state()`'s
  `breakdown["lora_slots"]` value, which is a genuine pre-existing unit bug
  (raw param COUNT, not bytes — confirmed by reading `models/lora.py`) that
  would have silently corrupted admission decisions. Left unfixed (out of
  scope — an existing inefficiency/bug unrelated to implementing the cap
  itself, not something this feature needs to touch to work correctly).
- `_refresh_orth_cache()` / `_orth_and_l2()` / `_train()`'s main loop: the 4
  real `self._cur_task` reads used for adapter routing (not the tree/regularizer's
  own task-identity bookkeeping — there is none for O-LoRA) redirected to
  `self._train_adapter()` — no-op pre-freeze (always equal), and post-freeze
  this makes the orthogonality cache freeze too: once `t` stops changing
  across tasks, the cache's own `!= t` rebuild-guard goes permanently false,
  so it's built once at the freeze point and never rebuilt again (a free
  performance win, and exactly "orthogonalized against frozen bank forever").
- `_forward_task()` (TIL eval): same redirect, for consistency with CIL's
  `_deployed_forward` (`models/lora.py`, already routes through
  `_train_adapter()` unmodified).
- Streaming hooks (`_stream_slot`, `_stream_begin_chunk`, `_ce_aux_macs_per_step`)
  deliberately untouched — they use their own `_stream_chunk`-based routing,
  not `_train_adapter()`, so they're unaffected either way; this feature is
  oracle-CIL only per explicit decision.
- Known, accepted non-monotonicity (round-3 finding, unchanged by
  implementation): near very high slot counts the orthogonality penalty's
  free complement subspace shrinks toward zero, which can collapse the
  pinned slot's training signal — accuracy vs. budget likely peaks then
  declines past that threshold. Documented as expected behavior, no code
  change.

### RainbowPrompt (`models/rainbowprompt.py`)
`Learner(BankCapMixin, StreamMixin, TILLearner)`; `_bank_cap_init(args)` +
`self._rp_pinned_slot = None` in `__init__`. `base_knowledge`/`base_key`/
`stored_prompts` grow by ROW CONCATENATION per task (`add_task_slot()`,
`backbone/rainbowprompt_module.py`) — the row-based storage mirrors the
LoRA-family slot pattern closely enough that the same "override the one hook
that decides which index to use" strategy (as O-LoRA's `_train_adapter()`)
applies directly.
- New `_rp_slot()`: mirrors `models/olora.py::_train_adapter()` exactly —
  no-op (`self._cur_task`) until a new row's admission is refused, then
  permanently returns the pinned last-admitted row. `add_task_slot()`'s own
  growth guard, redirected to call this, then naturally stops firing once
  frozen (pinned row always already exists) — same free "stop growing"
  property as O-LoRA.
- New `_rp_slot_bytes()` / `_bank_bytes()`: mirrors `persistent_state()`'s
  existing (already-accurate) total minus head — no persistent_state() gap
  existed here, unlike TUNA/CL-LoRA.
- New `_rp_task_to_slot(task)`: oracle-mode remap for TIL's ground-truth task
  ids (which can reference ANY past task, not just the current one — unlike
  `_rp_slot()`'s single current-task value). `min(task, pinned_slot)`.
- Redirected: `incremental_train()`'s `default_task` assignment and
  `add_task_slot()` growth guard; `_train()`'s main loop's `t`; `_eval_cnn()`
  and `_deployed_forward()`'s `task_id` (technically already self-limiting
  via `base_key[:task_id+1]`'s slice-clips-to-actual-size semantics even left
  as `self._cur_task`, but redirected anyway for explicitness/consistency
  with every other site in this file, rather than relying on that implicitly).
- No re-arm code needed for the pinned row's `requires_grad`:
  `incremental_train()`'s existing per-task loop already does an
  unconditional `bk.requires_grad_(True)` on each layer's WHOLE
  `base_knowledge` tensor (+ `base_key`) every task regardless — because
  `nn.Parameter` can't be frozen by row, this was already a blanket
  per-tensor un-freeze, not a per-row one, and the forward pass only ever
  reads/writes the pinned row's slice, so gradient still lands only there.
  Continuous fine-tuning of the pinned row falls out for free.
- **Caught during implementation, not by the three design-review passes**:
  `models/rainbowprompt.py` defines `_forward_task` TWICE in the same class
  body — an oracle-flavored version and, further down (near
  `_stream_cil_forward`), a streaming-flavored one using
  `_stream_task_to_chunk`. Python silently keeps only the LAST definition;
  the oracle-flavored one is dead code at runtime in BOTH oracle and
  streaming modes (this matches a round-3 finding already on record: "the
  originally-suspected `_forward_task` bug site is dead code... the live one
  already degrades safely" — but that finding was about a DIFFERENT,
  pre-existing bug search, not about where THIS feature's TIL remap needed
  to live). I initially wrote the bank-cap TIL remap into the dead copy
  before grepping for a second definition and finding the live one further
  down — caught before considering the method done, by re-checking which
  copy actually executes rather than trusting the first match. Fixed: the
  remap (`_rp_task_to_slot`) now lives in the live copy, chained after the
  existing `_stream_task_to_chunk` remap (mutually exclusive in practice —
  `_stream_task_to_chunk` only exists under streaming, `_rp_pinned_slot`
  only ever gets set from oracle-path methods — so each remap is an identity
  whenever the other one is active). The dead copy is left in place,
  unmodified, with a comment explaining why.
- Streaming hooks (`_stream_init`, `_stream_slot`, `_stream_begin_chunk`,
  `_stream_train_epoch`, `_stream_cil_forward`) deliberately untouched —
  oracle-CIL only.

### CL-LoRA (`models/cllora.py`, `backbone/vit_cllora.py`, `utils/inc_net.py`)
`Learner(BankCapMixin, BaseLearner)`; `_bank_cap_init(args)` called in
`__init__`. Three structures grow together every task boundary
(`backbone.add_adapter_to_list()`, called from `after_task()`):
`adapter_list` (specific-block adapters, a plain python list), `block_weight_
list` (banked block-weighting vectors, also a plain list), `old_adapter_list`
(full-`cur_adapter` KD-teacher snapshot, an `nn.ModuleList`). ONE atomic
admission check covers all three (not 3 independent checks) — `forward()`/
`forward_test()` index `adapter_list` and `block_weight_list` by the SAME
position `i`, so letting them diverge in length would silently pair a task's
adapter with the WRONG task's block-weight vector.
- `persistent_state()`: now counts `old_adapter_list` (was a proper
  `nn.ModuleList` the whole time — `named_parameters()` could see it, it was
  just never summed) and `block_weight_list` (same blind spot as
  `adapter_list` — a plain python list of Parameters, invisible to
  `named_parameters()`). New `_bank_bytes()` hook mirrors this (excludes the
  live `cur_adapter`/`block_weight`, which is never capped — matches every
  other method's live-slot treatment).
- New `_bank_admit_candidate_bytes()`: cheap upfront estimate of this task's
  combined growth (specific-block adapter slice of `cur_adapter` +
  `block_weight` + full `cur_adapter` for the KD snapshot), used once in
  `after_task()` before `add_adapter_to_list()` runs.
- `backbone/vit_cllora.py::add_adapter_to_list(self, admit=True)`: new
  `admit` param. When `False`, skips growing all three structures AND skips
  `get_new_adapter_msa()` (which both reinitializes `cur_adapter`'s
  specific_pos slots and is the only thing that properly re-enables
  `requires_grad` on `cur_adapter` post-`freeze()` — `freeze()`'s own
  `self.cur_adapter[i].requires_grad = True` is a plain attribute assignment
  on an `nn.ModuleList`, a **pre-existing no-op** for actual gradient
  tracking, true with or without this feature). New
  `_rearm_cur_adapter_grad()` replicates only the requires_grad half, so
  `cur_adapter` keeps training continuously (never reinitialized) once
  frozen, matching every other method's post-cap-slot treatment.
- `backbone/vit_cllora.py::forward_general_cls()`: KD-teacher indexing
  changed from `self.old_adapter_list[t_idx-1]` to `self.old_adapter_list[-1]`
  — pre-freeze these are always the same entry (mathematically identical,
  confirmed by tracing `old_adapter_list`'s append timing against when this
  method runs), so a no-op when unset; post-freeze this keeps the KD
  regularizer running against the frozen snapshot forever (decision: never
  disabled, just stale), instead of indexing past the end.
- `utils/inc_net.py::OurNet.feature_dim` (property): `len(backbone.
  adapter_list)` instead of `self._cur_task` — no-op pre-freeze (the two are
  always numerically equal at every point this property is read; verified
  against `backbone/vit_cllora.py::forward_test()`'s own concatenated-feature
  width, which already self-limits the same way), keeps the classifier's
  input width matched to the backbone's actual post-freeze output width.
- `utils/inc_net.py::OurNet.update_fc()`: the old-weight row-copy changed
  from `fc.weight.data[:old_nb_classes, :-self.out_dim]` to `fc.weight.data[
  :old_nb_classes, :old_width]` — genuinely WRONG (not just imprecise) once
  `feature_dim` stops growing: `new_width == old_width` post-freeze, so
  `:-out_dim` would drop the last `out_dim` real columns and raise a shape
  mismatch on assignment instead. `:old_width` is mathematically identical
  to the original slice in the pre-freeze case (`new_width == old_width +
  out_dim` there) and correct in both.
- `utils/inc_net.py::OurNet.forward()` (test path): `cur_task=len(backbone.
  adapter_list)` instead of `cur_task=self._cur_task`, passed to `CosineLinear
  Feature.forward_diagonal()` (`backbone/linears.py`) — same no-op-pre-freeze
  substitution, keeps its per-chunk loop bound matched to `x_input`'s actual
  width.
- `models/cllora.py::replace_fc()`: loop bound and the `use_diagonal`
  "only recompute the just-finished task's own prototype" check both changed
  from `self._cur_task` to `n_banked = len(backbone.adapter_list)` — no-op
  pre-freeze; post-freeze, every task recomputes and overwrites the SAME
  trailing prototype column block (`backbone.forward_proto`'s `adapt_index ==
  len(adapter_list)` branch already routes to the live `cur_adapter`), so
  every post-cap task's real class-prototype signal still reaches the
  classifier instead of being silently discarded or written out of bounds.
  `get_A_B_Ahat`/`solve_similarity`/`solve_sim_reset`/`replace_fc_proxy`
  confirmed DEAD CODE (never called anywhere in CL-LoRA's own pipeline — only
  `models/ease.py` calls its own similarly-named methods) — left untouched.

### EASE (`models/ease.py`, `backbone/vit_ease.py`, `utils/inc_net.py`)
`Learner(BankCapMixin, BaseLearner)`; `_bank_cap_init(args)` called in
`__init__`. Structurally close to CL-LoRA (`OurNet`/`vit_cllora.py` were
clearly forked from `EaseNet`/`vit_ease.py`) but simpler — no general_pos/
specific_pos split, no KD distillation/`old_adapter_list`, `cur_adapter` is a
single full 12-block adapter reset every task via `get_new_adapter()`
(instead of CL-LoRA's partial specific_pos-only reset). No `persistent_state()`
gap existed here (unlike TUNA/CL-LoRA) — already accurately summed
`adapter_list` + `cur_adapter` + `fc`.
- `after_task()`: new admission check (`_bank_admit_candidate_bytes()` /
  `_bank_bytes()`) before `add_adapter_to_list()`, threaded through as
  `admit=`.
- `backbone/vit_ease.py::add_adapter_to_list(self, admit=True)`: new
  `admit` param. When `False`, skips growing `adapter_list` AND skips
  `get_new_adapter()` (which discards `cur_adapter` and builds a fresh,
  randomly-initialized replacement every task — EASE's per-task reset,
  unlike CL-LoRA's continuous `cur_adapter`). Explicitly replicates
  `get_new_adapter()`'s `requires_grad_(True)` re-arm (needed for the same
  reason as CL-LoRA: `freeze()`'s own `self.cur_adapter[i].requires_grad =
  True` is a plain attribute assignment on an `nn.ModuleList`, a
  pre-existing no-op). Simpler than CL-LoRA's equivalent fix — no
  general_pos/specific_pos split to preserve, so a single
  `self.cur_adapter.requires_grad_(True)` call suffices.
- `utils/inc_net.py::EaseNet.feature_dim` / `update_fc()` / `forward()`
  (test path's `cur_task` argument to `forward_reweight`): same three fixes
  as CL-LoRA's `OurNet` (`len(backbone.adapter_list)` substitution, the
  `:old_width` row-copy fix, `cur_task=len(adapter_list)`) — verified
  `backbone/linears.py::EaseCosineLinear.forward_reweight`'s `/cur_task`
  normalization can't hit division-by-zero from this substitution: at
  `cur_task=0` the `j != i` branch (the only one that divides) is never
  reached regardless of which real task this corresponds to, confirmed by
  tracing the loop bounds for both `use_init_ptm` settings.
- `models/ease.py::replace_fc()`: loop bound and `use_diagonal`'s
  "only recompute the just-finished task's own prototype" check both
  changed from `self._cur_task` to `n_banked = len(adapter_list)` — same
  fix as CL-LoRA's `replace_fc()`. The separate `use_exemplars` branch
  (lines below) was checked and left UNCHANGED — it already routes safely
  regardless of `self._cur_task`'s value: `forward_proto`'s `i <
  len(adapter_list)` check routes any out-of-range index to `cur_adapter`
  (not just the exact boundary value), and its `fc.weight` write uses
  negative indexing (`-out_dim:`), which is position-independent of
  `self._cur_task`'s absolute value.
- `models/ease.py::replace_fc()` tail: `solve_similarity()`/
  `solve_sim_reset()` (LIVE code here, unlike CL-LoRA's dead copies of the
  same methods — only reached when `NOT use_diagonal AND NOT
  use_exemplars`) now guarded by `if self._bank_frozen: return` —
  **disabled entirely once frozen**, per the explicit design decision,
  rather than fixing their internal `range(self._cur_task)` indexing.
  Confirmed sufficient: both methods only ever refine `fc.weight`'s B-columns
  on top of the diagonal blocks `replace_fc`'s main loop already writes
  correctly — `fc.weight` retains every pre-freeze value, so skipping this
  refinement step leaves the classifier at its last correctly-computed
  state.

### TUNA (`models/tuna.py`)
`Learner(BankCapMixin, BaseLearner)`; `_bank_cap_init(args)` called in
`__init__`. TUNA has two independently-growing structures sharing ONE
budget/latch: `adapter_list` and `cls_mean`/`cls_cov`.
- `persistent_state()`: now counts `cls_mean`/`cls_cov` bytes (previously
  uncounted — a real gap, ~2.36MB/class at `ca_storage_efficient_method=
  "covariance"`, ~3KB/class at `"variance"`). New `_bank_bytes()` hook
  (adapter_list + cls_mean/cls_cov, fc head exempt) mirrors this.
- New `_bank_cls_stat_bytes_per_class()`: cheap upfront per-class byte
  estimate (no forward pass), used to decide admission before
  `_compute_mean()` actually runs.
- `_train()`: admission checked at PER-TASK granularity (whole task's cls
  stats, or whole task's adapter slot — not per-class/per-parameter), in
  priority order: cls stats first, adapter second. Round-3 rationale: a
  denied class gets zero further `classifer_align` anti-drift correction
  ever (TUNA's only mechanism for older classes), while a denied adapter
  slot only costs one task's entropy-vote diversity. Both checks are
  computed upfront (cheap, no heavy work) so the *order of `_bank_admit`
  calls* — not the order of the expensive `adapter_update()`/`_compute_mean()`
  calls themselves, which are left in their original position — implements
  the priority. `adapter_update()` is now gated by `if adapter_admit:`.
- `_compute_mean(self, model, admit=True)`: new `admit` param; returns
  immediately (no-op) when `False`. When `bank_cap_mb` is unset, `admit` is
  always `True` — byte-identical to prior behavior.
- `classifer_align()`: new `n_ca_classes = min(self._total_classes,
  len(self.cls_mean))` clamp, replacing `self._total_classes` in all three
  loops (distribution-building, sample-drawing, batch-iteration) that
  previously assumed `cls_mean`/`cls_cov` had an entry for every class up to
  `_total_classes` — would `KeyError` on a denied class otherwise. No-op
  when unset (cls_mean always has exactly `_total_classes` entries then).
- `orth_loss()`: bound changed from `range(self._cur_task)` to
  `range(min(self._cur_task, len(adapter_list)))` — would `IndexError` past
  a frozen bank otherwise. No-op when unset.
- 4 inference call sites (`_deployed_forward` x2, `_eval_cnn` x2): loop
  bound and "general adapter" id changed from `self._cur_task + 1` to
  `len(net.backbone.adapter_list)` (equivalently `n_banked`). Pre-freeze
  these are always equal (`adapter_update()` appends every task), so this is
  a no-op when unset; post-freeze it naturally caps the entropy-vote
  ensemble at the K banked slots + 1 live `cur_adapter`, and makes
  `forward_features`'s existing `adapter_id == len(adapter_list)` branch a
  tautology instead of needing a separate clamp.
- `cur_adapter` itself needs no training-side change — TUNA already trains
  it continuously every task by design (unlike EASE's per-task reset), so
  the "never reinitialize the post-cap slot" requirement is satisfied for
  free.
- Untouched, left running unconditionally post-freeze: `merge()` (its
  output, `merged_adapter`, is dead code in the current call pattern —
  confirmed no call site ever passes `adapter_id > len(adapter_list)` — so
  freezing or not freezing it has zero observable effect; not in scope to
  fix that pre-existing dead branch).

---

## Suspicious / flagged during implementation
*(anything that comes up during actual coding that wasn't caught by the three
design-review passes goes here, with a clear description and current
status.)*

- **CL-LoRA: a 6th coupled correctness site, missed by round 3.**
  `models/cllora.py::_init_train()`'s KD-phase gradient-reweighting block
  (`for j in range(len(...general_pos))`) directly indexes
  `self._network.backbone.old_adapter_list[self._cur_task-1][pos][jj]` —
  genuinely the same class of bug as `forward_general_cls`'s `old_adapter_list
  [t_idx-1]` (round 3's site #5), just a separate call site round 3's report
  didn't name. Fixed with the identical `[-1]` substitution. Status: FIXED.
  Worth noting for future per-method audits — a `grep` for every direct index
  into a bank-cap-relevant list, not just the call sites named in the design
  doc, is what caught this; the round-3 agents' reports should probably be
  re-verified against a full-file grep rather than trusted as exhaustive.

- **RainbowPrompt: dead-code near-miss.** `models/rainbowprompt.py` defines
  `_forward_task` TWICE in the same class body (an oracle-flavored version,
  and a streaming-flavored one further down using `_stream_task_to_chunk`).
  Python silently keeps only the LAST definition — the oracle-flavored one is
  dead code at runtime in both oracle and streaming modes. I initially wrote
  the bank-cap TIL remap (`_rp_task_to_slot`) into the dead copy before
  grepping for a second `def _forward_task` and finding the live one further
  down. Caught before considering the method done by re-checking which copy
  actually executes. Fixed: the remap now lives in the live copy, chained
  after the existing `_stream_task_to_chunk` remap (mutually exclusive in
  practice — `_stream_task_to_chunk` only exists under streaming,
  `_rp_pinned_slot` only ever gets set from oracle-path methods). The dead
  copy is left in place, unmodified, with a comment explaining why removing
  it is out of scope for this feature.

- **A class of "empty bank" crash bugs, caught while preparing verification
  smoke tests, not by the three design-review passes.** All three design
  rounds implicitly assumed a method's bank always ends up with AT LEAST one
  entry once past task 0 — true "for free" in O-LoRA/RainbowPrompt (slot/row
  0 is always preallocated at construction, never subject to the admission
  check) but NOT true for TUNA/CL-LoRA/TreeLoRA, whose very first admission
  (task 0's own contribution) goes through the same check as every later
  one. With an extreme `bank_cap_mb` smaller than even one task's minimum
  footprint, these three could end a run with a bank that is EMPTY FOREVER,
  and several consumers assumed "at least one entry" implicitly:
  - `models/tuna.py::classifer_align()`: `n_ca_classes==0` -> `torch.cat([])`
    on an empty sampled-data list. Fixed: early-return when `n_ca_classes==0`
    (nothing to align against).
  - `models/tuna.py::_deployed_forward()`/`_eval_cnn()`: `n_banked==0` at
    `_cur_task>0` -> `torch.stack([])` on the empty entropy-vote lists.
    Fixed: both fall back to the same single-adapter path task 0 itself
    already uses (`_eval_cnn1`/its inlined equivalent).
  - `models/cllora.py::_init_train()`'s KD phase: `old_adapter_list` empty ->
    `forward_kd`/`forward_general_cls`'s `old_adapter_list[-1]` indexes an
    empty `nn.ModuleList`. Fixed: the whole KD phase is skipped for a step
    when `old_adapter_list` is empty (no teacher to distill from).
  - `models/treelora.py::_train()`'s `_tree_regularizer`: all-`None`
    `all_accumulate_grads` -> `tree_search`'s `torch.stack([])` on the empty
    filtered list. Fixed: skip the regularizer for a step when no snapshot
    has ever been banked.
  All four fixes are no-ops when `bank_cap_mb` is unset (the "always at
  least one entry" invariant holds unconditionally there) and no-ops for any
  cap large enough to admit task 0's own contribution — they only change
  behavior in the genuinely degenerate case of a cap set below the minimum
  possible footprint, where the alternative was a crash. EASE was checked
  and found to already be safe in this scenario without any change (traced
  every consumer: `forward_test`'s loop and `feature_dim`'s width both
  naturally collapse to "just the live block" at `n_banked==0`, `replace_fc`
  routes correctly via `forward_proto`'s existing `i < len(adapter_list)`
  check, and `solve_similarity`/`solve_sim_reset` are already gated off by
  `self._bank_frozen`). Status at the time: FIXED (4 sites), VERIFIED SAFE
  (EASE), believed no fix needed for O-LoRA/RainbowPrompt/InfLoRA.

- **InfLoRA excluded from actual bank-cap experiments (2026-08-14, explicit
  user decision), even though the implementation is correct and stays in the
  codebase.** InfLoRA's persistent footprint is dominated by a large FIXED
  floor (`frozen_delta_q/v`, ~54MB, always allocated from construction
  regardless of budget — see the entry below) rather than a per-task
  increment comparable in scale to the floor. This makes the cap
  effectively binary for InfLoRA in the memory range the sensitivity study
  actually cares about: any budget below ~54MB freezes DualGPM from task 0
  (no gradual degradation to observe), and any budget comfortably above it
  never binds at all — there's no useful middle range where the cap
  meaningfully trades off against task count the way it does for the other
  six methods (whose growth is smaller-per-task and starts near 0). The
  code (`_bank_bytes()`, the coarse `_update_dualgpm` gate, the
  `_init_lora_A` empty-projector fallback below) is left in place — it's
  correct, harmless when `bank_cap_mb` is unset, and available if ever
  needed — but InfLoRA should be EXCLUDED from any bank-cap experiment
  campaign configs and from further bank-cap verification testing going
  forward. Focus remains on the other six methods (O-LoRA, TreeLoRA,
  RainbowPrompt, CL-LoRA, EASE, TUNA), where the cap's tradeoff is
  meaningful. The queued `inflora_cap_tiny` retry (to confirm the
  `_init_lora_A` fix below) was cancelled per this decision — the fix is
  still correct and stays in the code, just no longer a priority to
  re-verify by burning further GPU time on InfLoRA specifically.

- **InfLoRA: the "safe by construction" claim above was WRONG — caught by
  the verification smoke suite itself (`inflora_cap_tiny` failed with
  `IndexError: list index out of range` at task 1's `_init_lora_A`), not by
  static reasoning.** The reasoning that `_bank_bytes()` is "provably 0 at
  task 0" was never actually verified against the code — `frozen_delta_q`/
  `frozen_delta_v` are `register_buffer`s allocated at FULL `[dim,dim]` size
  in `backbone/vit_lora.py`'s `Attention_LoRA.__init__` (confirmed by
  reading it directly), not lazily grown by folding. `_bank_bytes()` sums
  their raw byte size unconditionally, so it reports ~54MB from the moment
  the model is constructed — task 0, before any folding has ever happened —
  not 0. An extreme cap (smaller than ~54MB) therefore refuses admission
  immediately at task 0, `_update_dualgpm` never runs even once, and
  `feature_list`/`feature_mat`/`project_type` stay permanently empty.
  `models/inflora.py::_init_lora_A()`'s `else` branch (`_cur_task > 0`)
  unconditionally indexed `self.feature_mat[kk]` for all 12 blocks, assuming
  it was always populated by then — crashes on the very next task. Fixed:
  the branch condition is now `self._cur_task == 0 or kk >= len(self.
  feature_mat)` — "no projector banked for this block yet" is treated
  identically to task 0 (plain SVD of the raw covariance, no projection). A
  no-op when `bank_cap_mb` is unset. The design choice to COUNT
  `frozen_delta` toward `_bank_bytes()` at all (rather than exempting it as
  a fixed floor) was correct and intentional — matches the project-wide
  "count the full non-head footprint, how it gets spent isn't our concern"
  convention, same as InfLoRA counting its own fixed frozen_delta and
  RainbowPrompt counting its fixed evolve_sublayers — only the "task 0 must
  be free" assumption was wrong. Practical implication for the O-LoRA/
  RainbowPrompt "safe by construction" claims above: those were re-checked
  and confirmed still correct (slot/row 0 is genuinely preallocated with
  zero admission-check involvement there, not merely assumed), but this
  is now confirmed by re-reading the code, not left as an unverified
  parallel to InfLoRA's wrong claim. Status: FIXED, and the "no fix needed"
  verdict for InfLoRA specifically is retracted and replaced by this entry.
