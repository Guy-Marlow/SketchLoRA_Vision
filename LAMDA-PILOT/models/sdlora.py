"""SD-LoRA -- "Scalable Decoupled Low-Rank Adaptation for Class Incremental
Learning" (ICLR 2025 Oral, Wu et al.), ported 2026-10-27 per user request.
See the research report this implementation was built from:
scratchpad/analysis_sdlora.md (verified directly against SD-LoRA-Ref/models/
sdlora.py and backbone/lora.py's actual code before any of this file was
written -- including grepping the reference repo to confirm its own
orthogonality-loss term is computed and then discarded, and its own
optimizer/scheduler config keys are dead/never read).

ALGORITHM, as actually implemented here (plain task-local cross-entropy,
nothing else -- the reference's own orthogonality term is never exercised in
its results, so it is intentionally NOT ported; see the research report §1
and §6.3 for the direct code evidence). Per task t:
  - a brand-new (A_t, B_t) LoRA pair is allocated for every (block, q/v)
    module (standard per-task adapter growth, reusing models/lora.py's base
    incremental_train/_train_adapter unmodified -- this is the one part of
    SD-LoRA that is NOT special, same growth models/olora.py also uses).
  - ONE new learnable scalar magnitude m_t, shared across EVERY block and
    both q/v projections (not per-module) -- the method's "scalable"
    claim. Lives on backbone/vit_sdlora.py's VisionTransformerSDLoRA as
    `task_scales`, NOT per-attention-module (see that file's docstring for
    why: a per-block copy would silently multiply the optimizer's effective
    learning rate on this parameter by 12x).
  - every PAST task's magnitude m_s (s<t) is re-activated (freeze_to_task,
    overridden in vit_sdlora.py) and jointly re-optimized alongside m_t --
    this project's port WARM-STARTS each m_s from its own previously-learned
    value rather than resetting to 0.8 every task, an explicit, documented
    deviation from SD-LoRA-Ref's as-released behavior (research report open
    question #1: SD-LoRA-Ref's own save_wrap_param, which would persist a
    trained scale to disk, is never called anywhere in that repo, so the
    reference's own released code has no genuine cross-task magnitude
    persistence despite that being the method's stated point).

Forward (both training and CIL eval, same formula, backbone/vit_sdlora.py's
Attention_LoRA_SDLoRA._lora_delta):
  delta(x) = m_t * B_t(A_t(x))
           + sum_{s<t} m_s * B_s(A_s(x)) / (||B_s||_F * ||A_s||_F)
i.e. every FROZEN past task's raw contribution is Frobenius-normalized before
being rescaled by its own global scalar; the CURRENT task's term is left
unnormalized. This needs its own backbone subclass (the base
Attention_LoRA._lora_delta only has a single fixed scaling constant for the
whole sum) -- see backbone/vit_sdlora.py.

PERSISTENT MEMORY: dominated by the per-task adapter bank -- NOT smaller
than O-LoRA's own per-task-adapter memory (the "decoupled" idea buys a
different stability/plasticity mechanism, not a memory reduction, at the
direction level -- research report §2). Formula: T*2*r*d*M*L*4 bytes for
the direction bank (M=2 modules/block, L=12 blocks) + T*4 bytes for the
scale scalars (negligible). persistent_state() below extends models/
lora.py's base implementation (which already sums the direction bank
correctly via the inherited attn_modules() passthrough) with the scale
scalars as their own breakdown key.

HYPERPARAMETERS: this project's standing ViT-B/16 LoRA convention (AdamW,
cosine-annealed init_lr=0.001, lora_rank=10, batch_size=48, tuned_epoch=20)
is used throughout, NOT SD-LoRA-Ref's own released config (which, per the
research report, is also internally inconsistent -- its own JSON configs
claim Adam/cosine/steplr schedulers but sdlora.py's actual _train()
hardcodes plain SGD+momentum=0.9 and MultiStepLR regardless, and never
reads lora_rank from config at all, hardcoding r=10). No regularizer weight
exists for this method (the dead orthogonality term aside).

SCOPE: oracle (task-boundary) CIL path only. SD-LoRA-Ref never defines a
working task-routed (TIL, single-slot) forward either -- this port's own
invented definition for merge=False (this task's own term alone,
unnormalized) is a project-local choice with no reference-code backing, see
backbone/vit_sdlora.py.

SD-LoRA-RR (rank-reduction variant, 2026-10-27 port): confirmed as a real,
concrete codepath in SD-LoRA-Ref (models/sdlora.py::update_network, labeled
"SD-LoRA-RR" in a comment, but DEAD -- fully commented out, `rank=10` is
hardcoded immediately after it regardless): new adapters allocated for
task>=4 use rank 8, task>=8 use rank 6, else rank 10 (lora_rank). Confirmed
there (by reading backbone/lora.py::LoRA_ViT_timm.__init__) that `r` governs
ONLY the fresh adapter being allocated for the upcoming task -- already-
trained past tasks' weights are loaded from disk at their own original
shape, never touched by a later rank reduction. Same semantics here, exposed
via `sdlora_rank_schedule` (a list of [task_threshold, rank] pairs, e.g.
[[4,8],[8,6]] reproduces SD-LoRA-Ref's own literal values; default None =
constant lora_rank every task, the unmodified base method). Needs its own
backbone/vit_sdlora.py::Attention_LoRA_SDLoRA.add_task_slot(rank=...)
override -- the base Attention_LoRA.add_task_slot() always uses the single
construction-time self.lora_rank, with no per-call override. persistent_
state() below needs NO extra code for this variant: it already sums each
slot's REAL allocated numel()/element_size(), so a smaller later slot is
reflected automatically.

SD-LoRA's paper also proposes a knowledge-distillation-based adapter-
consolidation variant (2026-10-27 user question). Searched SD-LoRA-Ref
directly (models/sdlora.py, backbone/lora.py, and a repo-wide grep for
distill/KD/teacher/merge/combine/consolidate/fuse/aggregat) and found NO
trace of it anywhere -- only the rank-reduction variant exists as actual
code in the reference repo. NOT implemented here; would need the paper
text itself (not available locally) or a description of the mechanism to
port correctly rather than guess at it.
"""
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.lora import Learner as LoRALearner, num_workers
from utils.rank_diagnostics import log_bank_effective_rank


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # SD-LoRA's forward always accumulates every task's contribution
        # (0..task), both during training and at inference -- there is no
        # separate "single slot alone" mode in the method. Force this rather
        # than relying on every config remembering to set both flags.
        self.train_merge = True
        self._network.merge = True
        # models/lora.py's base Learner never stores lora_rank on self (only
        # the backbone factory reads it directly from args) -- needed here
        # for _rank_for_task's "else" case.
        self.lora_rank = args.get("lora_rank", 10)
        # SD-LoRA-RR: optional [[task_threshold, rank], ...] schedule, sorted
        # ascending by threshold. None (default) = constant lora_rank every
        # task (base method, unmodified).
        self.rank_schedule = args.get("sdlora_rank_schedule", None)
        if self.rank_schedule is not None:
            self.rank_schedule = sorted(self.rank_schedule, key=lambda pair: pair[0])
        # 2026-11-09 user request: same opt-in summed-bank effective-rank
        # diagnostic as O-LoRA/LoRA-Subtraction (utils/rank_diagnostics.py),
        # see _log_bank_effective_rank below for why this needs its OWN
        # delta construction rather than reusing theirs.
        self.log_bank_rank = bool(args.get("log_bank_rank", False))
        self.bank_rank_energy_target = args.get("bank_rank_energy_target", 0.99)

    def _rank_for_task(self, cur_task):
        if not self.rank_schedule:
            return self.lora_rank
        rank = self.lora_rank
        for threshold, r in self.rank_schedule:
            if cur_task >= threshold:
                rank = r
        return rank

    def incremental_train(self, data_manager):
        """Byte-for-byte copy of models/lora.py's base incremental_train,
        EXCEPT the add_task_slot() call below passes this task's rank from
        the SD-LoRA-RR schedule -- the base version's own add_task_slot()
        call takes no rank argument at all, so this cannot be done via a
        smaller override. See this file's module docstring for the
        rank-reduction variant's full design."""
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = self._train_adapter()
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        rank = self._rank_for_task(self._cur_task)
        if self._train_adapter() >= self._network.backbone.n_tasks:
            self._network.add_task_slot(rank=rank)
            if self.rank_schedule is not None:
                logging.info("[SD-LoRA-RR] task {}: new adapter slot allocated at rank={}".format(
                    self._cur_task, rank))

        self._network.freeze_to_task(self._train_adapter())
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._log_trainable()

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader):
        super()._train(train_loader)
        net = self._network.module if hasattr(self._network, "module") else self._network
        scales = [round(s.item(), 4) for s in net.backbone.task_scales]
        logging.info("[SD-LoRA] task {} done. task_scales so far (m_0..m_{}): {}".format(
            self._cur_task, self._cur_task, scales))
        self._log_bank_effective_rank()

    @torch.no_grad()
    def _log_bank_effective_rank(self):
        """2026-11-09 user request -- same diagnostic as O-LoRA/LoRA-
        Subtraction (utils/rank_diagnostics.py), but SD-LoRA's merge is NOT
        a plain sum of B_s@A_s, so it cannot reuse their B_cat@A_cat
        construction. backbone/vit_sdlora.py::Attention_LoRA_SDLoRA.
        _lora_delta (confirmed directly against SD-LoRA-Ref/backbone/
        lora.py's own forward, lines 182-191 -- scaling_factor_prev[i] is a
        plain ParameterWrapper, i.e. x * scale, nothing more) applies:

            delta = m_t * B_t@A_t
                  + sum_{s<t} m_s * (B_s@A_s) / (||B_s||_F * ||A_s||_F)

        i.e. every past task's raw contribution is Frobenius-normalized by
        its OWN factors before being rescaled by its shared global scalar,
        and the current task's term is left unnormalized -- reusing a plain
        B_cat@A_cat here would silently measure the wrong matrix (the UN-
        normalized sum SD-LoRA's forward never actually applies). No-op
        unless log_bank_rank is set."""
        if not self.log_bank_rank:
            return
        net = self._network.module if hasattr(self._network, "module") else self._network
        t = self._train_adapter()
        scales = net.backbone.task_scales
        deltas = {}
        for li, attn in enumerate(net.attn_modules()):
            for proj, (A_list, B_list) in (("q", (attn.lora_A_q, attn.lora_B_q)),
                                            ("v", (attn.lora_A_v, attn.lora_B_v))):
                delta = scales[t] * (B_list[t].weight @ A_list[t].weight)
                for s in range(t):
                    B_s, A_s = B_list[s].weight, A_list[s].weight
                    norm = torch.linalg.norm(B_s) * torch.linalg.norm(A_s)
                    delta = delta + scales[s] * (B_s @ A_s) / norm
                deltas[(li, proj)] = delta
        log_bank_effective_rank("SD-LoRA", self._cur_task, deltas, self.bank_rank_energy_target)

    # persistent state: extends models/lora.py's base (direction bank: every
    # allocated task's A/B, q/v, all blocks -- already correct via the
    # inherited attn_modules() passthrough, see backbone/vit_sdlora.py's
    # docstring) with the per-task magnitude scalars as their own breakdown
    # key -- the exact mechanism the method is named after, worth surfacing
    # on its own rather than folded into "lora_slots".
    def persistent_state(self):
        base = super().persistent_state()
        net = self._network.module if hasattr(self._network, "module") else self._network
        scale_params = sum(p.numel() for p in net.backbone.task_scales)
        scale_bytes = sum(p.numel() * p.element_size() for p in net.backbone.task_scales)
        base["params"] += scale_params
        base["bytes"] += scale_bytes
        base["breakdown"]["scale_params"] = scale_params
        if self.rank_schedule is not None:
            # SD-LoRA-RR diagnostic: each slot's actual allocated rank, so a
            # shrinking schedule's real memory effect is directly visible
            # rather than only inferable from the aggregate byte count above.
            attn0 = next(iter(net.attn_modules()))
            base["breakdown"]["slot_ranks"] = [m.out_features for m in attn0.lora_A_q]
        return base
