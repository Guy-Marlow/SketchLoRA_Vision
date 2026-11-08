"""CL-LoRA-Sketch (2026-11-04): a variant of CL-LoRA (backbone/vit_cllora.py)
that replaces its specific-pos (deep-layer) per-task frozen adapter bank with
a SketchLoRA-style fixed-size, continuously-folded sketch -- see the
conversation record for the full design discussion. Fully separate file, per
this project's "new experimental mechanism = isolated codepath" convention
(models/inflora_sketch.py's own stated rule) -- not a modification of
backbone/vit_cllora.py.

WHAT STAYS THE SAME AS CL-LoRA
-------------------------------
general_pos (shallow) layers: completely unchanged. One shared Adapter_lora
per branch (Q/V, msa=[1,0,1]), trained continuously across every task, its
down-projection (`lora_B`, CL-LoRA's own reversed A/B naming) frozen to a
random-orthogonal basis at creation, regularized against catastrophic drift
by a KD loss (forward_general_cls, lives in the model file) against a frozen
snapshot of itself from the previous task.

The block-weight / orthogonality mechanism itself: COMPLETELY UNCHANGED.
Still one nn.Parameter of shape (3, len(specfic_pos)) per the ORIGINAL
Q/K/V-branch convention (not narrowed to Q/V only, even though K is never
read in the forward pass when msa[1]==0) -- confirmed-intentional design
choice (2026-11-04 conversation): the unused K row costs nothing (CE loss
can't see it, since it's never read in forward) and gives the orthogonality
penalty extra free dimensionality to spread tasks' vectors apart in. Still
drawn uniform(.5, 1.5) fresh per task, still trained jointly with everything
else via the SAME optimizer (no separate training phase), still frozen the
instant its task ends. compute_orthogonality_loss itself is UNCHANGED and
lives in models/cllora_sketch.py (ported verbatim from models/cllora.py).
Applied with its RAW TRAINED MAGNITUDE, unnormalized -- matching exactly how
baseline CL-LoRA applies it in Attention_lora.forward (`q += block_weight[0]
* adapt_x`), per explicit 2026-11-04 user decision: don't deviate from how
baseline does it.

WHAT CHANGES: specific-pos layers
----------------------------------
Baseline CL-LoRA keeps ONE FULL FROZEN Adapter_lora PER TASK per specific-pos
layer+branch (backbone.adapter_list), requiring forward_test to run a
SEPARATE full forward pass per past task and the classifier to block-
diagonal-match each task's classes only against its own feature chunk
(CosineLinearFeature.forward_diagonal).

This variant instead gives each specific-pos layer a fixed 2-slot SketchQV
pair (SKETCH=folded history, RESIDUAL=this task's live adapter) per active
branch (Q, V). At every task boundary, RESIDUAL's trained weights are scaled
by that task's own trained block-weight scalar (exactly the scale baseline
CL-LoRA would have applied to this adapter's output every forward pass this
task -- applying it directly to the weight matrix once, now that training is
done, is mathematically identical to scaling the activation every step and
is simpler to fold correctly), concatenated against SKETCH's existing
(already-scaled-from-prior-folds) factors, and re-expressed via the exact
QR-SVD merge (see SketchQV.fold's docstring) at a fixed rank (`svd_rank`).
RESIDUAL is then replaced by a fresh Adapter_lora for the next task. No
per-task bank growth, ever -- the whole point.

KEY SIMPLIFICATION this enables, not present in baseline: a freshly
constructed Adapter_lora's up-projection (`lora_A`) is zero-initialized by
construction (see Adapter_lora.__init__, `init_option="lora"`), so SKETCH
starts out contributing exactly zero. This means NO special-casing is needed
for "the very first fold" (unlike models/inflora_sketch.py's own
_sketch_populated flag/branch) -- running the identical QR-SVD merge
unconditionally, every fold including the first, already produces the
mathematically correct result (an all-zero SKETCH correctly contributes
nothing, recovered automatically via SVD's own rank truncation), provided
`svd_rank <= ffn_num` (the per-task adapter's own rank) so no real signal is
discarded on that first fold.

CLASSIFIER CHOICE (per explicit 2026-11-04 user decision -- "cosine match is
fine"): since specific-pos no longer produces a growing per-task feature
list, there is exactly ONE feature representation total (general-pos's live
shared adapters + specific-pos's live SKETCH+RESIDUAL sum), not N. The
classifier (models/cllora_sketch.py + utils/inc_net.py::OurNetSketch) uses a
single plain cosine match (CosineLinearFeature.forward, which already exists
unmodified in backbone/linears.py) instead of forward_diagonal's block-
diagonal per-task scoring -- there's only one block to match against, so the
block-diagonal machinery is simply inapplicable, not approximated away.
`use_init_ptm` (the optional extra raw-backbone feature block) is dropped
entirely for this variant for the same reason -- it would need its own
second classifier block, out of scope here.

NOT CHANGED: use_distillation / forward_general_cls / old_adapter_list (the
general-pos KD snapshot) -- lives in the model file, reused verbatim.

THREE VARIANTS (2026-11-04 extension, same file, selected by two config
flags, not three separate files):
  - `use_block_weight_scaling` (bool, default True): the mechanism described
    above (block-weight scalar, trained + orthogonalized every step exactly
    as baseline CL-LoRA, applied to RESIDUAL's output during training and
    baked into RESIDUAL's lora_A before folding). When False, this entire
    mechanism is ABSENT, not merely inert: no block_weight parameter exists
    at all, RESIDUAL folds into SKETCH with an implicit weight of 1.0 (a
    plain, unweighted SketchLoRA-style fold), and no block-weight
    orthogonality loss term is added.
  - `orth_a_side` (bool, default False): an ADDITIONAL, independent
    regularizer -- the orthogonality penalty "already established as part
    of SketchLoRA" (models/sketchlora_align.py's `_orth_loss`, itself
    O-LoRA's `_orth_and_l2` orth term verbatim, applied here between
    RESIDUAL and SKETCH instead of between a growing list of frozen slots).
    Targets the DOWN-projection / input-facing matrix -- CL-LoRA's own
    `lora_B` attribute name, confirmed via the 2026-11-04 conversation to be
    the functional equivalent of O-LoRA/SketchLoRA's "A" (NOT CL-LoRA's own
    `lora_A`, which is the up/output-facing matrix and would be a different,
    less-motivated ablation). Weight: `sketchlora_align_weight` (default
    0.5, same key/default as sketchlora_align.py's own "orth" mode). Lives
    in `sketch_orth_loss()` below; added into the training loss in the
    model file. Independent of block-weight scaling by construction: that
    mechanism only ever touches `lora_A` (up-proj); this one only ever
    touches `lora_B` (down-proj) -- no interaction between them either way.

This gives three requested combinations (a fourth, neither flag set, is
also mechanically available but untested/not requested):
  1. use_block_weight_scaling=True,  orth_a_side=False  (the original build)
  2. use_block_weight_scaling=True,  orth_a_side=True   (both mechanisms)
  3. use_block_weight_scaling=False, orth_a_side=True   (orth only)
"""
import copy
import logging
import math

import timm
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import PatchEmbed

from backbone.vit_cllora import Adapter_lora


class SketchQV(nn.Module):
    """Fixed 2-slot (SKETCH, RESIDUAL) factored LoRA pair for one specific-pos
    layer's Q and V branches. Each slot is a real backbone/vit_cllora.py
    Adapter_lora instance (same bottleneck module CL-LoRA already uses for
    its specific-pos adapters) -- reused as-is rather than inventing a new
    module shape, so the merge math below operates on exactly the same
    (down-proj, up-proj) factor pair CL-LoRA's own adapters already are.
    """

    def __init__(self, config, bottleneck):
        super().__init__()
        self.bottleneck = bottleneck
        self.sketch_q = self._new_adapter(config, bottleneck)
        self.residual_q = self._new_adapter(config, bottleneck)
        self.sketch_v = self._new_adapter(config, bottleneck)
        self.residual_v = self._new_adapter(config, bottleneck)

    @staticmethod
    def _new_adapter(config, bottleneck):
        return Adapter_lora(config, dropout=0.0, bottleneck=bottleneck,
                             init_option=config.ffn_adapter_init_option,
                             adapter_scalar=config.ffn_adapter_scalar,
                             adapter_layernorm_option=config.ffn_adapter_layernorm_option)

    def forward_q(self, x, block_weight_q=None):
        bw = 1.0 if block_weight_q is None else block_weight_q
        return self.sketch_q(x) + bw * self.residual_q(x)

    def forward_v(self, x, block_weight_v=None):
        bw = 1.0 if block_weight_v is None else block_weight_v
        return self.sketch_v(x) + bw * self.residual_v(x)

    def orth_loss(self):
        """"orth_a_side" regularizer -- O-LoRA's `_orth_and_l2` orth term
        verbatim (same formula models/sketchlora_align.py's own `_orth_loss`
        already uses), applied between RESIDUAL's and SKETCH's DOWN-
        projection (`lora_B`, CL-LoRA's own naming -- confirmed the
        functional equivalent of O-LoRA/SketchLoRA's "A"). Summed over Q
        and V. SKETCH's factor is detached -- this penalty only ever moves
        the live RESIDUAL, never the frozen history."""
        orth = 0.0
        for sketch_mod, residual_mod in ((self.sketch_q, self.residual_q), (self.sketch_v, self.residual_v)):
            A_new = residual_mod.lora_B.weight                  # [down_size, n_embd], live
            A_sketch = sketch_mod.lora_B.weight.detach()         # [down_size, n_embd], frozen
            orth = orth + torch.abs(A_new @ A_sketch.t()).sum()
        return orth

    @torch.no_grad()
    def fold(self, block_weight_q, block_weight_v, svd_rank, config):
        """Scale each branch's RESIDUAL by its own just-trained block-weight
        scalar (raw magnitude, unnormalized -- see this module's own
        docstring), fold into SKETCH via the exact QR-SVD merge (identical
        algorithm to models/inflora_sketch.py::_sketch_fold, just applied to
        Adapter_lora's lora_A/lora_B naming instead of vit_lora.py's A/B),
        then replace RESIDUAL with a fresh Adapter_lora. Returns a dict of
        retained-energy diagnostics (one scalar per branch), same
        instrumentation philosophy as SketchDiag/InfLoRA-Sketch.
        """
        retained = {}
        for branch, sketch_mod, residual_mod, bw in (
                ("q", self.sketch_q, self.residual_q, block_weight_q),
                ("v", self.sketch_v, self.residual_v, block_weight_v)):
            # CL-LoRA's own reversed naming: lora_A is the UP-projection
            # ([n_embd, down_size]), lora_B is the DOWN-projection
            # ([down_size, n_embd]); forward is lora_A(lora_B(x)), i.e.
            # delta_W = lora_A.weight @ lora_B.weight. Scaling lora_A (one
            # factor only) by the scalar reproduces exactly what baseline
            # CL-LoRA's `block_weight * adapt_x` would have computed every
            # forward pass this task, now baked into the weight once.
            up_sk, down_sk = sketch_mod.lora_A.weight, sketch_mod.lora_B.weight
            up_res = residual_mod.lora_A.weight.float() * float(bw)
            down_res = residual_mod.lora_B.weight

            delta_true = up_sk.float() @ down_sk.float() + up_res @ down_res.float()
            up_cat = torch.cat([up_sk.float(), up_res], dim=1)       # [n_embd, r_sk + r_res]
            down_cat = torch.cat([down_sk.float(), down_res.float()], dim=0)  # [r_sk + r_res, n_embd]

            Q_up, R_up = torch.linalg.qr(up_cat)
            Q_down, R_down = torch.linalg.qr(down_cat.t())
            M_small = R_up @ R_down.t()
            U_small, S, Vh_small = torch.linalg.svd(M_small)
            U = Q_up @ U_small
            Vh = Vh_small @ Q_down.t()

            r = min(svd_rank, S.shape[0])
            sqrt_s = S[:r].sqrt()
            up_new = U[:, :r] * sqrt_s
            down_new = sqrt_s.unsqueeze(1) * Vh[:r, :]

            recon = up_new @ down_new
            fro_true = torch.linalg.norm(delta_true).clamp_min(1e-12)
            retained[branch] = (1.0 - (torch.linalg.norm(delta_true - recon) / fro_true) ** 2).item()

            sketch_mod.lora_A.weight.data.copy_(up_new.to(sketch_mod.lora_A.weight.dtype))
            sketch_mod.lora_B.weight.data.copy_(down_new.to(sketch_mod.lora_B.weight.dtype))

        self.residual_q = self._new_adapter(config, self.bottleneck).to(self.sketch_q.lora_A.weight.device)
        self.residual_v = self._new_adapter(config, self.bottleneck).to(self.sketch_v.lora_A.weight.device)
        return retained


class Attention_lora(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., msa=(0, 0, 0)):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.msa = msa

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x, adapt=None, sketch=None, block_weight=None):
        B, N, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if adapt is not None:
            # general-pos path: unchanged from vit_cllora.py -- always full
            # strength, no block-weight multiply (baseline never applies one
            # here either: use_block_weight is gated to specfic_pos only).
            if self.msa[0] == 1:
                q = q + adapt[0](x)
            if self.msa[2] == 1:
                v = v + adapt[2](x)
        elif sketch is not None:
            bw_q, bw_v = block_weight if block_weight is not None else (None, None)
            if self.msa[0] == 1:
                q = q + sketch.forward_q(x, bw_q)
            if self.msa[2] == 1:
                v = v + sketch.forward_v(x, bw_v)

        k = self._shape(k, -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(v, -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None):
        super().__init__()
        from timm.models.layers import DropPath
        self.norm1 = norm_layer(dim)
        self.attn = Attention_lora(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop, msa=config.msa)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x, adapt=None, sketch=None, block_weight=None):
        x = x + self.drop_path(self.attn(self.norm1(x), adapt=adapt, sketch=sketch, block_weight=block_weight))
        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))
        x = residual + x
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 global_pool=False, tuning_config=None):
        super().__init__()
        self.tuning_config = tuning_config
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.use_distillation = tuning_config.use_distillation
        self.msa = tuning_config.msa
        assert self.msa[1] == 0, \
            "CL-LoRA-Sketch does not implement K-branch sketching (K is never adapted in every " \
            "config this project uses, msa[1]==0) -- got msa={}".format(self.msa)
        self.general_pos = tuning_config.general_pos
        self.specfic_pos = tuning_config.specfic_pos
        # adapt_pos is GENERAL-POS ONLY here (unlike vit_cllora.py, where it's
        # general+specific) -- cur_adapter/old_adapter_list/KD now only ever
        # concern general_pos; specific_pos is handled entirely through the
        # separate sketch_list below.
        self.adapt_pos = sorted(self.general_pos)
        self.svd_rank = tuning_config.svd_rank
        self.use_block_weight_scaling = getattr(tuning_config, "use_block_weight_scaling", True)
        self.orth_a_side = getattr(tuning_config, "orth_a_side", False)

        if self.use_distillation:
            self.old_adapter_list = nn.ModuleList()

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                        embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer, config=tuning_config)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.global_pool = global_pool
        self.head = nn.Identity()

        self.config = tuning_config
        self._device = tuning_config._device

        self.cur_adapter = nn.ModuleList()
        self.get_new_adapter_initial_msa()

        self.sketch_list = nn.ModuleList(
            [SketchQV(tuning_config, tuning_config.ffn_num) for _ in self.specfic_pos])
        # block_weight/block_weight_list ABSENT (not merely inert) when
        # use_block_weight_scaling=False -- see this file's own module
        # docstring, "THREE VARIANTS".
        self.block_weight_list = []   # grows one frozen (3, n_specific) tensor per admitted task
        if self.use_block_weight_scaling:
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, .5, 1.5)
        else:
            self.block_weight = None

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad = True

    def get_new_adapter_initial_msa(self):
        config = self.config
        for i in range(len(self.adapt_pos)):
            temp_adapter = nn.ModuleList()
            for j in self.msa:
                if j == 1:
                    adapter = Adapter_lora(config, dropout=0.0, bottleneck=config.ffn_num,
                                            init_option=config.ffn_adapter_init_option,
                                            adapter_scalar=config.ffn_adapter_scalar,
                                            adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                            ).to(self._device)
                else:
                    adapter = nn.Identity()
                temp_adapter.append(adapter)
            self.cur_adapter.append(temp_adapter)
        self.cur_adapter.requires_grad_(True)
        # NOTE: lora_B is NOT frozen here -- the checkpoint-loading freeze
        # pass in _build_from_timm_checkpoint runs AFTER __init__ and would
        # silently re-enable it (lora_B.weight is a "missing key" relative to
        # the pretrained checkpoint, so that pass's blanket
        # `requires_grad = name in missing_keys` would set it back to True).
        # The real, final freeze happens there instead, same ordering
        # constraint vit_cllora.py's own builder functions already have.

    def after_task_fold(self, admit_bw, admit_kd):
        """Task-boundary bookkeeping. The SKETCH fold ALWAYS runs (fixed
        size, never gated -- see this file's own docstring for why no bank-
        cap admission applies to it). `admit_bw`/`admit_kd` gate only the
        two structures that genuinely grow per task: block_weight_list (tiny,
        kept purely for future orthogonality-loss comparisons) and
        old_adapter_list (general-pos KD teacher snapshot, unchanged from
        baseline CL-LoRA's own bank-cap handling)."""
        all_retained = []
        for pos in range(len(self.specfic_pos)):
            if self.use_block_weight_scaling:
                bw_q = self.block_weight[0, pos].detach()
                bw_v = self.block_weight[2, pos].detach()
            else:
                bw_q = bw_v = 1.0
            retained = self.sketch_list[pos].fold(bw_q, bw_v, self.svd_rank, self.config)
            all_retained.extend(retained.values())
        if all_retained:
            logging.info("[CL-LoRA-Sketch] fold: retained_energy mean=%.4f min=%.4f",
                          sum(all_retained) / len(all_retained), min(all_retained))

        if self.use_block_weight_scaling:
            if admit_bw:
                self.block_weight_list.append(self.block_weight.detach().clone())
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)).to(self._device))
            nn.init.uniform_(self.block_weight, .5, 1.5)

        if self.use_distillation and admit_kd:
            self.old_adapter_list.append(copy.deepcopy(self.cur_adapter).requires_grad_(False))

        self.cur_adapter.requires_grad_(True)
        for pos in range(len(self.adapt_pos)):
            for j in range(len(self.msa)):
                if self.msa[j] == 1:
                    self.cur_adapter[pos][j].lora_B.requires_grad_(False)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if idx in self.general_pos:
                pos = self.adapt_pos.index(idx)
                x = blk(x, adapt=self.cur_adapter[pos])
            elif idx in self.specfic_pos:
                pos = self.specfic_pos.index(idx)
                bw = (self.block_weight[0, pos], self.block_weight[2, pos]) \
                    if self.use_block_weight_scaling else None
                x = blk(x, sketch=self.sketch_list[pos], block_weight=bw)
            else:
                x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, test=False):
        # No train/test branching needed in THIS method -- unlike
        # vit_cllora.py, there is no per-task feature list to iterate either
        # way (see this file's docstring). train()/eval() dropout/norm mode
        # is governed by self.training via the Learner's model.train()/
        # .eval() calls, not by this argument.
        return self.forward_features(x)

    def sketch_orth_loss(self):
        """"orth_a_side" regularizer, summed over every specific-pos layer
        -- see SketchQV.orth_loss's own docstring and this file's module
        docstring, "THREE VARIANTS". Returns a plain float 0.0 (not a
        tensor) if there are no specific-pos layers at all, matching this
        project's convention elsewhere for a structurally-inapplicable
        regularizer term."""
        if not self.specfic_pos:
            return 0.0
        return sum(sk.orth_loss() for sk in self.sketch_list)

    def forward_general_cls(self, x, t_idx):
        """KD teacher/student pass for general-pos layers -- identical logic
        to vit_cllora.py's own forward_general_cls, unchanged (general-pos
        is untouched by this variant)."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x_teacher = copy.deepcopy(x)

        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            x = self.blocks[j](x, adapt=self.cur_adapter[pos])
        x = self.norm(x)
        output_new = x[:, 0, :]

        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            adapt = self.old_adapter_list[-1][pos]
            x_teacher = self.blocks[j](x_teacher, adapt=adapt)
        x_teacher = self.norm(x_teacher)
        output_teacher = x_teacher[:, 0, :]

        return output_new, output_teacher


def _build_from_timm_checkpoint(tuning_config, timm_name):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                               qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                               tuning_config=tuning_config)
    checkpoint_model = timm.create_model(timm_name, pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = qkv_weight[:768]
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = qkv_weight[768:768 * 2]
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = qkv_weight[768 * 2:]
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = qkv_bias[:768]
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = qkv_bias[768:768 * 2]
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = qkv_bias[768 * 2:]
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            state_dict[key.replace('mlp.', '')] = state_dict.pop(key)

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    for name, p in model.named_parameters():
        p.requires_grad = name in msg.missing_keys
    # Final freeze pass, deliberately AFTER the blanket missing-keys pass
    # above (which would otherwise re-enable both of these -- see
    # get_new_adapter_initial_msa's own comment):
    #  - lora_B (general-pos adapters' frozen random-orthogonal basis).
    #  - sketch_q/sketch_v (every specific-pos layer's frozen folded
    #    history) -- must NEVER be trained by the optimizer; only
    #    SketchQV.fold's explicit .data.copy_() ever changes them.
    for pos in range(len(model.adapt_pos)):
        for j in range(len(model.msa)):
            if model.msa[j] == 1:
                model.cur_adapter[pos][j].lora_B.requires_grad_(False)
    for sketch_qv in model.sketch_list:
        sketch_qv.sketch_q.requires_grad_(False)
        sketch_qv.sketch_v.requires_grad_(False)
    return model


def vit_base_patch16_224_cllora_sketch(pretrained=False, **kwargs):
    return _build_from_timm_checkpoint(kwargs["tuning_config"], "vit_base_patch16_224")


def vit_base_patch16_224_in21k_cllora_sketch(pretrained=False, **kwargs):
    return _build_from_timm_checkpoint(kwargs["tuning_config"], "vit_base_patch16_224_in21k")
