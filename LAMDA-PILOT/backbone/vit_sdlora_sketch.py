"""SD-LoRA + Sketch backbone -- SD-LoRA's magnitude/direction decoupling
applied to SketchLoRA's fixed-size, QR-SVD-folded sketch, with the sketch's
own singular values made per-task re-learnable via a small rescaling vector.
Built from a web-claude implementation brief (2026-11-06 user request); see
models/sdlora_sketch.py's module docstring for the full algorithm, the
flagged deviations from both SD-LoRA and plain SketchLoRA, and the
rationale behind each one. This file holds only the backbone mechanics
(forward math, the per-block QR-SVD fold); the Learner (training loop,
regularizer, task-boundary orchestration) lives in that other file.

Reuses backbone/vit_lora.py's Attention_LoRA/Block/VisionTransformer
scaffold the same way backbone/vit_sdlora.py does: inherit for the frozen
q/k/v/proj machinery and the overridable `_lora_delta` seam, discard the
base class's per-task ModuleList-of-slots growth entirely (n_tasks is
forced to 1 and NEVER grows -- see VisionTransformerSDLoRASketch.
add_task_slot, which raises).

STORAGE PER (BLOCK, PROJECTION) -- Q and V each get their own, independent
(no sharing, see the DEVIATION note in models/sdlora_sketch.py):
  - sketch_U/_S/_Vh (buffers, non-trainable, fixed shape [dim,r_hat]/
    [r_hat]/[r_hat,dim]): the explicit factored SVD of the folded history.
    Kept with S exposed SEPARATELY (not absorbed into U/Vh via sqrt(S), the
    way plain SketchLoRA's slot 0 stores it) -- everything in this file
    depends on being able to rescale S directly.
  - rescale (nn.Parameter, length r_hat, init 0): this task's proposed
    multiplicative adjustment to each singular value, S_eff = S*(1+rescale).
    Structured behind _apply_rescale() below so a later dense variant can
    swap the vector for a square matrix without touching any caller (the
    brief's own requirement).
  - mag (nn.Parameter, scalar, init 0): this task's new-adapter magnitude.
  - lora_A_{q,v}[0] / lora_B_{q,v}[0] (reused from the base class's own
    single-slot ModuleList, which this method never grows past index 0):
    this task's new-adapter DIRECTION factors, used only through their
    unit-normalized product -- see _new_adapter_term.
"""

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from backbone.vit_lora import (
    Attention_LoRA, Block, PatchEmbed, VisionTransformer, _load_pretrained_split_qkv,
)

# Small-nonzero init for BOTH new-adapter factors (A and B) -- NOT standard
# LoRA's kaiming-A/zero-B init, which is unusable here: this method
# normalizes by ||B_new @ A_new||_F, undefined at B_new=0. Flagged deviation
# -- see models/sdlora_sketch.py's module docstring, "INIT DEVIATION".
NEW_ADAPTER_INIT_STD = 0.02

_PROJS = ("q", "v")


class Attention_SDLoRASketch(Attention_LoRA):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 lora_rank=10, lora_alpha=None, svd_rank=10, freeze_rescale=False):
        # n_tasks forced to 1 -- this method never grows a per-task slot
        # bank; slot 0 is permanently reused as "this task's new adapter".
        # lora_alpha/lora_scaling are accepted for constructor-signature
        # compatibility with the base class and the factory functions below,
        # but intentionally UNUSED by this class's own _lora_delta -- the
        # learned magnitude `mag` already serves as this method's scale
        # knob; layering a second fixed alpha/rank multiplier on top of it
        # would just rescale what is already a free parameter.
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                          attn_drop=attn_drop, proj_drop=proj_drop,
                          n_tasks=1, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.svd_rank = svd_rank
        self.freeze_rescale = freeze_rescale
        for proj in _PROJS:
            self.register_buffer("sketch_U_{}".format(proj), torch.zeros(dim, svd_rank))
            self.register_buffer("sketch_S_{}".format(proj), torch.zeros(svd_rank))
            self.register_buffer("sketch_Vh_{}".format(proj), torch.zeros(svd_rank, dim))
            setattr(self, "rescale_{}".format(proj), nn.Parameter(torch.zeros(svd_rank)))
            setattr(self, "mag_{}".format(proj), nn.Parameter(torch.zeros(1)))
            getattr(self, "rescale_{}".format(proj)).requires_grad = not freeze_rescale
        self._new_task_params()  # overwrite the base class's kaiming-A/zero-B init on slot 0

    @torch.no_grad()
    def _new_task_params(self):
        """Fresh per-task trainable pieces: the new adapter's direction
        factors, its magnitude, and the rescale vector -- called at
        construction and again at every task boundary (fold(), below).
        NEVER warm-started from the previous task's values (see models/
        sdlora_sketch.py's "WHY NO WARM-START" note) -- the next _train()
        call builds a brand-new optimizer around whatever these are reset
        to, so in-place reinitialization (rather than replacing the
        Parameter objects) is sufficient; no stale Adam state survives
        either way."""
        for proj in _PROJS:
            A = getattr(self, "lora_A_{}".format(proj))[0].weight
            B = getattr(self, "lora_B_{}".format(proj))[0].weight
            nn.init.normal_(A, std=NEW_ADAPTER_INIT_STD)
            nn.init.normal_(B, std=NEW_ADAPTER_INIT_STD)
            getattr(self, "rescale_{}".format(proj)).data.zero_()
            getattr(self, "mag_{}".format(proj)).data.zero_()

    @staticmethod
    def _apply_rescale(S, rescale):
        """S*(1+rescale) -- the vector case. Every caller below only ever
        needs the resulting length-r_hat effective S, so a later dense
        variant swaps this one function for, e.g., S @ (I + matrix) without
        touching fold() or _sketch_term() at all (the brief's own
        requirement)."""
        return S * (1.0 + rescale)

    def _sketch_term(self, x, proj):
        """Apply the sketch as project-in -> scale -> project-out. Never
        materializes the dense [dim,dim] delta."""
        U = getattr(self, "sketch_U_{}".format(proj))
        S = getattr(self, "sketch_S_{}".format(proj))
        Vh = getattr(self, "sketch_Vh_{}".format(proj))
        rescale = getattr(self, "rescale_{}".format(proj))
        S_eff = self._apply_rescale(S, rescale)
        proj_in = F.linear(x, Vh)       # [..., r_hat] = x @ Vh^T
        scaled = proj_in * S_eff         # elementwise scale by the (rescaled) singular values
        return F.linear(scaled, U)       # [..., dim] = scaled @ U^T

    def _new_adapter_term(self, x, proj):
        """mag * normalize(B_new @ A_new) applied to x, via the trace trick
        (||BA||_F^2 = trace(B^T B @ A A^T)) so the dense [dim,dim] product
        is never formed just to compute its norm."""
        A = getattr(self, "lora_A_{}".format(proj))[0]
        B = getattr(self, "lora_B_{}".format(proj))[0]
        mag = getattr(self, "mag_{}".format(proj))
        Aw, Bw = A.weight, B.weight          # [r,dim], [dim,r]
        AAt = Aw @ Aw.t()
        BtB = Bw.t() @ Bw
        normsq = (AAt * BtB).sum().clamp_min(1e-12)
        scale = mag / normsq.sqrt()
        return scale * B(A(x))

    def _lora_delta(self, x, A_list, B_list, frozen_delta):
        # frozen_delta (the base class's dense fold buffer) is unused here
        # by design -- this method keeps no dense frozen state, only the
        # explicit factored sketch buffers above (see module docstring:
        # "never materialize a dense weight matrix").
        proj = "q" if A_list is self.lora_A_q else "v"
        return self._sketch_term(x, proj) + self._new_adapter_term(x, proj)

    @torch.no_grad()
    def fold(self):
        """Task-boundary merge: bake this task's rescale into S, scale the
        new adapter by mag/||B_new A_new||_F, QR-SVD-merge both into the
        sketch (double precision throughout -- a rescale entry below -1
        drives S_eff negative; the signed-sqrt factorization below plus
        the merge's own fresh SVD absorb that sign flip with no special
        case), truncate to svd_rank, report retained energy, then reset
        the new-adapter/rescale/magnitude for the next task.

        Same QR + small-SVD + truncate algorithm as models/
        lorasub_sketch.py::_sketch_fold, generalized to a possibly-nonzero
        S (there, the merge always starts from a fresh random "SKETCH"
        slot that needs an explicit first-task transplant to avoid folding
        in garbage; here the sketch buffers start at true zero, so the
        SAME general path below is already exact for the first fold too --
        no transplant special-case needed).

        Returns {"q": retained_energy, "v": retained_energy}."""
        retained = {}
        for proj in _PROJS:
            U = getattr(self, "sketch_U_{}".format(proj))
            S = getattr(self, "sketch_S_{}".format(proj))
            Vh = getattr(self, "sketch_Vh_{}".format(proj))
            rescale = getattr(self, "rescale_{}".format(proj))
            mag = getattr(self, "mag_{}".format(proj))
            A = getattr(self, "lora_A_{}".format(proj))[0].weight
            B = getattr(self, "lora_B_{}".format(proj))[0].weight

            S_eff = S.double() * (1.0 + rescale.double())
            sign = torch.sign(S_eff)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            sqrt_abs = S_eff.abs().sqrt()
            B_sk = U.double() * sqrt_abs                            # [dim, r_hat]
            A_sk = (sign * sqrt_abs).unsqueeze(1) * Vh.double()     # [r_hat, dim]

            Ad, Bd = A.double(), B.double()
            AAt = Ad @ Ad.t()
            BtB = Bd.t() @ Bd
            normsq = (AAt * BtB).sum().clamp_min(1e-12)
            adapter_scale = mag.double() / normsq.sqrt()
            B_ad = Bd * adapter_scale                                # fold the scalar magnitude into one factor
            A_ad = Ad

            delta_true = (B_sk @ A_sk) + (B_ad @ A_ad)               # transient, diagnostic-only

            B_cat = torch.cat([B_sk, B_ad], dim=1)                   # [dim, r_hat+r]
            A_cat = torch.cat([A_sk, A_ad], dim=0)                   # [r_hat+r, dim]
            Q_B, R_B = torch.linalg.qr(B_cat)
            Q_A, R_A = torch.linalg.qr(A_cat.t())
            M_small = R_B @ R_A.t()
            U_small, S_new, Vh_small = torch.linalg.svd(M_small)
            U_full = Q_B @ U_small
            Vh_full = Vh_small @ Q_A.t()

            r = min(self.svd_rank, S_new.shape[0])
            recon = (U_full[:, :r] * S_new[:r]) @ Vh_full[:r, :]
            fro_true = delta_true.norm().clamp_min(1e-12)
            retained_energy = (1.0 - ((delta_true - recon).norm() / fro_true) ** 2).item()
            retained[proj] = retained_energy

            U.zero_()
            S.zero_()
            Vh.zero_()
            U[:, :r] = U_full[:, :r].to(U.dtype)
            S[:r] = S_new[:r].to(S.dtype)
            Vh[:r, :] = Vh_full[:r, :].to(Vh.dtype)

        self._new_task_params()
        return retained


class Block_SDLoRASketch(Block):
    def __init__(self, dim, num_heads=12, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 lora_rank=10, lora_alpha=None, svd_rank=10, freeze_rescale=False):
        super().__init__(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                          act_layer=act_layer, norm_layer=norm_layer,
                          n_tasks=1, lora_rank=lora_rank, lora_alpha=lora_alpha)
        # replace the plain Attention_LoRA super().__init__ built (thrown away).
        self.attn = Attention_SDLoRASketch(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop, proj_drop=drop,
                                           lora_rank=lora_rank, lora_alpha=lora_alpha,
                                           svd_rank=svd_rank, freeze_rescale=freeze_rescale)


class VisionTransformerSDLoRASketch(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, lora_rank=10, lora_alpha=None, n_lora_blocks=None,
                 svd_rank=10, freeze_rescale=False):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                          num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                          num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                          drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                          drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                          norm_layer=norm_layer, act_layer=act_layer, n_tasks=1,
                          lora_rank=lora_rank, lora_alpha=lora_alpha, n_lora_blocks=n_lora_blocks)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # replace the plain-Block self.blocks super().__init__ built (thrown away).
        self.blocks = nn.ModuleList([
            Block_SDLoRASketch(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                               norm_layer=norm_layer, act_layer=act_layer,
                               lora_rank=lora_rank, lora_alpha=lora_alpha,
                               svd_rank=svd_rank, freeze_rescale=freeze_rescale)
            for i in range(depth)])
        self.svd_rank = svd_rank

    def freeze_to_task(self, task, train_a=True):
        """This method never grows slots -- slot 0 is reused forever as
        "this task's new adapter" (see models/sdlora_sketch.py). `task`/
        `train_a` are accepted for interface compatibility with
        LoRAVitNet.freeze_to_task but ignored -- the operation is always
        the same regardless of task index: freeze everything, then
        re-enable slot 0's A/B and the magnitude/rescale params (rescale
        gated per-block by that block's own freeze_rescale ablation flag).
        Respects self.n_lora_blocks exactly as the base class's own
        freeze_to_task does (deeper blocks stay fully frozen -- mag never
        trains there, so they contribute exactly 0 forever)."""
        for p in self.parameters():
            p.requires_grad = False
        train_blocks = self.blocks if self.n_lora_blocks is None else self.blocks[:self.n_lora_blocks]
        for blk in train_blocks:
            attn = blk.attn
            for proj in _PROJS:
                getattr(attn, "lora_A_{}".format(proj))[0].weight.requires_grad = True
                getattr(attn, "lora_B_{}".format(proj))[0].weight.requires_grad = True
                getattr(attn, "mag_{}".format(proj)).requires_grad = True
                getattr(attn, "rescale_{}".format(proj)).requires_grad = not attn.freeze_rescale

    def add_task_slot(self):
        raise NotImplementedError(
            "SD-LoRA-Sketch is a fixed-size method (one sketch + one reused "
            "new-adapter slot per block) -- it never grows a per-task slot bank.")

    def fold(self):
        """Fold every block's current new-adapter into its sketch. Returns
        {block_idx: {"q": retained_energy, "v": retained_energy}} for the
        Learner to aggregate/log."""
        return {i: blk.attn.fold() for i, blk in enumerate(self.blocks)}


def vit_base_patch16_224_sdlora_sketch(pretrained=False, lora_rank=10, lora_alpha=None,
                                        svd_rank=10, freeze_rescale=False, **kwargs):
    model = VisionTransformerSDLoRASketch(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                                          mlp_ratio=4, qkv_bias=True,
                                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                          lora_rank=lora_rank, lora_alpha=lora_alpha,
                                          svd_rank=svd_rank, freeze_rescale=freeze_rescale, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224")


def vit_base_patch16_224_in21k_sdlora_sketch(pretrained=False, lora_rank=10, lora_alpha=None,
                                              svd_rank=10, freeze_rescale=False, **kwargs):
    model = VisionTransformerSDLoRASketch(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                                          mlp_ratio=4, qkv_bias=True,
                                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                          lora_rank=lora_rank, lora_alpha=lora_alpha,
                                          svd_rank=svd_rank, freeze_rescale=freeze_rescale, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224_in21k")
