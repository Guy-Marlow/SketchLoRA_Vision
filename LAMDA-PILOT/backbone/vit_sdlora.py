"""SD-LoRA -- "Scalable Decoupled Low-Rank Adaptation for Class Incremental
Learning" (ICLR 2025 Oral, Wu et al.), ported 2026-10-27 per user request.
See the research report this was built from: scratchpad/analysis_sdlora.md
(verified against SD-LoRA-Ref/models/sdlora.py and backbone/lora.py's actual
code before any of this file was written).

Reuses backbone/vit_lora.py's VisionTransformer/Block/Attention_LoRA almost
entirely unmodified (same per-task ModuleList-of-slots scaffold, same
pretrained-checkpoint loading) -- the ONE thing SD-LoRA genuinely needs that
the base backbone cannot express is a per-task LEARNABLE scalar magnitude
that is (a) shared across every block/projection (one scalar per task,
globally, not per-module) and (b) applied asymmetrically: each FROZEN past
task's raw B_s(A_s(x)) term is divided by that task's own (block-specific)
||B_s||_F * ||A_s||_F before being rescaled by its global scalar, while the
CURRENT task's term is left unnormalized. The base Attention_LoRA._lora_delta
only has a single fixed lora_scaling constant for the whole sum -- not
expressive enough -- hence the small subclass hierarchy below.

Per-task magnitude storage: a SINGLE nn.ParameterList (`task_scales`) owned
by the top-level VisionTransformerSDLoRA module, grown by one scalar per
task. Every block's Attention_LoRA_SDLoRA holds a REFERENCE to this same
object (via set_task_scales(), stored in a plain python list wrapper --
NOT assigned as an nn.Module attribute -- so nn.Module.__setattr__ never
re-registers it as a submodule of 12 different blocks, which would make
net.parameters() yield each scale 12x and silently apply a 12x learning-rate
multiplier to the optimizer). Growing task_scales via .append() mutates the
shared object in place, so every block's reference stays valid with no
re-propagation needed.

Design decision (research report open question #1, SD-LoRA-Ref's own
save_wrap_param -- which would persist a trained scale back to disk -- is
never called anywhere in the reference repo, so every task's scalar is
constructed fresh at 0.8 and re-optimized from scratch every task in the
released code, despite "magnitude continues to be optimized across the
sequence" being the method's stated point): this port WARM-STARTS each
scale from its own previously-learned value (freeze_to_task re-activates
0..task's scales for joint re-optimization, matching the paper's prose) --
an explicit, visible deviation from SD-LoRA-Ref's as-released behavior,
not a silent one. Flip by setting a fresh nn.Parameter(0.8) per task instead
of reusing task_scales[s] if literal reproduction of the reference's
apparent bug is ever wanted.
"""
import math
from functools import partial

import torch
from torch import nn

from backbone.vit_lora import (
    Attention_LoRA, Block, PatchEmbed, VisionTransformer, _load_pretrained_split_qkv,
)

SCALE_INIT = 0.8


class Attention_LoRA_SDLoRA(Attention_LoRA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scales_ref = None   # set via set_task_scales() -- plain list, never an nn.Module attr

    def set_task_scales(self, scales):
        self._scales_ref = [scales]

    @torch.no_grad()
    def add_task_slot(self, rank=None):
        """SD-LoRA-RR (2026-10-27 port): same as the base Attention_LoRA.
        add_task_slot, except the NEW slot may be allocated at a DIFFERENT
        rank than every previous slot (`rank=None` -> self.lora_rank,
        unchanged base behavior). Confirmed against SD-LoRA-Ref's own
        commented-out rank schedule (models/sdlora.py::update_network,
        "SD-LoRA-RR") before writing this: there, `r` governs only the
        FRESH LoRA_ViT_timm construction for the upcoming task -- already-
        trained past tasks' weights are loaded separately from disk at
        their own original shape, untouched by a later rank change. Same
        semantics here: each slot keeps whatever rank it was allocated at;
        only the newest slot's own rank is ever reduced."""
        r = rank if rank is not None else self.lora_rank
        ref = self.frozen_delta_q
        device, dtype = ref.device, ref.dtype
        new_A_q = nn.Linear(self.dim, r, bias=False).to(device=device, dtype=dtype)
        new_B_q = nn.Linear(r, self.dim, bias=False).to(device=device, dtype=dtype)
        new_A_v = nn.Linear(self.dim, r, bias=False).to(device=device, dtype=dtype)
        new_B_v = nn.Linear(r, self.dim, bias=False).to(device=device, dtype=dtype)
        nn.init.kaiming_uniform_(new_A_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(new_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(new_B_q.weight)
        nn.init.zeros_(new_B_v.weight)
        self.lora_A_q.append(new_A_q)
        self.lora_B_q.append(new_B_q)
        self.lora_A_v.append(new_A_v)
        self.lora_B_v.append(new_B_v)
        self.n_tasks += 1

    def _lora_delta(self, x, A_list, B_list, frozen_delta):
        task, merge = self._task, self._merge
        if task < 0:
            return 0.0
        scales = self._scales_ref[0]
        if not merge:
            # SD-LoRA-Ref never defines a task-routed (TIL, single-slot) forward
            # -- project-local choice for this unsupported-by-the-paper case:
            # this task's own term alone, unnormalized (same shape as the base
            # Attention_LoRA's own merge=False branch).
            return scales[task] * B_list[task](A_list[task](x))
        delta = scales[task] * B_list[task](A_list[task](x))
        for s in range(task):
            B_s, A_s = B_list[s].weight, A_list[s].weight
            norm = torch.linalg.norm(B_s) * torch.linalg.norm(A_s)
            delta = delta + scales[s] * (B_list[s](A_list[s](x)) / norm)
        return delta


class Block_SDLoRA(Block):
    def __init__(self, dim, num_heads=12, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 n_tasks=1, lora_rank=10, lora_alpha=None):
        super().__init__(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                          act_layer=act_layer, norm_layer=norm_layer,
                          n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha)
        # replace the plain Attention_LoRA super().__init__ built (thrown away) --
        # simplest correct way to swap the attn class without duplicating Block's
        # whole constructor body.
        self.attn = Attention_LoRA_SDLoRA(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                          attn_drop=attn_drop, proj_drop=drop, n_tasks=n_tasks,
                                          lora_rank=lora_rank, lora_alpha=lora_alpha)


class VisionTransformerSDLoRA(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, n_tasks=1, lora_rank=10, lora_alpha=None, n_lora_blocks=None):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                          num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                          num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                          drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                          drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                          norm_layer=norm_layer, act_layer=act_layer, n_tasks=n_tasks,
                          lora_rank=lora_rank, lora_alpha=lora_alpha, n_lora_blocks=n_lora_blocks)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # replace the plain-Block self.blocks super().__init__ built (thrown away).
        self.blocks = nn.ModuleList([
            Block_SDLoRA(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                        norm_layer=norm_layer, act_layer=act_layer, n_tasks=n_tasks,
                        lora_rank=lora_rank, lora_alpha=lora_alpha)
            for i in range(depth)])
        self.task_scales = nn.ParameterList([nn.Parameter(torch.tensor([SCALE_INIT]))])
        for attn in self.attn_modules():
            attn.set_task_scales(self.task_scales)

    def add_task_slot(self, rank=None):
        # NOT calling super().add_task_slot() -- the base VisionTransformer
        # version always uses self.lora_rank, with no way to override it per
        # call; re-implemented here so SD-LoRA-RR's per-task rank actually
        # reaches each block's new slot.
        for blk in self.blocks:
            blk.attn.add_task_slot(rank=rank)
        self.n_tasks += 1
        ref = self.task_scales[0]
        self.task_scales.append(nn.Parameter(torch.tensor([SCALE_INIT], device=ref.device, dtype=ref.dtype)))
        # every block's _scales_ref already points at this SAME ParameterList
        # object (set once in __init__) -- append() mutates it in place, so no
        # re-propagation to the blocks is needed.

    def freeze_to_task(self, task, train_a=True):
        super().freeze_to_task(task, train_a=train_a)
        # SD-LoRA re-activates EVERY past task's magnitude scalar alongside the
        # new task's own (paper: "the magnitude continues to be optimized
        # across the whole sequence") -- the base freeze_to_task only
        # re-enables the current task's own A/B, nothing about this
        # network-level scale list.
        for s in range(task + 1):
            self.task_scales[s].requires_grad = True


def vit_base_patch16_224_sdlora(pretrained=False, n_tasks=1, lora_rank=10, lora_alpha=None, **kwargs):
    model = VisionTransformerSDLoRA(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                    n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224")


def vit_base_patch16_224_in21k_sdlora(pretrained=False, n_tasks=1, lora_rank=10, lora_alpha=None, **kwargs):
    model = VisionTransformerSDLoRA(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                    n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224_in21k")
