# --------------------------------------------------------
# "LoRA Subtraction for Drift-Resistant Space" (LoRA_Subtraction_Ref/) ViT
# backbone for LAMDA-PILOT.
#
# Forked from backbone/vit_lora.py (baseline per-task LoRA on Q,V). LoRA
# Subtraction instead places its per-task LoRA bank on K and V -- Q stays
# plain. Everything else (growable slot list via add_task_slot(), merge=True
# CIL-style summed forward, frozen-slot dense folding, input-covariance
# accumulation for InfLoRA-style subspace construction) is the SAME
# machinery vit_lora.py already provides, just renamed q->k where that
# branch's LoRA now lives. This file is used by BOTH LoRA Subtraction
# variants (models/lorasub.py, growing per-task bank, and
# models/lorasub_sketch.py, fixed 2-slot SketchLoRA-style fold) -- the
# backbone is bank-size-agnostic, exactly like vit_lora.py already is for
# O-LoRA (growing) vs InfLoRA-Sketch (fixed 2-slot).
#
# THE NAMESAKE "SUBTRACTION" TRICK
# ---------------------------------
# LoRA_Subtraction_Ref/methods/lorasub_drs.py builds the input-covariance
# statistics used for its gradient-projection subspace (see
# models/lorasub.py's _ProjectedAdam) from a "cleaned" feature distribution:
# a special probe forward pass computes k/v normally, then explicitly
# SUBTRACTS the sum of every PAST task's own frozen (B_k[t]@A_k[t]) / v
# analog back out, before attention runs and before the residual stream
# carries the result into later blocks (see that file's
# Attention_LoRA.forward, `get_cur_x=True` branch).
#
# Here, that probe is `self._subtract_mode = True`: it skips adding ANY
# LoRA delta to k/v at all, for every block, using only the frozen
# k_proj/v_proj. This is not a simplification that changes behavior -- it
# is the EXACT same computation as the reference's explicit subtraction,
# given this project's (and the reference's own add_task_slot-equivalent)
# zero-B-init convention: the probe always runs BEFORE the current task's
# own slot has been trained, so delta_current(x) == 0 identically, and the
# reference's `k - sum_{t<task} delta_t(x)` reduces to
# `(k_proj(x) + sum_{t<=task} delta_t(x)) - sum_{t<task} delta_t(x))
#   == k_proj(x) + delta_current(x) == k_proj(x)`.
# So "probe = pure frozen-base forward" is a direct algebraic consequence
# of zero-B-init, not a departure from the reference's design.
# --------------------------------------------------------

import math
from functools import partial

import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed


class Attention_LoRASub(nn.Module):
    """Multi-head attention with per-task LoRA on the K and V projections."""

    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 n_tasks=1, lora_rank=10, lora_alpha=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim = dim

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # ---- per-task LoRA on key and value ----
        self.n_tasks = n_tasks
        self.lora_rank = lora_rank
        self.rank = lora_rank  # alias used by covariance-init methods (InfLoRA-style)
        self.lora_scaling = (lora_alpha if lora_alpha is not None else lora_rank) / lora_rank
        self.lora_A_k = nn.ModuleList([nn.Linear(dim, lora_rank, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(lora_rank, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, lora_rank, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(lora_rank, dim, bias=False) for _ in range(n_tasks)])
        for t in range(n_tasks):
            nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

        # routing state (set via set_task before each forward)
        self._task = -1
        self._merge = False

        # ---- optional input-covariance accumulation (InfLoRA-style / DRS) ----
        # cur_matrix = running average of x^T x over tokens for the block's
        # attention input. Off by default.
        self._collect = False
        self.register_buffer('cur_matrix', torch.zeros(dim, dim))
        self.n_cur_matrix = 0

        # ---- "subtraction" probe mode (LoRA Subtraction's namesake trick) ----
        # When True, forward skips BOTH LoRA deltas entirely (k=k_proj(x),
        # v=v_proj(x)) -- see this module's docstring for why that equals the
        # reference's explicit per-task subtraction.
        self._subtract_mode = False

        # ---- frozen-slot folding (opt-in, unused by either LoRA-Subtraction
        # variant today, kept for structural parity with backbone/vit_lora.py) ----
        self._fold_enabled = False
        self._folded_upto = -1
        self.register_buffer('frozen_delta_k', torch.zeros(dim, dim))
        self.register_buffer('frozen_delta_v', torch.zeros(dim, dim))

    def enable_frozen_folding(self):
        self._fold_enabled = True

    @torch.no_grad()
    def add_task_slot(self):
        """Append one more (A,B) pair for both k and v, matching the
        constructor's device/dtype/init convention exactly (kaiming A, zero B)."""
        ref = self.frozen_delta_k
        device, dtype = ref.device, ref.dtype
        new_A_k = nn.Linear(self.dim, self.lora_rank, bias=False).to(device=device, dtype=dtype)
        new_B_k = nn.Linear(self.lora_rank, self.dim, bias=False).to(device=device, dtype=dtype)
        new_A_v = nn.Linear(self.dim, self.lora_rank, bias=False).to(device=device, dtype=dtype)
        new_B_v = nn.Linear(self.lora_rank, self.dim, bias=False).to(device=device, dtype=dtype)
        nn.init.kaiming_uniform_(new_A_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(new_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(new_B_k.weight)
        nn.init.zeros_(new_B_v.weight)
        self.lora_A_k.append(new_A_k)
        self.lora_B_k.append(new_B_k)
        self.lora_A_v.append(new_A_v)
        self.lora_B_v.append(new_B_v)
        self.n_tasks += 1

    @torch.no_grad()
    def fold_up_to(self, task):
        if not self._fold_enabled or task <= self._folded_upto:
            return
        for t in range(self._folded_upto + 1, task + 1):
            self.frozen_delta_k += self.lora_B_k[t].weight @ self.lora_A_k[t].weight
            self.frozen_delta_v += self.lora_B_v[t].weight @ self.lora_A_v[t].weight
        self._folded_upto = task

    @torch.no_grad()
    def free_folded_slot(self, task):
        if task < 0 or task > self._folded_upto:
            return
        self.lora_A_k[task] = nn.Identity()
        self.lora_B_k[task] = nn.Identity()
        self.lora_A_v[task] = nn.Identity()
        self.lora_B_v[task] = nn.Identity()

    def set_task(self, task, merge=False):
        self._task = task
        self._merge = merge

    def set_collect(self, flag):
        self._collect = flag

    def set_subtract_mode(self, flag):
        self._subtract_mode = flag

    def reset_cur_matrix(self):
        self.cur_matrix = torch.zeros_like(self.cur_matrix)
        self.n_cur_matrix = 0

    def _accumulate_cov(self, x):
        # x: [B, N, C] input to the q/k/v projections (post norm1)
        xd = x.detach()
        cov = torch.bmm(xd.permute(0, 2, 1), xd).sum(dim=0)
        n = x.shape[0] * x.shape[1]
        self.cur_matrix = (self.cur_matrix * self.n_cur_matrix + cov) / (self.n_cur_matrix + n)
        self.n_cur_matrix += n

    def _lora_delta(self, x, A_list, B_list, frozen_delta):
        task, merge = self._task, self._merge
        if task < 0:
            return 0.0
        if not merge:
            return B_list[task](A_list[task](x)) * self.lora_scaling
        if self._fold_enabled:
            assert task >= self._folded_upto, (
                "merge=True requested for task={} but frozen_delta already folded through "
                "task={}".format(task, self._folded_upto))
            delta = nn.functional.linear(x, frozen_delta)
            if task > self._folded_upto:
                delta = delta + B_list[task](A_list[task](x))
            return delta * self.lora_scaling
        delta = 0.0
        for t in range(task + 1):
            delta = delta + B_list[t](A_list[t](x))
        return delta * self.lora_scaling

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape
        if self._collect:
            self._accumulate_cov(x)

        q = self.q_proj(x)
        if self._subtract_mode:
            # LoRA Subtraction's probe pass: pure frozen-base k/v, no adapter
            # contribution at all (past or current) -- see module docstring.
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            k = self.k_proj(x) + self._lora_delta(x, self.lora_A_k, self.lora_B_k, self.frozen_delta_k)
            v = self.v_proj(x) + self._lora_delta(x, self.lora_A_v, self.lora_B_v, self.frozen_delta_v)

        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)
        k = self._shape(k, -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(v, -1, B).view(B * self.num_heads, -1, self.head_dim)

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
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 n_tasks=1, lora_rank=10, lora_alpha=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_LoRASub(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                      attn_drop=attn_drop, proj_drop=drop,
                                      n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def set_task(self, task, merge=False):
        self.attn.set_task(task, merge)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))
        x = residual + x
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, n_tasks=1, lora_rank=10, lora_alpha=None, n_lora_blocks=None):
        super().__init__()
        print("I'm using ViT with per-task LoRA Subtraction (k,v).")
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.n_tasks = n_tasks
        self.n_lora_blocks = n_lora_blocks

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                  act_layer=act_layer, n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

    # -- routing helpers ------------------------------------------------
    def set_task(self, task, merge=False):
        for blk in self.blocks:
            blk.set_task(task, merge)

    def freeze_to_task(self, task, train_a=True):
        """Freeze the whole backbone except task `task`'s LoRA parameters.
        LoRA Subtraction trains BOTH A and B of the current task's slot via
        plain gradient descent (the projection is applied to the Adam
        update, not baked into A analytically) -- train_a defaults True,
        unlike InfLoRA's own analytic-A convention."""
        self.fold_frozen_slots(task - 1)
        for p in self.parameters():
            p.requires_grad = False
        train_blocks = self.blocks if self.n_lora_blocks is None else self.blocks[:self.n_lora_blocks]
        for blk in train_blocks:
            train_lists = [blk.attn.lora_B_k, blk.attn.lora_B_v]
            if train_a:
                train_lists += [blk.attn.lora_A_k, blk.attn.lora_A_v]
            for mlist in train_lists:
                for p in mlist[task].parameters():
                    p.requires_grad = True

    def add_task_slot(self):
        for blk in self.blocks:
            blk.attn.add_task_slot()
        self.n_tasks += 1

    # -- frozen-slot folding (opt-in; unused today, see Attention_LoRASub) --
    def enable_frozen_folding(self):
        for attn in self.attn_modules():
            attn.enable_frozen_folding()

    def fold_frozen_slots(self, task):
        if task < 0:
            return
        for attn in self.attn_modules():
            attn.fold_up_to(task)

    def free_folded_slots(self, task):
        if task < 0:
            return
        for attn in self.attn_modules():
            attn.free_folded_slot(task)

    # -- input-covariance collection / subtraction probe -----------------
    def attn_modules(self):
        return [blk.attn for blk in self.blocks]

    def set_collect(self, flag):
        for attn in self.attn_modules():
            attn.set_collect(flag)

    def set_subtract_mode(self, flag):
        for attn in self.attn_modules():
            attn.set_subtract_mode(flag)

    def reset_cur_matrix(self):
        for attn in self.attn_modules():
            attn.reset_cur_matrix()

    # -- forward --------------------------------------------------------
    def forward_features(self, x, task=-1, merge=False):
        self.set_task(task, merge)
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, task=-1, merge=False):
        return self.forward_features(x, task=task, merge=merge)


def _load_pretrained_split_qkv(model, timm_name):
    """Load a timm ViT-B/16 checkpoint into the split-qkv LoRA-Subtraction model."""
    checkpoint_model = timm.create_model(timm_name, pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            w = state_dict.pop(key)
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = w[:768]
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = w[768:768 * 2]
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = w[768 * 2:]
        elif 'qkv.bias' in key:
            b = state_dict.pop(key)
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = b[:768]
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = b[768:768 * 2]
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = b[768 * 2:]
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            state_dict[key.replace('mlp.', '')] = state_dict.pop(key)
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def vit_base_patch16_224_lorasub(pretrained=False, n_tasks=1, lora_rank=10, lora_alpha=None, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224")


def vit_base_patch16_224_in21k_lorasub(pretrained=False, n_tasks=1, lora_rank=10, lora_alpha=None, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224_in21k")
