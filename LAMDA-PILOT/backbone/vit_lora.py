# --------------------------------------------------------
# Baseline per-task LoRA ViT for LAMDA-PILOT.
#
# Modelled on backbone/vit_adapter.py (which already splits the fused timm
# qkv weight into separate q/k/v projections and loads the pretrained ViT-B/16
# checkpoint).  Here we drop the AdaptFormer adapter and instead attach a
# *per-task* low-rank (LoRA) update to the query and value projections of every
# attention block.  Query/value placement is the convention shared by all of
# the methods we are porting into this bench.
#
# Task routing
# ------------
#   * Each attention block owns `n_tasks` LoRA pairs per projection.
#   * `set_task(task, merge)` selects the routing behaviour before a forward:
#       - merge=False  -> only LoRA[task] is added (task-routed / TIL style).
#       - merge=True   -> sum of LoRA[0..task] is added (merged / CIL style).
#   * Only the current task's LoRA is left trainable; everything else (incl.
#     the frozen ViT backbone and past-task LoRAs) has requires_grad=False.
# --------------------------------------------------------

import math
from functools import partial

import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed


class Attention_LoRA(nn.Module):
    """Multi-head attention with per-task LoRA on the Q and V projections."""

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

        # ---- per-task LoRA on query and value ----
        self.n_tasks = n_tasks
        self.lora_rank = lora_rank
        self.rank = lora_rank  # alias used by covariance-init methods (InfLoRA)
        self.lora_scaling = (lora_alpha if lora_alpha is not None else lora_rank) / lora_rank
        self.lora_A_q = nn.ModuleList([nn.Linear(dim, lora_rank, bias=False) for _ in range(n_tasks)])
        self.lora_B_q = nn.ModuleList([nn.Linear(lora_rank, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, lora_rank, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(lora_rank, dim, bias=False) for _ in range(n_tasks)])
        for t in range(n_tasks):
            nn.init.kaiming_uniform_(self.lora_A_q[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_q[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

        # routing state (set via set_task before each forward)
        self._task = -1
        self._merge = False

        # ---- optional input-covariance accumulation (InfLoRA / DualGPM) ----
        # cur_matrix = running average of x^T x over tokens for the LoRA input x.
        # Off by default so the baseline / O-LoRA paths pay nothing.
        self._collect = False
        self.register_buffer('cur_matrix', torch.zeros(dim, dim))
        self.n_cur_matrix = 0

        # ---- frozen-slot folding (opt-in, see fold_frozen_slots()) ----
        # merge=True forward normally loops `for t in range(task+1): B_t(A_t(x))` every
        # call -- O(K) matmuls/forward. If a method's frozen slots are IMMUTABLE once
        # frozen (true for olora/inflora/treelora/hidelora; NOT true for sketchlora,
        # whose sketch slot is periodically overwritten by compression), every frozen
        # slot's contribution can be folded ONCE into a dense [dim,dim] delta and reused
        # every forward until the next fold, turning the per-forward cost O(1). Off by
        # default (`_fold_enabled=False`) so nothing changes unless a method opts in via
        # enable_frozen_folding() -- sketchlora/seqlora/plain lora never call it.
        self._fold_enabled = False
        self._folded_upto = -1   # highest task index already folded into frozen_delta_*
        self.register_buffer('frozen_delta_q', torch.zeros(dim, dim))
        self.register_buffer('frozen_delta_v', torch.zeros(dim, dim))

    def enable_frozen_folding(self):
        self._fold_enabled = True

    @torch.no_grad()
    def add_task_slot(self):
        """Append one more (A,B) pair for both q and v, matching the constructor's
        device/dtype/init convention exactly (kaiming A, zero B). Used by
        boundary-agnostic streaming (models/stream_mixin.py) when the adapter-fold
        clock advances past however many slots were preallocated at construction --
        the fold count there is driven by a memory-constraint sample threshold, not
        by real task count, so it is not generically bounded by `n_tasks`.

        Device/dtype reference is frozen_delta_q (a register_buffer, always a real
        tensor) rather than lora_A_q[0].weight -- for methods that opt into
        free_folded_slot (InfLoRA), slot 0 is replaced with nn.Identity() once
        folded, which has no .weight and would crash this lookup once a later
        task needs a new slot."""
        ref = self.frozen_delta_q
        device, dtype = ref.device, ref.dtype
        new_A_q = nn.Linear(self.dim, self.lora_rank, bias=False).to(device=device, dtype=dtype)
        new_B_q = nn.Linear(self.lora_rank, self.dim, bias=False).to(device=device, dtype=dtype)
        new_A_v = nn.Linear(self.dim, self.lora_rank, bias=False).to(device=device, dtype=dtype)
        new_B_v = nn.Linear(self.lora_rank, self.dim, bias=False).to(device=device, dtype=dtype)
        nn.init.kaiming_uniform_(new_A_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(new_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(new_B_q.weight)
        nn.init.zeros_(new_B_v.weight)
        self.lora_A_q.append(new_A_q)
        self.lora_B_q.append(new_B_q)
        self.lora_A_v.append(new_A_v)
        self.lora_B_v.append(new_B_v)
        self.n_tasks += 1

    @torch.no_grad()
    def fold_up_to(self, task):
        """Fold every newly-frozen slot in (self._folded_upto, task] into the dense
        frozen_delta buffers. Idempotent -- calling with a `task` <= the current
        `_folded_upto` is a no-op."""
        if not self._fold_enabled or task <= self._folded_upto:
            return
        for t in range(self._folded_upto + 1, task + 1):
            self.frozen_delta_q += self.lora_B_q[t].weight @ self.lora_A_q[t].weight
            self.frozen_delta_v += self.lora_B_v[t].weight @ self.lora_A_v[t].weight
        self._folded_upto = task

    @torch.no_grad()
    def free_folded_slot(self, task):
        """FLAGGED CHANGE (2026-07-21): free a single already-folded slot's (lora_A/
        lora_B, q/v) weight memory by replacing it with a zero-parameter nn.Identity
        placeholder. `fold_up_to` only ever ADDS a slot's contribution into
        frozen_delta -- it never freed the original tensors, so every folded method
        was carrying an ever-growing, fully redundant O(K) bank of dead per-task
        weights (their contribution already lives in frozen_delta; forward's fold
        branch above never reads an old slot by index again once folded).

        NOT wired into fold_up_to/freeze_to_task itself, and NOT safe to call for
        every folding method: O-LoRA's orthogonality penalty reads every individual
        past lora_A forever (genuinely needs them), so freeing here would silently
        break it. This is opt-in, called explicitly only by models/inflora.py,
        which never reads a slot again once its own fold_up_to has consumed it
        (confirmed: _lora_delta's fold branch only reads frozen_delta + the current
        live slot; InfLoRA's TIL routing and DualGPM update both also go through
        the merged/fold-aware forward, never indexing an old slot directly)."""
        if task < 0 or task > self._folded_upto:
            return  # only free slots that have actually been folded
        self.lora_A_q[task] = nn.Identity()
        self.lora_B_q[task] = nn.Identity()
        self.lora_A_v[task] = nn.Identity()
        self.lora_B_v[task] = nn.Identity()

    def set_task(self, task, merge=False):
        self._task = task
        self._merge = merge

    def set_collect(self, flag):
        self._collect = flag

    def reset_cur_matrix(self):
        self.cur_matrix = torch.zeros_like(self.cur_matrix)
        self.n_cur_matrix = 0

    def _accumulate_cov(self, x):
        # x: [B, N, C] input to the q/v projections (post norm1)
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
            # merge=True is only ever queried (in this codebase) for the CURRENT deployed
            # task, so frozen_delta (folded through _folded_upto) is always either exactly
            # this task's state (task == _folded_upto) or one task behind it, with the
            # live/still-training slot's branch added on top (task == _folded_upto + 1).
            # Assert rather than silently mishandle a usage pattern that would need
            # frozen_delta to represent a DIFFERENT task's history than what's stored.
            assert task >= self._folded_upto, (
                "merge=True requested for task={} but frozen_delta already folded through "
                "task={} -- folding assumes merge=True is only ever queried at or one task "
                "ahead of the most recently folded task".format(task, self._folded_upto))
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

        q = self.q_proj(x) + self._lora_delta(x, self.lora_A_q, self.lora_B_q, self.frozen_delta_q)
        k = self.k_proj(x)
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
        self.attn = Attention_LoRA(dim, num_heads=num_heads, qkv_bias=qkv_bias,
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
        print("I'm using ViT with per-task LoRA (q,v).")
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.n_tasks = n_tasks
        # restrict trainable LoRA to the first n transformer blocks (q,v -> 2n matrices).
        # None = all blocks. Blocks >= n keep B=0 (no contribution). Applies to every
        # method that routes trainability through freeze_to_task (seqlora/olora/inflora;
        # sketchlora additionally restricts its sketch/compress via its own n_lora_blocks).
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

        ``train_a=False`` keeps the A (down-projection) matrices frozen and
        trains only the B matrices -- used by InfLoRA, which sets A analytically
        from the input subspace and learns only B.

        Also folds every slot below `task` into frozen_delta_{q,v} (a no-op unless
        a method opted in via enable_frozen_folding() -- see Attention_LoRA.fold_up_to).
        By the time task `task` starts, task `task-1`'s slot has finished training
        (and any post-processing like HiDeLoRA's momentum blend, which runs inside
        that task's own _train, strictly before this point) -- so folding
        `task - 1` here is always folding a slot that is now permanently frozen.
        """
        self.fold_frozen_slots(task - 1)
        for p in self.parameters():
            p.requires_grad = False
        train_blocks = self.blocks if self.n_lora_blocks is None else self.blocks[:self.n_lora_blocks]
        for blk in train_blocks:
            train_lists = [blk.attn.lora_B_q, blk.attn.lora_B_v]
            if train_a:
                train_lists += [blk.attn.lora_A_q, blk.attn.lora_A_v]
            for mlist in train_lists:
                for p in mlist[task].parameters():
                    p.requires_grad = True

    def add_task_slot(self):
        """Grow every block's attention module by one adapter slot (see
        Attention_LoRA.add_task_slot). Only ever called from a streaming
        Learner's _stream_begin_chunk -- the regular (non-streaming)
        task-incremental path preallocates exactly nb_tasks slots at
        construction and never calls this."""
        for blk in self.blocks:
            blk.attn.add_task_slot()
        self.n_tasks += 1

    # -- frozen-slot folding (opt-in; see Attention_LoRA.fold_up_to) -----
    def enable_frozen_folding(self):
        for attn in self.attn_modules():
            attn.enable_frozen_folding()

    def fold_frozen_slots(self, task):
        if task < 0:
            return
        for attn in self.attn_modules():
            attn.fold_up_to(task)

    def free_folded_slots(self, task):
        """FLAGGED CHANGE (2026-07-21): network-level wrapper for
        Attention_LoRA.free_folded_slot -- see that method's docstring. Opt-in,
        called explicitly only by models/inflora.py; never called from the shared
        fold_up_to/freeze_to_task path so O-LoRA/TreeLoRA/HideLoRA are unaffected."""
        if task < 0:
            return
        for attn in self.attn_modules():
            attn.free_folded_slot(task)

    # -- input-covariance collection (InfLoRA) --------------------------
    def attn_modules(self):
        return [blk.attn for blk in self.blocks]

    def set_collect(self, flag):
        for attn in self.attn_modules():
            attn.set_collect(flag)

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
    """Load a timm ViT-B/16 checkpoint into the split-qkv LoRA model."""
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
    # freeze the pretrained backbone; LoRA params are left for the learner to manage
    for name, p in model.named_parameters():
        if name in msg.missing_keys:      # LoRA (+ any uninitialised) params
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def vit_base_patch16_224_lora(pretrained=False, n_tasks=1, lora_rank=10, lora_alpha=None, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224")


def vit_base_patch16_224_in21k_lora(pretrained=False, n_tasks=1, lora_rank=10, lora_alpha=None, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224_in21k")
