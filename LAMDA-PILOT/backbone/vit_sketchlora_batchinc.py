"""ViT backbone for the new batch-incremental + drift-detection setting
(boundary_mode TBD -- see the forthcoming models/sketchlora_batchinc.py
driver, not yet wired to trainer.py). Forked from backbone/vit_lora.py
(same Q,V-only LoRA placement, same per-task ModuleList-of-slots
scaffold, same pretrained-checkpoint loading) per this project's standing
"new experimental mechanism = isolated codepath" convention -- nothing in
vit_lora.py is modified.

WHAT'S DIFFERENT FROM EVERY EXISTING SKETCH BACKBONE: every existing
sketch method here (sdlora_sketch, lorasub_sketch, cllora_sketch) has
exactly two slots, SKETCH (fixed, folded history) and RESIDUAL (the one
live slot), and folds the instant a new slot would be needed. This
setting decouples spawning a new live adapter from folding it into the
sketch, so there can be any number (0..cap) of FROZEN-BUT-NOT-YET-FOLDED
("pending") slots sitting between the sketch and the live slot at once.

STATE: a growing nn.ModuleList of (A,B) slots, exactly like vit_lora.py's
own add_task_slot() mechanism, PLUS one fixed-size sketch (its own single
(A,B) pair at svd_rank, not a slot in the list). At any time exactly one
slot index is "live" (the newest one, the only one with trainable
parameters) and zero or more OTHER slot indices are "pending" (frozen,
once-live, not yet merged into the sketch). `_pending`/`_live_idx` are
tracked ONCE at the VisionTransformer level and broadcast down to every
block's attention module via set_routing(), exactly mirroring how
set_task()/_task/_merge already broadcast from the top in vit_lora.py --
not duplicated per-block state that could go out of sync.

FORWARD (merge is unconditional here -- this setting has no non-merged
mode): delta(x) = sketch_B(sketch_A(x)) + sum_{i in pending} B_i(A_i(x))
+ B_live(A_live(x)) * lora_scaling. Every forward always includes the
sketch, every pending slot, and the live slot -- the model being trained
in is always the model being evaluated, per the setting's own spec.

FOLD (fold_pending(), called by the driver on its own schedule/cap clock,
NEVER on the spawn clock): merges the current sketch plus every slot
currently in `pending` into a new svd_rank-sized sketch, via the exact
same QR-SVD factored truncation every existing sketch method in this
project already uses (concatenate every input factor, QR each side, a
small SVD of the resulting small matrix, truncate to svd_rank) --
generalized here from a fixed 2-way merge (one sketch + one residual) to
an N-way one (one sketch + however many slots happen to be pending),
since the concatenation step does not care how many pieces it is given.
The live slot is never part of this concatenation and is never touched by
a fold. Folded slots are freed afterward (replaced with nn.Identity, the
same convention backbone/vit_lora.py's own free_folded_slot already uses)
since nothing ever reads a pending index again once it has been folded.

ORTHOGONALITY: a soft penalty only (this project never does forced/hard
orthogonal projection anywhere, including here) -- the live slot's own
down-projection (input-side) factor against the sketch's own down-
projection factor (detached) plus one more term of the identical shape
for every slot currently in `pending` (each detached), summed. This is a
direct generalization of models/cllora_sketch.py-adjacent backbone/
vit_cllora_sketch.py::SketchQV.orth_loss's own two-term version (sketch
term + one residual term) to a variable number of pending terms. A fold
collapses however many individual pending terms existed into the single
new sketch term, so the live slot's penalty can only ever get smaller at
a fold, never larger -- no re-projection is needed or used anywhere in
this file, since the live slot's factor already satisfies whatever
(strictly looser) constraint the sketch alone represents, having already
satisfied the stricter pre-fold one.

INITIALIZATION: kaiming-uniform A, zero B for every newly spawned slot --
this project's own standing SketchLoRA-family convention (NOT LoRA-
Subtraction's reference-matched sparse_ init, which is specific to
replicating that method's own reference code and has no bearing here,
since this setting has no reference repository of its own to match).
"""
import math
from functools import partial

import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed


class Attention_LoRABatchInc(nn.Module):
    """Multi-head attention with Q,V LoRA: one fixed sketch, a growing list
    of slots (one live, zero or more pending), per the module docstring."""

    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 lora_rank=10, lora_alpha=None, svd_rank=10):
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

        self.lora_rank = lora_rank
        self.svd_rank = svd_rank
        self.lora_scaling = (lora_alpha if lora_alpha is not None else lora_rank) / lora_rank

        # the one live slot, preallocated at construction (slot 0).
        self.lora_A_q = nn.ModuleList([nn.Linear(dim, lora_rank, bias=False)])
        self.lora_B_q = nn.ModuleList([nn.Linear(lora_rank, dim, bias=False)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, lora_rank, bias=False)])
        self.lora_B_v = nn.ModuleList([nn.Linear(lora_rank, dim, bias=False)])
        nn.init.kaiming_uniform_(self.lora_A_q[0].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q[0].weight)
        nn.init.zeros_(self.lora_B_v[0].weight)

        # the one fixed sketch -- starts at an exact zero map (zero B),
        # same "nothing folded yet" convention every other sketch method
        # in this project starts from.
        self.sketch_A_q = nn.Linear(dim, svd_rank, bias=False)
        self.sketch_B_q = nn.Linear(svd_rank, dim, bias=False)
        self.sketch_A_v = nn.Linear(dim, svd_rank, bias=False)
        self.sketch_B_v = nn.Linear(svd_rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.sketch_A_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.sketch_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.sketch_B_q.weight)
        nn.init.zeros_(self.sketch_B_v.weight)

        # routing state, broadcast from VisionTransformerBatchInc.set_routing()
        # once per forward -- never independently tracked per-block.
        self._pending = []
        self._live_idx = 0
        # frozen-feature probe mode for the KL detector (see
        # VisionTransformerBatchInc.set_adapters_disabled) -- when True,
        # _delta returns exactly 0 regardless of pending/live/sketch state,
        # giving the pure frozen-base forward the detector needs.
        self._adapters_disabled = False

    def set_routing(self, pending, live_idx):
        self._pending = pending
        self._live_idx = live_idx

    def set_adapters_disabled(self, flag):
        self._adapters_disabled = flag

    @torch.no_grad()
    def add_task_slot(self):
        """Append one more (A,B) pair for q and v -- same device/dtype/init
        convention as the constructor's own slot 0 (kaiming A, zero B).
        Does NOT touch pending/live bookkeeping; that is the caller's
        (VisionTransformerBatchInc.add_task_slot's) responsibility, since
        it is tracked once at the top level, not per block."""
        ref = self.sketch_B_q.weight
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

    @torch.no_grad()
    def fold_pending(self, pending_indices):
        """Merge the fixed sketch plus every slot in `pending_indices`
        (frozen, not the live slot) into a new svd_rank-sized sketch --
        see module docstring for the exact algorithm and why it generalizes
        cleanly from this project's existing 2-way merges. Returns a dict
        {"q": retained_energy, "v": retained_energy}. Frees the folded
        slots afterward (nn.Identity placeholders)."""
        retained = {}
        for proj, A_list, B_list, sketch_A, sketch_B in (
                ("q", self.lora_A_q, self.lora_B_q, self.sketch_A_q, self.sketch_B_q),
                ("v", self.lora_A_v, self.lora_B_v, self.sketch_A_v, self.sketch_B_v)):
            A_sk, B_sk = sketch_A.weight, sketch_B.weight
            dt = B_sk.dtype
            parts_A = [A_sk.float()] + [A_list[i].weight.float() for i in pending_indices]
            parts_B = [B_sk.float()] + [B_list[i].weight.float() for i in pending_indices]

            delta_true = sum(B @ A for A, B in zip(parts_A, parts_B))
            B_cat = torch.cat(parts_B, dim=1)     # [dim, svd_rank + sum(pending ranks)]
            A_cat = torch.cat(parts_A, dim=0)     # [svd_rank + sum(pending ranks), dim]
            Q_B, R_B = torch.linalg.qr(B_cat)
            Q_A, R_A = torch.linalg.qr(A_cat.t())
            M_small = R_B @ R_A.t()
            U_small, S, Vh_small = torch.linalg.svd(M_small)
            U = Q_B @ U_small
            Vh = Vh_small @ Q_A.t()

            r = min(self.svd_rank, S.shape[0])
            sqrt_s = S[:r].sqrt()
            A_new = (sqrt_s.unsqueeze(1) * Vh[:r, :]).to(dt)
            B_new = (U[:, :r] * sqrt_s).to(dt)

            recon = B_new.float() @ A_new.float()
            fro_true = torch.linalg.norm(delta_true).clamp_min(1e-12)
            retained[proj] = (1.0 - (torch.linalg.norm(delta_true - recon) / fro_true) ** 2).item()

            sketch_A.weight.data.copy_(A_new)
            sketch_B.weight.data.copy_(B_new)

        for i in pending_indices:
            self.lora_A_q[i] = nn.Identity()
            self.lora_B_q[i] = nn.Identity()
            self.lora_A_v[i] = nn.Identity()
            self.lora_B_v[i] = nn.Identity()
        return retained

    def orth_loss(self):
        """Soft penalty only -- see module docstring. Live slot's down-
        projection factor against the sketch's (detached) plus one term per
        currently-pending slot (each detached). Zero if there is no live
        slot yet (never happens in practice -- slot 0 is always live from
        construction) or nothing to be orthogonal to yet (sketch all-zero
        and no pending slots, i.e. before the first fold -- the loss is
        still well-defined there, just numerically ~0 against an all-zero
        sketch, same as every other sketch method's own orth loss at task 0)."""
        orth = 0.0
        for A_list, sketch_A in ((self.lora_A_q, self.sketch_A_q), (self.lora_A_v, self.sketch_A_v)):
            A_live = A_list[self._live_idx].weight
            A_sketch = sketch_A.weight.detach()
            orth = orth + torch.abs(A_live @ A_sketch.t()).sum()
            for i in self._pending:
                A_other = A_list[i].weight.detach()
                orth = orth + torch.abs(A_live @ A_other.t()).sum()
        return orth

    def _delta(self, x, A_list, B_list, sketch_A, sketch_B):
        if self._adapters_disabled:
            return 0.0
        delta = sketch_B(sketch_A(x))
        for i in self._pending:
            delta = delta + B_list[i](A_list[i](x))
        delta = delta + B_list[self._live_idx](A_list[self._live_idx](x))
        return delta * self.lora_scaling

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x) + self._delta(x, self.lora_A_q, self.lora_B_q, self.sketch_A_q, self.sketch_B_q)
        k = self.k_proj(x)
        v = self.v_proj(x) + self._delta(x, self.lora_A_v, self.lora_B_v, self.sketch_A_v, self.sketch_B_v)

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
                 lora_rank=10, lora_alpha=None, svd_rank=10):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_LoRABatchInc(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop, proj_drop=drop,
                                           lora_rank=lora_rank, lora_alpha=lora_alpha, svd_rank=svd_rank)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def set_routing(self, pending, live_idx):
        self.attn.set_routing(pending, live_idx)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))
        x = residual + x
        return x


class VisionTransformerBatchInc(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, lora_rank=10, lora_alpha=None, svd_rank=10, pending_cap=4):
        super().__init__()
        print("I'm using ViT with the batch-incremental sketch+pending+live LoRA (q,v).")
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.lora_rank = lora_rank
        self.svd_rank = svd_rank
        self.pending_cap = pending_cap
        # n_tasks here means "slots ever spawned," not "real CIL tasks" --
        # kept as the name every other backbone in this project already
        # uses for this exact counter, for consistency.
        self.n_tasks = 1
        self._pending = []     # list of frozen, not-yet-folded slot indices
        self._live_idx = 0     # the one slot currently receiving gradients

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
                  act_layer=act_layer, lora_rank=lora_rank, lora_alpha=lora_alpha, svd_rank=svd_rank)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

    def attn_modules(self):
        return [blk.attn for blk in self.blocks]

    # -- spawn / fold (the two independent clocks' own mechanics; WHEN to
    # call these is the driver's decision, not this backbone's) ----------
    def add_task_slot(self):
        """Freeze the current live slot into pending (if one exists yet --
        it always does, from construction onward) and spawn a fresh live
        slot. Does NOT check or enforce pending_cap itself -- the driver
        checks len(self.pending) after calling this and triggers a fold of
        its own accord if the cap is reached, keeping the spawn clock and
        the sketch clock genuinely independent per the setting's spec."""
        if self._live_idx is not None:
            self._pending.append(self._live_idx)
        new_idx = self.n_tasks
        for blk in self.blocks:
            blk.attn.add_task_slot()
        self.n_tasks += 1
        self._live_idx = new_idx

    def fold_pending(self):
        """Fold every currently-pending slot (never the live one) into the
        sketch, for every block. Returns {(block_idx, proj): retained_energy}."""
        pending_indices = list(self._pending)
        retained = {}
        for li, attn in enumerate(self.attn_modules()):
            r = attn.fold_pending(pending_indices)
            retained[(li, "q")] = r["q"]
            retained[(li, "v")] = r["v"]
        self._pending = []
        return retained

    def orth_loss(self):
        return sum(attn.orth_loss() for attn in self.attn_modules())

    # -- routing -----------------------------------------------------------
    def set_routing(self):
        for blk in self.blocks:
            blk.set_routing(self._pending, self._live_idx)

    def set_adapters_disabled(self, flag):
        """For the KL detector's own frozen-feature probe: when flag=True,
        every block's sketch+pending+live contribution is forced to exactly
        0 regardless of their actual values, giving the pure frozen-base
        forward the detector needs. A dedicated flag rather than routing
        through an empty pending list, since the real live slot's B is not
        generally zero once trained -- unlike backbone/vit_lora.py's own
        task<0 shortcut (this backbone's forward has no task argument at
        all, it always includes sketch+pending+live), this adds the
        equivalent one explicitly. Same set_flag(bool) style as every other
        routing toggle in this project (e.g. backbone/vit_lorasub.py's own
        set_subtract_mode) -- caller is responsible for setting it back to
        False after the probe forward."""
        for attn in self.attn_modules():
            attn.set_adapters_disabled(flag)

    # -- memory accounting (live excluded, pending counted, sketch counted,
    # per the setting's own spec -- a genuine 3-way split, unlike every
    # existing sketch method's live/bank 2-way split) ---------------------
    def persistent_bytes(self):
        sketch_bytes = pending_bytes = live_bytes = 0
        for attn in self.attn_modules():
            for sketch_mod in (attn.sketch_A_q, attn.sketch_B_q, attn.sketch_A_v, attn.sketch_B_v):
                sketch_bytes += sum(p.numel() * p.element_size() for p in sketch_mod.parameters())
            for i in self._pending:
                for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                    pending_bytes += sum(p.numel() * p.element_size() for p in mlist[i].parameters())
            for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                live_bytes += sum(p.numel() * p.element_size() for p in mlist[self._live_idx].parameters())
        return {"sketch": sketch_bytes, "pending": pending_bytes, "live": live_bytes}

    # -- forward ------------------------------------------------------------
    def forward_features(self, x):
        self.set_routing()
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

    def forward(self, x):
        return self.forward_features(x)


def _load_pretrained_split_qkv(model, timm_name):
    """Identical to backbone/vit_lora.py's own helper of the same name --
    duplicated rather than imported, per this project's "isolated
    codepath" convention (the whole point is that nothing here depends on
    an existing file staying exactly as it is)."""
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


def vit_base_patch16_224_sketchlora_batchinc(pretrained=False, lora_rank=10, lora_alpha=None,
                                             svd_rank=10, pending_cap=4, **kwargs):
    model = VisionTransformerBatchInc(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                      lora_rank=lora_rank, lora_alpha=lora_alpha, svd_rank=svd_rank,
                                      pending_cap=pending_cap, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224")


def vit_base_patch16_224_in21k_sketchlora_batchinc(pretrained=False, lora_rank=10, lora_alpha=None,
                                                   svd_rank=10, pending_cap=4, **kwargs):
    model = VisionTransformerBatchInc(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                      lora_rank=lora_rank, lora_alpha=lora_alpha, svd_rank=svd_rank,
                                      pending_cap=pending_cap, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224_in21k")
