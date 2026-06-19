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
        self.cur_matrix = torch.zeros(dim, dim)
        self.n_cur_matrix = 0

    def set_task(self, task, merge=False):
        self._task = task
        self._merge = merge

    def set_collect(self, flag):
        self._collect = flag

    def reset_cur_matrix(self):
        self.cur_matrix = torch.zeros(self.dim, self.dim)
        self.n_cur_matrix = 0

    def _accumulate_cov(self, x):
        # x: [B, N, C] input to the q/v projections (post norm1)
        xd = x.detach()
        cov = torch.bmm(xd.permute(0, 2, 1), xd).sum(dim=0).cpu()
        n = x.shape[0] * x.shape[1]
        self.cur_matrix = (self.cur_matrix * self.n_cur_matrix + cov) / (self.n_cur_matrix + n)
        self.n_cur_matrix += n

    def _lora_delta(self, x, A_list, B_list):
        task, merge = self._task, self._merge
        if task < 0:
            return 0.0
        if merge:
            delta = 0.0
            for t in range(task + 1):
                delta = delta + B_list[t](A_list[t](x))
            return delta * self.lora_scaling
        return B_list[task](A_list[task](x)) * self.lora_scaling

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape
        if self._collect:
            self._accumulate_cov(x)

        q = self.q_proj(x) + self._lora_delta(x, self.lora_A_q, self.lora_B_q)
        k = self.k_proj(x)
        v = self.v_proj(x) + self._lora_delta(x, self.lora_A_v, self.lora_B_v)

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
                 act_layer=None, n_tasks=1, lora_rank=10, lora_alpha=None):
        super().__init__()
        print("I'm using ViT with per-task LoRA (q,v).")
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.n_tasks = n_tasks

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
        """
        for p in self.parameters():
            p.requires_grad = False
        for blk in self.blocks:
            train_lists = [blk.attn.lora_B_q, blk.attn.lora_B_v]
            if train_a:
                train_lists += [blk.attn.lora_A_q, blk.attn.lora_A_v]
            for mlist in train_lists:
                for p in mlist[task].parameters():
                    p.requires_grad = True

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
