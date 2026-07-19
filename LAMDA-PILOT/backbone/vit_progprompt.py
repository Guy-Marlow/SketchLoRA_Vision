"""Progressive Prompts (ProgPrompt) ViT-B/16 backbone for LAMDA-PILOT.

Good-faith port of Razdaibiedina et al., "Progressive Prompts: Continual Learning
for Language Models" (ICLR 2023), svd_sketching_vision/ProgPrompt/T5_codebase/ --
the reference is T5 (encoder-decoder, soft prompts prepended to the ENCODER input
embeddings), not a ViT, so this translates the METHOD (per-task soft prompts,
progressively concatenated, prior prompts frozen) onto our ViT-B/16 CIL scaffold
rather than porting T5-specific code.

Two things pulled from the reference, verified directly rather than assumed:
  * `prefix_len=10` (t5_continual.py's per-task prompt length default).
  * Concatenation order is NEWEST-first: `self.previous_prompts = torch.concat(
    [new_prompt, self.previous_prompts], axis=0)` (t5_continual.py:319) and at
    inference `prompt = torch.concat([prompt, self.previous_prompts], axis=0)`
    (t5_continual.py:786) -- the CURRENT task's prompt always goes first, followed
    by progressively older ones.

One thing NOT ported: the reference's `lr=0.3` (T5_codebase configs). That value is
calibrated for T5's embedding scale/AdamW-no-clipping setup; blindly transplanting
an LR across architectures without the reference's own validation is exactly the
kind of mistake that caused RainbowPrompt's silent divergence earlier this session
-- ProgPrompt's Learner therefore uses this project's own validated prompt-tuning
LR range (L2P/DualPrompt's existing exps/*.json: ~1e-3-2e-3), not a transplanted
T5 value. See models/progprompt.py's docstring.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import PatchEmbed


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp_drop(self.fc2(self.act(self.fc1(self.norm2(x)))))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=None, act_layer=None, n_tasks=1, prompt_len=10):
        super().__init__()
        print("I'm using ViT with per-task progressive soft prompts.")
        self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.n_tasks = n_tasks
        self.prompt_len = prompt_len

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        # per-task soft prompts, each [prompt_len, embed_dim]; init matches the
        # reference's own embedding-scale init (small random values, not the
        # pretrained token-embedding sampling the T5 reference also supports as an
        # option -- ViT has no token embedding table to sample from).
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.empty(prompt_len, embed_dim)) for _ in range(n_tasks)])
        for p in self.prompts:
            nn.init.normal_(p, std=0.02)

    @torch.no_grad()
    def add_task_slot(self):
        """Append one more per-task soft prompt, matching the constructor's
        device/dtype/init convention exactly. Used by boundary-agnostic
        streaming (models/progprompt.py::_stream_begin_chunk) when the
        adapter-fold clock advances past however many slots were preallocated
        at construction -- see the matching comment in utils/inc_net.py's
        get_backbone '_progprompt' branch."""
        ref = self.prompts[0]
        new_p = nn.Parameter(torch.empty(self.prompt_len, self.embed_dim,
                                          device=ref.device, dtype=ref.dtype))
        nn.init.normal_(new_p, std=0.02)
        self.prompts.append(new_p)
        self.n_tasks += 1

    def forward_features(self, x, task=-1):
        B = x.shape[0]
        h = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1) + self.pos_embed
        h = self.pos_drop(h)

        if task >= 0:
            # newest-first progressive concat (t5_continual.py:319,786): task's own
            # prompt goes first, then task-1, task-2, ..., 0.
            prompt_seq = torch.cat([self.prompts[t] for t in range(task, -1, -1)], dim=0)
            prompt_seq = prompt_seq.unsqueeze(0).expand(B, -1, -1)
            h = torch.cat((prompt_seq, h), dim=1)
            cls_idx = prompt_seq.shape[1]
        else:
            cls_idx = 0

        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return h[:, cls_idx]

    def forward(self, x, task=-1):
        return self.forward_features(x, task=task)


def _load_pretrained(model, timm_name):
    checkpoint_model = timm.create_model(timm_name, pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            state_dict[key.replace('mlp.', '')] = state_dict.pop(key)
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    for name, p in model.named_parameters():
        p.requires_grad = name in msg.missing_keys
    return model


def vit_base_patch16_224_progprompt(pretrained=False, n_tasks=1, prompt_len=10, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              n_tasks=n_tasks, prompt_len=prompt_len, **kwargs)
    return _load_pretrained(model, "vit_base_patch16_224")
