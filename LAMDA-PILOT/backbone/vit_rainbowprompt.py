"""RainbowPrompt ViT-B/16 backbone for LAMDA-PILOT.

Port of Hong et al., "RainbowPrompt: Diversity-Enhanced Prompt-Evolving for
Continual Learning" (ICCV 2025), svd_sketching_vision/RainbowPrompt/. That repo is
built on the DualPrompt-pytorch codebase (its own README's acknowledgement) -- the
K/V prefix-tuning injection here (`Attention.forward`'s `prompt` argument) is the
same mechanism our existing `backbone/vit_dualprompt.py` already implements.

Two simplifications confirmed against the ACTUAL reference configs/call graph
(not assumed from the paper text):
  * `adaptive_prompting.py`'s Gumbel-softmax per-layer gate (`soft_gate_dict`,
    `train_sample_policy`, `backward_policy`, etc.) is NEVER CALLED anywhere in
    engine.py/main.py/models.py -- dead code in the reference. The method as
    actually run uses a STATIC config list (`self_attn_idx`) to fix which layers
    self-evolve vs. cross-task-evolve; ported as a plain config list, no learned
    gate.
  * `use_linear=False` is every reference config's actual default (the simpler,
    no-projection attention path in `Prompt_Evolution`/`Evolving` -- direct
    scaled-dot-product attention, no query/key/value Linear matchers, no FFN
    step). Ported as the ONLY path (the `use_linear=True` branch, an ablation
    option never exercised by any shipped config, is omitted).

`top_k=1` (one new base-knowledge entry per task per layer, confirmed the actual
default in every reference config) and `length=20` (prefix length, split 10/10
into key-prefix/value-prefix) are used directly, not re-derived.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import PatchEmbed


class Attention(nn.Module):
    """Standard multi-head attention with an optional K/V prefix (prompt) --
    same mechanism as vit_dualprompt.py's PreT_Attention, split q/k/v projections."""
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, t, seq_len, bsz):
        return t.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x, prompt=None):
        B, N, C = x.shape
        q = self._shape(self.q_proj(x), N, B)
        k = self._shape(self.k_proj(x), N, B)
        v = self._shape(self.v_proj(x), N, B)
        if prompt is not None:
            key_prefix, value_prefix = prompt   # each [B, num_heads, prefix_len, head_dim]
            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)
        q = q.reshape(B * self.num_heads, -1, self.head_dim)
        k = k.reshape(B * self.num_heads, -1, self.head_dim)
        v = v.reshape(B * self.num_heads, -1, self.head_dim)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.bmm(attn, v).view(B, self.num_heads, N, self.head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out


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

    def forward(self, x, prompt=None):
        x = x + self.attn(self.norm1(x), prompt=prompt)
        x = x + self.mlp_drop(self.fc2(self.act(self.fc1(self.norm2(x)))))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=None, act_layer=None, prompt_module=None):
        super().__init__()
        print("I'm using ViT with RainbowPrompt (evolving base-knowledge prefixes).")
        self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

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
        self.num_heads = num_heads

        self.prompt_module = prompt_module   # RainbowPromptModule, set post-construction

    def _cls_feature_pass(self, x):
        """One frozen-backbone forward (no prompts) to get the CLS-token feature
        used for task-key matching -- the reference's `cls_features` input,
        computed from the ORIGINAL model in the paper's terminology; here it's
        simply an un-prompted forward through this same backbone (consistent with
        HiDeLoRA's analogous "un-adapted forward" trick elsewhere in this project)."""
        B = x.shape[0]
        h = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1) + self.pos_embed
        h = self.pos_drop(h)
        for blk in self.blocks:
            h = blk(h, prompt=None)
        h = self.norm(h)
        return h[:, 0]

    def forward(self, x, task_id=None, train=False, known_task=None):
        B = x.shape[0]
        with torch.no_grad():
            cls_feat = self._cls_feature_pass(x)

        sim_loss = 0.0
        h = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1) + self.pos_embed
        h = self.pos_drop(h)
        for i, blk in enumerate(self.blocks):
            key_p, value_p, layer_sim = self.prompt_module(
                i, task_id, cls_feat, train=train, batch_size=B, known_task=known_task)
            if layer_sim is not None:
                sim_loss = sim_loss + layer_sim
            h = blk(h, prompt=(key_p, value_p))
        h = self.norm(h)
        return {"features": h[:, 0], "sim_loss": sim_loss}


def _load_pretrained(model, timm_name):
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
        p.requires_grad = name in msg.missing_keys
    return model


def vit_base_patch16_224_rainbowprompt(pretrained=False, prompt_module=None, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              prompt_module=prompt_module, **kwargs)
    return _load_pretrained(model, "vit_base_patch16_224")
