"""RainbowPromptModule: the base-knowledge pool + task-conditioning + Prompt_Evolution
+ stored-prompt inference, port of RainbowPrompt/prompt.py::RainbowPrompt onto our
gated-layer scaffold (backbone/vit_rainbowprompt.py). See that file's docstring for
the confirmed simplification (the Gumbel-softmax gate in adaptive_prompting.py is
dead code, replaced by the reference's own static `self_attn_idx` config list).

**CORRECTED 2026-07-16**: an earlier pass at this port assumed `use_linear=False`
(the config file's own argparse default) was the reference's real setting -- it is
not. `RainbowPrompt/training_scripts/../run_cifar100.sh` (the actual launch script,
not the config defaults) passes `--use_linear True` for both its 10-task and
20-task CIFAR-100 commands. With `use_linear=True`, `Prompt_Evolution`
(prompt.py:113-177) is a full transformer-block-style update, not a single plain
attention step: an attention sublayer (query/key/value projected to a D2-dim
bottleneck via `query_matcher`/`key_matcher`/`value_matcher`, a *second* attention
pass along the projected-feature axis using the same q/k -- prompt.py's
`Attention_based_Transformation`, "transpose" branch, lines 124-128 -- then
`dense` projects D2 back to embed_dim), residual+LayerNorm, then an FFN sublayer
(`fc1`/`fc2`, a D1-dim bottleneck, `Task_guided_Alignment`), residual+LayerNorm.
D1/D2 scale with task count in the reference (10-task: D1=56/D2=96; 20-task:
D1=28/D2=56) -- this project's roster-wide task-split decision is 20 tasks, so
D1=28/D2=56 is what `models/rainbowprompt.py` should pass by default.

top_k=1 (confirmed default): exactly one new base-knowledge entry per task per
layer -- collapses the reference's general pool_size=n_tasks*top_k bookkeeping to
simply `base_knowledge[layer][task_id]`.

Task-id inference (test time) is BATCH-LEVEL, not per-sample: the reference sums
each candidate task's key-similarity over the WHOLE batch before taking argmax
(prompt.py's `forward`, else-branch: `sim_score = torch.sum(sim_score) /
embed_norm.shape[0]` per task, then `torch.argmax` over tasks) -- ported exactly,
not the per-sample routing HiDeLoRA's own (different) reference uses.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RainbowPromptModule(nn.Module):
    def __init__(self, num_layers, embed_dim, n_tasks, num_heads=12, length=20,
                 self_attn_idx=None, KI_iter=10, use_linear=True, D1=28, D2=56,
                 evolve_dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.num_heads = num_heads
        self.length = length
        self.self_attn_idx = set(self_attn_idx if self_attn_idx is not None else [])
        self.KI_iter = KI_iter
        self.use_linear = use_linear
        self.evolve_dropout = evolve_dropout

        if self.use_linear:
            self.query_matcher = nn.ModuleList([nn.Linear(embed_dim, D2) for _ in range(num_layers)])
            self.key_matcher = nn.ModuleList([nn.Linear(embed_dim, D2) for _ in range(num_layers)])
            self.value_matcher = nn.ModuleList([nn.Linear(embed_dim, D2) for _ in range(num_layers)])
            self.dense = nn.ModuleList([nn.Linear(D2, embed_dim) for _ in range(num_layers)])
            self.fc1 = nn.ModuleList([nn.Linear(embed_dim, D1) for _ in range(num_layers)])
            self.fc2 = nn.ModuleList([nn.Linear(D1, embed_dim) for _ in range(num_layers)])

        # top_k=1: one base-knowledge entry per task per layer, [n_tasks, length, embed_dim].
        # Init range matches the reference's own `nn.init.uniform_(p)` call
        # (tensor_matrix in prompt.py) -- PyTorch's default bounds for that call
        # are [0, 1), NOT [-1, 1]; using the wider range destabilized early
        # training (task-0 loss ~14 instead of ~ln(5)=1.6, confirmed live 2026-07-16).
        self.base_knowledge = nn.ParameterList([
            nn.Parameter(torch.empty(n_tasks, length, embed_dim).uniform_())
            for _ in range(num_layers)])
        self.base_key = nn.Parameter(torch.empty(n_tasks, embed_dim).uniform_())

        self.register_buffer('stored_prompts', torch.zeros(n_tasks, num_layers, length, embed_dim))

    @torch.no_grad()
    def add_task_slot(self):
        """Append one more task's row to base_knowledge/base_key/stored_prompts,
        matching the constructor's init convention exactly (uniform_() default
        [0,1) range, zeros for stored_prompts). Used by boundary-agnostic
        streaming (models/rainbowprompt.py::_stream_begin_chunk) when the
        adapter-fold clock advances past however many slots were preallocated
        at construction -- see the matching comment in utils/inc_net.py's
        get_backbone '_rainbowprompt' branch."""
        device, dtype = self.base_key.device, self.base_key.dtype
        for layer in range(self.num_layers):
            new_row = torch.empty(1, self.length, self.embed_dim, device=device, dtype=dtype).uniform_()
            self.base_knowledge[layer] = nn.Parameter(
                torch.cat([self.base_knowledge[layer].data, new_row], dim=0))
        new_key_row = torch.empty(1, self.embed_dim, device=device, dtype=dtype).uniform_()
        self.base_key = nn.Parameter(torch.cat([self.base_key.data, new_key_row], dim=0))
        new_stored = torch.zeros(1, self.num_layers, self.length, self.embed_dim,
                                  device=device, dtype=dtype)
        self.stored_prompts = torch.cat([self.stored_prompts, new_stored], dim=0)
        self.n_tasks += 1

    @staticmethod
    def l2n(x, dim=-1):
        return F.normalize(x, p=2, dim=dim, eps=1e-12)

    def task_conditioning(self, base_knowledge, task_key):
        """Attention-based relevance re-weighting (relation_type='attention', the
        confirmed reference default). base_knowledge: [n, length, embed_dim];
        task_key: [embed_dim]."""
        key_exp = task_key.view(1, 1, -1).expand(base_knowledge.size(0), base_knowledge.size(1), -1)
        scores = torch.matmul(base_knowledge, key_exp.transpose(-1, -2))
        scores = F.softmax(scores, dim=-1)
        return torch.matmul(scores, base_knowledge)

    def evolve_step(self, prev, curr, layer):
        """`Evolving` (prompt.py:141-155). prev/curr: [length, embed_dim] (our
        port's tensors are already squeezed of the reference's leading top_k=1
        batch dim -- see forward()'s [0] index -- so the reference's 3D
        batch-matmuls below use .transpose(-1,-2)/.t()-style 2D equivalents)."""
        d = curr.shape[-1]
        if not self.use_linear:
            attn = torch.softmax(torch.matmul(curr, prev.transpose(-1, -2)) / math.sqrt(d), dim=-1)
            out = torch.matmul(attn, prev)
            return F.layer_norm(prev + out, [d])

        # use_linear=True (the reference's actual CIFAR-100 default):
        # Attention_based_Transformation (prompt.py:114-134) -- q=curr, k=v=prev.
        q = self.query_matcher[layer](curr)   # [length, D2]
        k = self.key_matcher[layer](prev)     # [length, D2]
        v = self.value_matcher[layer](prev)   # [length, D2]

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        weights = torch.softmax(logits, dim=-1)
        out = torch.matmul(weights, v)                                    # [length, D2]

        # second attention pass along the projected-feature axis (prompt.py:124-128,
        # the "transpose" branch) -- same q/k, not re-projected.
        q_t, k_t = q.transpose(-1, -2), k.transpose(-1, -2)                # [D2, length]
        t_logits = torch.matmul(q_t, k_t.transpose(-1, -2)) / math.sqrt(q_t.shape[-1])
        t_weights = torch.softmax(t_logits, dim=-1)
        out = torch.matmul(t_weights, out.transpose(-1, -2)).transpose(-1, -2)  # [length, D2]
        attn_out = self.dense[layer](out)                                  # [length, embed_dim]

        attn_out = F.dropout(attn_out, self.evolve_dropout, training=True)
        out1 = F.layer_norm(prev + attn_out, [d])

        ffn_out = self.fc2[layer](F.relu(self.fc1[layer](out1)))
        ffn_out = F.dropout(ffn_out, self.evolve_dropout, training=True)
        return F.layer_norm(out1 + ffn_out, [d])

    def _inject_shape(self, evolved, batch_size):
        """[length, embed_dim] -> key/value prefix pair, each [B, num_heads, length/2, head_dim]."""
        half = self.length // 2
        head_dim = self.embed_dim // self.num_heads
        key_prompt = evolved[:half].view(half, self.num_heads, head_dim).permute(1, 0, 2)
        value_prompt = evolved[half:].view(half, self.num_heads, head_dim).permute(1, 0, 2)
        key_prompt = key_prompt.unsqueeze(0).expand(batch_size, -1, -1, -1)
        value_prompt = value_prompt.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return key_prompt, value_prompt

    def forward(self, layer, task_id, cls_feat, train, batch_size, known_task=None):
        """known_task: if set (TIL eval -- ground-truth task id known), skip key
        -matching entirely and retrieve that task's stored prompt directly. Only
        meaningful when train=False; train=True always (re)computes evolution."""
        if not train and known_task is not None:
            evolved = self.stored_prompts[known_task, layer]
            key_prompt, value_prompt = self._inject_shape(evolved, batch_size)
            return key_prompt, value_prompt, None

        base_knowledge = self.base_knowledge[layer]   # [n_tasks, length, embed_dim]
        if train:
            task_key = self.base_key[task_id]
            key_n = self.l2n(task_key, dim=0)
            feat_n = self.l2n(cls_feat, dim=-1)
            sim_loss = torch.matmul(key_n, feat_n.t()).sum() / feat_n.shape[0]

            curr_bk = base_knowledge[task_id:task_id + 1]                # [1, length, embed_dim]
            attended_curr = self.task_conditioning(curr_bk, key_n)[0]    # [length, embed_dim]

            if layer in self.self_attn_idx:
                p = attended_curr
                for _ in range(self.KI_iter):
                    p = self.evolve_step(p, p, layer)
                evolved = p
            elif task_id == 0:
                # Reference still calls Evolving(curr, curr) once here (the single
                # KI_layer==task_id==0 iteration of Prompt_Evolution's cross-evolve
                # loop) -- NOT a raw pass-through of the task-conditioned base
                # knowledge. Skipping this (an earlier version of this port did)
                # left task 0's prompt unnormalized and badly scaled -- confirmed
                # live: task-0 loss ~14 (vs ~ln(5)=1.6 expected) and chance-level
                # accuracy, while task 1+ (which already went through evolve_step
                # via the else branch) trained normally. Fixed 2026-07-16.
                evolved = self.evolve_step(attended_curr, attended_curr, layer)
            else:
                prev_bk = base_knowledge[0:task_id].detach()             # [task_id, length, embed_dim]
                attended_prev = self.task_conditioning(prev_bk, key_n)   # [task_id, length, embed_dim]
                evolved_list = [self.evolve_step(attended_prev[t], attended_curr, layer) for t in range(task_id)]
                evolved = torch.stack(evolved_list, dim=0).mean(dim=0)

            with torch.no_grad():
                self.stored_prompts[task_id, layer] = evolved.detach()
        else:
            key_n = self.l2n(self.base_key[:task_id + 1], dim=-1)     # [task_id+1, embed_dim]
            feat_n = self.l2n(cls_feat, dim=-1)                       # [B, embed_dim]
            sims = torch.matmul(key_n, feat_n.t()).sum(dim=1) / feat_n.shape[0]   # [task_id+1]
            pred_task = int(sims.argmax().item())
            evolved = self.stored_prompts[pred_task, layer]
            sim_loss = None

        key_prompt, value_prompt = self._inject_shape(evolved, batch_size)
        return key_prompt, value_prompt, sim_loss
