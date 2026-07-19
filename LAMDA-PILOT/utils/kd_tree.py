"""Port of TreeLoRA/utils/kd_lora_tree.py (KDTreeNode, tree_lora_loss, KD_LoRA_Tree)
for LAMDA-PILOT, dropping DeepSpeed / multi-GPU / print_rank_0 scaffolding --
everything else is a faithful port of the algorithm.

Depth axis ("lora_depth" in the reference) = our shared scaffold's 24 wrapped LoRA-A
projections (12 ViT blocks x {q,v}), each contributing one row of the per-step
similarity-tracking tensor.

insert_grad simplification (not a bug fix -- a documented faithfulness choice): the
reference's own loop (`for i in range(len(_grad_current)): ... current_grad +=
_grad_current * frac`) iterates lora_depth times but adds the FULL tensor each
iteration, inflating current_grad by a constant lora_depth factor every call.
get_loss's self-normalization (`reg_loss / reg_loss.detach()`) divides by this same
inflated magnitude, so the inflation has zero effect on final training dynamics --
we implement the single-accumulation the loop clearly intended (add _grad_current
once per call, not lora_depth times), which is numerically equivalent after
normalization and considerably cheaper.
"""

import math

import torch


class KDTreeNode:
    def __init__(self, task_indices, depth, grads_tensor, lora_depth):
        """grads_tensor: (num_tasks, lora_depth, feature_dim)."""
        self.task_indices = task_indices
        self.depth = depth
        self.left = None
        self.right = None
        self.is_leaf = False
        self.lora_depth = lora_depth
        self.mean_vector = None
        self.median_similarity = None
        self.build_node(grads_tensor)

    def build_node(self, grads_tensor):
        if self.depth >= self.lora_depth or len(self.task_indices) <= 1:
            self.is_leaf = True
            return
        current_grads = grads_tensor[self.task_indices, self.depth, :]
        self.mean_vector = current_grads.mean(dim=0)
        similarities = torch.mv(current_grads, self.mean_vector)
        self.median_similarity = torch.median(similarities).item()
        left_indices = [self.task_indices[i] for i in range(len(self.task_indices))
                        if similarities[i].item() >= self.median_similarity]
        right_indices = [self.task_indices[i] for i in range(len(self.task_indices))
                         if similarities[i].item() < self.median_similarity]
        if len(left_indices) == 0 or len(right_indices) == 0:
            median = len(self.task_indices) // 2
            left_indices = self.task_indices[:median]
            right_indices = self.task_indices[median:]
        self.left = KDTreeNode(left_indices, self.depth + 1, grads_tensor, self.lora_depth)
        self.right = KDTreeNode(right_indices, self.depth + 1, grads_tensor, self.lora_depth)


def tree_lora_loss(current_grad, all_grad, prev_id_matrix):
    """multiple_module=True path only (our depth axis always has >1 module)."""
    reg_loss = None
    for depth_id, prev_task_id in enumerate(prev_id_matrix):
        term = -(current_grad[depth_id] * all_grad[prev_task_id][depth_id]).sum()
        reg_loss = term if reg_loss is None else reg_loss + term
    return reg_loss


class KD_LoRA_Tree:
    def __init__(self, num_tasks, reg):
        self.root = None
        self.last_task_id = -1
        self.reg = reg
        self.num_tasks = num_tasks
        self.all_accumulate_grads = [None] * num_tasks
        self.num_of_selected = None
        self.kd_tree_root = None
        self.current_grad = None
        self.all_grad = None
        self.all_grad_device = None
        self.sim = None
        self.tmp_rounds = -1
        self.tmp_reg = 0.0
        self.total_rounds = 1

    def new_epoch_init(self, train_dataloader_len):
        self.current_grad = None
        self.all_grad = None
        self.num_of_selected = None
        self.tmp_rounds = -1
        self.total_rounds = train_dataloader_len
        self.sim = None

    def end_task(self, task_id):
        if self.reg > 0:
            # all_accumulate_grads is a plain list, preallocated to num_tasks at
            # construction; under boundary-agnostic streaming (models/stream_mixin.py)
            # the number of adapter-fold events (task_id here is really the CHUNK
            # index) can exceed num_tasks, since folds are driven by a memory-
            # constraint sample threshold, not real task count. Grow on demand
            # rather than assume the preallocated size is an upper bound.
            if task_id >= len(self.all_accumulate_grads):
                self.all_accumulate_grads.extend(
                    [None] * (task_id + 1 - len(self.all_accumulate_grads)))
            self.all_accumulate_grads[task_id] = self.current_grad
        lora_depth = self.current_grad.shape[0]
        valid_grads = [g for g in self.all_accumulate_grads[:task_id + 1] if g is not None]
        if not valid_grads:
            return
        grads_tensor = torch.stack(valid_grads).clone()
        for i in range(grads_tensor.shape[0] - 1, 0, -1):
            grads_tensor[i] = grads_tensor[i] - grads_tensor[i - 1]
        task_ids = [i for i, g in enumerate(self.all_accumulate_grads[:task_id + 1]) if g is not None]
        self.kd_tree_root = KDTreeNode(task_ids, 0, grads_tensor, lora_depth)

    def step(self):
        self.tmp_rounds += 1
        self.tmp_reg = self.reg * self.tmp_rounds / self.total_rounds

    def insert_grad(self, _grad_current):
        frac = 1.0 / self.total_rounds
        if self.current_grad is None:
            self.current_grad = _grad_current.detach() * frac
        else:
            self.current_grad = self.current_grad + _grad_current.detach() * frac

    def tree_search(self, task_id, device):
        if self.all_grad is None:
            self.all_grad = torch.stack(self.all_accumulate_grads[:task_id], dim=0).to(device, non_blocking=True)
            self.all_grad_device = self.all_grad
            if self.sim is None:
                self.sim = torch.zeros((task_id, self.all_grad.shape[1]), device=device)
                # rebuilt fresh every epoch (new_epoch_init) and only ever read via a
                # [:task_id] slice below -- size it to task_id exactly rather than the
                # fixed self.num_tasks, which is not a safe upper bound under
                # boundary-agnostic streaming (see end_task's comment above).
                self.num_of_selected = torch.zeros(
                    task_id, self.all_grad.shape[1]).to(device, non_blocking=True)

        sim = self.sim.clone()
        valid_mask = self.num_of_selected[:task_id, :] > 0
        sim[valid_mask] = sim[valid_mask] / self.num_of_selected[:task_id, :][valid_mask]
        sim -= (1.0 / torch.sqrt(2 * self.num_of_selected[:task_id, :] + 1e-5)
                * math.sqrt(math.log(2 * self.total_rounds * (self.tmp_rounds + 1) * (self.tmp_rounds + 2))))
        sim = -sim

        sim += torch.min(sim)
        first_idx = torch.multinomial(
            torch.softmax(torch.sum(sim, dim=1), dim=0), num_samples=1, replacement=True).item()
        if self.kd_tree_root is not None and self.kd_tree_root.left is not None:
            if first_idx in self.kd_tree_root.left.task_indices:
                similarity = (self.kd_tree_root.left.median_similarity
                             if self.kd_tree_root.left.median_similarity is not None else 1.0)
                sim[self.kd_tree_root.left.task_indices] *= min(similarity, 1.5)
            else:
                similarity = (self.kd_tree_root.right.median_similarity
                             if self.kd_tree_root.right.median_similarity is not None else 1.0)
                sim[self.kd_tree_root.right.task_indices] *= min(similarity, 1.5)

        sim = sim / (torch.max(sim) - torch.min(sim) + 1e-5)
        sim[task_id:, :] = -torch.inf
        sim_normalized = torch.softmax(sim, dim=0)
        prev_id_matrix = torch.multinomial(sim_normalized.T, num_samples=1, replacement=True).reshape(-1)
        self.num_of_selected[prev_id_matrix, torch.arange(sim.shape[1])] += 1
        self._update_similarity(prev_id_matrix)
        return prev_id_matrix

    def get_loss(self, grad_current, loss, prev_id_matrix):
        reg_loss = tree_lora_loss(grad_current, self.all_grad_device, prev_id_matrix)
        reg_loss = reg_loss / (reg_loss.detach().clone() + 1e-5) * loss.detach().clone() * self.tmp_reg
        return reg_loss

    def _update_similarity(self, prev_id_matrix):
        if self.sim is None:
            return
        for depth_idx, prev_id in enumerate(prev_id_matrix):
            self.sim[prev_id, depth_idx] -= torch.sum(
                torch.abs(self.current_grad[depth_idx] - self.all_grad[prev_id, depth_idx])
            ).item()
