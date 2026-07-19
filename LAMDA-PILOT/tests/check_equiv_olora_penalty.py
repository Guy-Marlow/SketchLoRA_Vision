#!/usr/bin/env python3
"""Equivalence test for the O-LoRA penalty vectorization (plan doc sec 6 item 1).

Verifies models/olora.py::Learner._orth_and_l2's vectorized cross-task penalty
against a naive per-pair loop, on synthetic data -- no real backbone/GPU needed
(the method only touches self._network.attn_modules() and self._cur_task, so a
minimal fake stands in for both). Also exercises the cache invalidation path
(sweeping task indices forward, then jumping BACK to an earlier index out of
order, which must rebuild rather than reuse a stale stacked-A from a later task).

Run: python tests/check_equiv_olora_penalty.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.olora import Learner as OLoRALearner

torch.manual_seed(1993)

ATOL, RTOL = 1e-5, 1e-4


class _FakeWeight:
    def __init__(self, tensor):
        self.weight = tensor


class _FakeAttn:
    def __init__(self, n_slots, r, dim):
        self.lora_A_q = [_FakeWeight(torch.randn(r, dim)) for _ in range(n_slots)]
        self.lora_B_q = [_FakeWeight(torch.randn(dim, r)) for _ in range(n_slots)]
        self.lora_A_v = [_FakeWeight(torch.randn(r, dim)) for _ in range(n_slots)]
        self.lora_B_v = [_FakeWeight(torch.randn(dim, r)) for _ in range(n_slots)]


class _FakeNetwork:
    def __init__(self, n_blocks, n_slots, r, dim):
        self._attns = [_FakeAttn(n_slots, r, dim) for _ in range(n_blocks)]

    def attn_modules(self):
        return self._attns


class _FakeLearner:
    """Minimal stand-in exposing exactly what _orth_and_l2 touches."""
    def __init__(self, network, cur_task):
        self._network = network
        self._cur_task = cur_task


def naive_orth_and_l2(network, cur_task):
    """Reference: the ORIGINAL per-pair loop (pre-vectorization), reimplemented
    independently here (not imported) so this test doesn't just check the code
    against itself."""
    t = cur_task
    orth = 0.0
    l2 = 0.0
    for attn in network.attn_modules():
        for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q), (attn.lora_A_v, attn.lora_B_v)):
            A_t = A_list[t].weight
            l2 = l2 + torch.norm(A_t, p=2) + torch.norm(B_list[t].weight, p=2)
            for s in range(t):
                A_s = A_list[s].weight
                orth = orth + torch.abs(A_t @ A_s.t()).sum()
    return orth, l2


def run():
    n_blocks, n_slots, r, dim = 3, 12, 8, 32
    network = _FakeNetwork(n_blocks, n_slots, r, dim)

    failures = []
    # Sweep forward, then jump BACK to an earlier index out of order, to make
    # sure the per-task cache invalidates rather than silently reusing a stale
    # stacked-A built for a later task.
    task_sequence = list(range(1, n_slots)) + [2, n_slots - 1]
    for t in task_sequence:
        fake_self = _FakeLearner(network, t)
        orth_vec, l2_vec = OLoRALearner._orth_and_l2(fake_self)
        orth_naive, l2_naive = naive_orth_and_l2(network, t)

        orth_ok = torch.allclose(torch.as_tensor(orth_vec), torch.as_tensor(orth_naive), atol=ATOL, rtol=RTOL)
        l2_ok = torch.allclose(torch.as_tensor(l2_vec), torch.as_tensor(l2_naive), atol=ATOL, rtol=RTOL)
        status = "OK" if (orth_ok and l2_ok) else "FAIL"
        print(f"  task={t:2d}  orth: vec={float(orth_vec):.6f} naive={float(orth_naive):.6f}  "
              f"l2: vec={float(l2_vec):.6f} naive={float(l2_naive):.6f}  [{status}]")
        if not (orth_ok and l2_ok):
            failures.append(t)

    if failures:
        print(f"\nFAILED at task indices: {failures}")
        sys.exit(1)
    print("\nAll checks passed (vectorized O-LoRA penalty == naive per-pair loop, "
          f"atol={ATOL}, rtol={RTOL}), including out-of-order cache invalidation.")


if __name__ == "__main__":
    run()
