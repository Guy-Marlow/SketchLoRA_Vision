#!/usr/bin/env python3
"""Verification tests for SD-LoRA-Sketch (backbone/vit_sdlora_sketch.py),
required by the implementation brief BEFORE any real training run. No
GPU/pretrained backbone needed -- exercises Attention_SDLoRASketch
directly, on small synthetic dims.

Checks (brief's own list, same order):
  1. Without truncation, the sketch after merging equals the explicit sum
     of all scaled adapters.
  2. Without truncation, model outputs are identical immediately before
     and after a merge.
  3. The reported discarded energy equals the true reconstruction error,
     computed densely on a small layer.
  4. A known rescaling with a silenced adapter produces exactly the
     rescaled singular values.
  5. Sketch directions stay orthonormal after many synthetic merges.
  6. The rescaling gradient matches a dense-weight reference on one layer.

Run: python tests/check_sdlora_sketch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.nn import functional as F

from backbone.vit_sdlora_sketch import Attention_SDLoRASketch

torch.manual_seed(1993)
ATOL, RTOL = 1e-6, 1e-5
DIM, HEADS, LORA_RANK = 16, 4, 3


def _dense(U, S, Vh):
    return (U * S) @ Vh


def _set_random_adapter(attn, proj, scale=1.0):
    A = getattr(attn, "lora_A_{}".format(proj))[0].weight
    B = getattr(attn, "lora_B_{}".format(proj))[0].weight
    with torch.no_grad():
        A.copy_(torch.randn_like(A) * scale)
        B.copy_(torch.randn_like(B) * scale)


def _adapter_contrib_dense(attn, proj):
    """Dense reference of _new_adapter_term's actual contribution, formed
    independently (not reusing any of the module's own helper code)."""
    A = getattr(attn, "lora_A_{}".format(proj))[0].weight
    B = getattr(attn, "lora_B_{}".format(proj))[0].weight
    mag = getattr(attn, "mag_{}".format(proj))
    BA = B @ A
    norm = BA.norm().clamp_min(1e-12)
    return (mag / norm) * BA


def test_1_exact_merge_no_truncation():
    """r_hat generous relative to lora_rank, across 3 folds -- the merged
    sketch delta must equal sketch_old_rescaled + adapter_scaled exactly."""
    svd_rank = 12  # >= 3 * lora_rank, so 3 folds never exceed capacity
    attn = Attention_SDLoRASketch(DIM, num_heads=HEADS, lora_rank=LORA_RANK, svd_rank=svd_rank)
    for _ in range(3):
        _set_random_adapter(attn, "q", scale=0.5)
        with torch.no_grad():
            attn.rescale_q.copy_(torch.randn(svd_rank) * 0.3)
            attn.mag_q.copy_(torch.randn(1))

        U_old, S_old, Vh_old = (attn.sketch_U_q.clone(), attn.sketch_S_q.clone(), attn.sketch_Vh_q.clone())
        S_eff = attn._apply_rescale(S_old, attn.rescale_q.detach())
        sketch_old_dense = _dense(U_old, S_eff, Vh_old)
        adapter_dense = _adapter_contrib_dense(attn, "q").detach()
        expected = (sketch_old_dense + adapter_dense).double()

        attn.fold()

        actual = _dense(attn.sketch_U_q, attn.sketch_S_q, attn.sketch_Vh_q).double()
        err = (expected - actual).norm() / expected.norm().clamp_min(1e-12)
        assert err < 1e-5, "test 1 failed: relative merge error {:.2e}".format(err.item())
    print("test 1 (exact merge, no truncation) OK")


def test_2_output_unchanged_across_merge():
    svd_rank = 12
    attn = Attention_SDLoRASketch(DIM, num_heads=HEADS, lora_rank=LORA_RANK, svd_rank=svd_rank)
    _set_random_adapter(attn, "v", scale=0.5)
    with torch.no_grad():
        attn.rescale_v.copy_(torch.randn(svd_rank) * 0.3)
        attn.mag_v.copy_(torch.randn(1))

    x = torch.randn(5, 7, DIM)
    with torch.no_grad():
        out_before = attn._sketch_term(x, "v") + attn._new_adapter_term(x, "v")
        attn.fold()
        out_after = attn._sketch_term(x, "v") + attn._new_adapter_term(x, "v")

    err = (out_before - out_after).norm() / out_before.norm().clamp_min(1e-12)
    assert err < 1e-4, "test 2 failed: relative output drift {:.2e}".format(err.item())
    print("test 2 (output unchanged across a lossless merge) OK")


def test_3_reported_energy_matches_true_reconstruction():
    """Deliberately small svd_rank so the fold actually truncates."""
    svd_rank = 2
    attn = Attention_SDLoRASketch(DIM, num_heads=HEADS, lora_rank=LORA_RANK, svd_rank=svd_rank)
    # pre-populate the sketch with one lossless fold first, so there is
    # real rank-bearing history to truncate against on the second fold.
    _set_random_adapter(attn, "q", scale=0.5)
    with torch.no_grad():
        attn.mag_q.copy_(torch.ones(1))
    attn.fold()

    _set_random_adapter(attn, "q", scale=0.5)
    with torch.no_grad():
        attn.rescale_q.copy_(torch.randn(svd_rank) * 0.4)
        attn.mag_q.copy_(torch.randn(1))

    U_old, S_old, Vh_old = (attn.sketch_U_q.clone().double(), attn.sketch_S_q.clone().double(),
                             attn.sketch_Vh_q.clone().double())
    S_eff = (S_old * (1.0 + attn.rescale_q.detach().double()))
    true_delta = _dense(U_old, S_eff, Vh_old) + _adapter_contrib_dense(attn, "q").detach().double()

    retained = attn.fold()

    recon = _dense(attn.sketch_U_q.double(), attn.sketch_S_q.double(), attn.sketch_Vh_q.double())
    true_retained = 1.0 - ((true_delta - recon).norm() / true_delta.norm().clamp_min(1e-12)) ** 2
    assert abs(retained["q"] - true_retained.item()) < 1e-4, (
        "test 3 failed: reported={:.6f} true={:.6f}".format(retained["q"], true_retained.item()))
    print("test 3 (reported vs. true retained energy) OK -- retained={:.4f}".format(retained["q"]))


def test_4_silenced_adapter_reproduces_rescaled_singular_values():
    svd_rank = 5
    attn = Attention_SDLoRASketch(DIM, num_heads=HEADS, lora_rank=LORA_RANK, svd_rank=svd_rank)
    # first fold: populate a real orthonormal (U,S,Vh) from a random adapter.
    _set_random_adapter(attn, "v", scale=0.7)
    with torch.no_grad():
        attn.mag_v.copy_(torch.ones(1))
    attn.fold()
    S_old = attn.sketch_S_v.clone()

    # second fold: silence the adapter (mag=0 -> its contribution is exactly
    # 0 regardless of A/B), apply a KNOWN rescale.
    _set_random_adapter(attn, "v", scale=0.7)  # irrelevant: about to be silenced
    known_rescale = torch.tensor([0.5, -0.3, 0.0, -0.9, 2.0][:svd_rank])
    with torch.no_grad():
        attn.rescale_v.copy_(known_rescale)
        attn.mag_v.zero_()
    attn.fold()

    expected = torch.sort((S_old * (1.0 + known_rescale)).abs(), descending=True).values
    actual = torch.sort(attn.sketch_S_v, descending=True).values
    err = (expected - actual).abs().max()
    assert err < 1e-3, "test 4 failed: max abs diff {:.2e}\nexpected={}\nactual={}".format(
        err.item(), expected, actual)
    print("test 4 (silenced adapter -> exactly rescaled singular values) OK")


def test_5_orthonormality_after_many_merges():
    svd_rank = 4
    attn = Attention_SDLoRASketch(DIM, num_heads=HEADS, lora_rank=LORA_RANK, svd_rank=svd_rank)
    for _ in range(20):
        _set_random_adapter(attn, "q", scale=0.6)
        with torch.no_grad():
            attn.rescale_q.copy_(torch.randn(svd_rank) * 0.5)
            attn.mag_q.copy_(torch.randn(1))
        attn.fold()

    U, Vh = attn.sketch_U_q, attn.sketch_Vh_q
    UtU = U.t() @ U
    VVt = Vh @ Vh.t()
    eye = torch.eye(svd_rank)
    err_u = (UtU - eye).abs().max()
    err_v = (VVt - eye).abs().max()
    assert err_u < 1e-3, "test 5 failed: U^T U deviates from I by {:.2e}".format(err_u.item())
    assert err_v < 1e-3, "test 5 failed: Vh Vh^T deviates from I by {:.2e}".format(err_v.item())
    print("test 5 (orthonormality after 20 synthetic merges) OK")


def test_6_rescale_gradient_matches_dense_reference():
    svd_rank = 5
    attn = Attention_SDLoRASketch(DIM, num_heads=HEADS, lora_rank=LORA_RANK, svd_rank=svd_rank)
    _set_random_adapter(attn, "q", scale=0.5)
    with torch.no_grad():
        attn.mag_q.copy_(torch.ones(1))
    attn.fold()  # populate a real sketch to rescale

    x = torch.randn(3, 6, DIM)

    # module's own path
    attn.rescale_q.grad = None
    out_module = attn._sketch_term(x, "q") + attn._new_adapter_term(x, "q")
    loss_module = (out_module ** 2).sum()
    loss_module.backward()
    grad_module = attn.rescale_q.grad.clone()

    # dense reference: materialize the [dim,dim] delta explicitly and
    # recompute the identical scalar loss through it, on a FRESH leaf
    # tensor so autograd tracks only this path.
    rescale_ref = attn.rescale_q.detach().clone().requires_grad_(True)
    U, S, Vh = attn.sketch_U_q.detach(), attn.sketch_S_q.detach(), attn.sketch_Vh_q.detach()
    S_eff = S * (1.0 + rescale_ref)
    dense_delta = (U * S_eff) @ Vh                    # [dim, dim]
    sketch_out = F.linear(x, dense_delta)             # x @ dense_delta^T
    adapter_out = attn._new_adapter_term(x, "q").detach()
    out_ref = sketch_out + adapter_out
    loss_ref = (out_ref ** 2).sum()
    loss_ref.backward()
    grad_ref = rescale_ref.grad

    err = (grad_module - grad_ref).abs().max() / grad_ref.abs().max().clamp_min(1e-12)
    assert err < 1e-3, "test 6 failed: relative grad diff {:.2e}\nmodule={}\nref={}".format(
        err.item(), grad_module, grad_ref)
    print("test 6 (rescale gradient matches dense reference) OK")


if __name__ == "__main__":
    test_1_exact_merge_no_truncation()
    test_2_output_unchanged_across_merge()
    test_3_reported_energy_matches_true_reconstruction()
    test_4_silenced_adapter_reproduces_rescaled_singular_values()
    test_5_orthonormality_after_many_merges()
    test_6_rescale_gradient_matches_dense_reference()
    print("\nALL SD-LoRA-Sketch CHECKS PASSED")
