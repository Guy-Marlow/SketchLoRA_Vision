"""CountSketch merge for SketchLoRA's `merge_op=countsketch` ablation (Experiments_Timeline.pdf
sec 1.b.iii.2, plan doc §5.3.2). Vision-side port of the language reference
(svd_sketching_language/tokmem/atomic/svdlora_layer.py::_compress_countsketch) -- ported
per-(layer,proj) here (unbatched); native batching across the 24 identical-shape (768x768)
matrices per boundary is a separate optimization (plan doc §6 item 4), not done here.

Unlike the language version, vision's `lora_scaling` is a single constant shared by every slot
(sketch and every residual alike -- see models/sketchlora.py's own "unscaled factor products"
convention in _compress), so there is no per-component `sqrt(scale)` folding step to carry over;
concatenating the RAW factors is already equivalent up to the shared scalar `s`, matching what
the rand_svd path already does. The column-norm rebalancing step IS kept -- that is a property of
the sketch itself (makes hash-collision error independent of how magnitude happens to split
between a column's A and B factor), not of the scale hyperparameter.
"""

import torch


def countsketch_compress(B_list, A_list, cs_rank, seed):
    """B_list/A_list: lists of [out,r_i] / [r_i,in] weight tensors (sketch slot first, then every
    residual slot) to merge into a single rank-k factor pair, k = min(cs_rank, out, in).

    Returns (B_new, A_new), each on the same device/dtype as B_list[0].
    """
    dev, dt = B_list[0].device, B_list[0].dtype
    F_B = torch.cat([b.float() for b in B_list], dim=1)   # [out, m]
    F_A = torch.cat([a.float() for a in A_list], dim=0)   # [m, in]

    out_features, in_features = F_B.shape[0], F_A.shape[1]
    k = min(cs_rank, out_features, in_features)

    nb, na = F_B.norm(dim=0), F_A.norm(dim=1)
    keep = (nb * na) > 0
    if not bool(keep.any()):
        return (torch.zeros(out_features, k, dtype=dt, device=dev),
                torch.zeros(k, in_features, dtype=dt, device=dev))

    F_B, F_A = F_B[:, keep], F_A[keep, :]
    nb, na = nb[keep], na[keep]
    # rebalance: ||F_B[:,i]|| == ||F_A[i,:]|| == sqrt(nb_i * na_i); product unchanged
    t = torch.sqrt(na / nb)
    F_B = F_B * t.unsqueeze(0)
    F_A = F_A / t.unsqueeze(1)
    m = F_B.shape[1]

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) % (2 ** 63 - 1))
    h = torch.randint(0, k, (m,), generator=g).to(F_B.device)
    sgn = (torch.randint(0, 2, (m,), generator=g) * 2 - 1).to(F_B.dtype).to(F_B.device)

    B_new = torch.zeros(out_features, k, dtype=F_B.dtype, device=F_B.device)
    A_new = torch.zeros(k, in_features, dtype=F_A.dtype, device=F_A.device)
    B_new.index_add_(1, h, F_B * sgn.unsqueeze(0))
    A_new.index_add_(0, h, F_A * sgn.unsqueeze(1))

    return B_new.to(dt).to(dev), A_new.to(dt).to(dev)
