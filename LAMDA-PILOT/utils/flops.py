"""Analytic inference-FLOPs-per-image, via PyTorch's built-in profiler
(torch.profiler with_flops=True) -- NOT empirical wall-clock timing, and NOT
any expensive per-batch decomposition (no SVDs, no rank computations, nothing
beyond a single forward pass). Every standard op the profiler already knows how
to FLOP-count (aten::linear, aten::matmul, aten::bmm, aten::addmm, aten::conv2d)
fires naturally during a method's own real _deployed_forward call -- so this
works identically for every method (LoRA-scaffold, prompt-based, prototype-
based) with zero per-architecture bespoke formulas. Whatever ops actually run
for that method's real deployed inference path are exactly what gets counted,
which is why methods with genuinely different inference cost (e.g. TUNA's
per-task-count-growing entropy ensemble vs. a folded O(1) LoRA merge) are
measured faithfully rather than approximated by a single shared formula.
"""

import logging

import torch


def measure_inference_flops(learner, sample_input):
    """Total FLOPs for one forward pass of learner._deployed_forward(sample_input).
    Caller divides by sample_input.shape[0] for a per-image figure (matching
    metrics_logger.py's ms_per_image convention). Returns None (logged, not
    fatal) if _deployed_forward is absent or profiling fails for any reason --
    this metric is opportunistic per Experiments_Timeline.pdf's "where
    computationally feasible" scope, not a hard requirement for every method."""
    if not hasattr(learner, "_deployed_forward"):
        logging.warning("[flops] {} has no _deployed_forward; skipping FLOPs "
                        "measurement".format(type(learner).__name__))
        return None
    learner._network.eval()
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.no_grad(), torch.profiler.profile(
                activities=activities, with_flops=True) as prof:
            learner._deployed_forward(sample_input)
        total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops)
        return int(total_flops) if total_flops > 0 else None
    except Exception as e:
        logging.warning("[flops] measurement failed ({}); skipping".format(e))
        return None
