"""Equivalence test for the InfLoRA DualGPM numpy->torch port (plan doc §6 item 3).

Compares the ORIGINAL numpy/CPU update_DualGPM (pasted verbatim below, pre-port)
against the new torch/GPU version now live in models/inflora.py::Learner.update_DualGPM,
across several synthetic "tasks" (random covariance-like PSD matrices), checking the
resulting feature_list bases and project_type decisions match to float32 tolerance.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.inflora import Learner as InfLoRALearner


class _Mock:
    def __init__(self, lamb, lame, total_sessions):
        self.lamb = lamb
        self.lame = lame
        self.total_sessions = total_sessions
        self.feature_list = []
        self.project_type = []
        self._cur_task = 0


def update_DualGPM_numpy_reference(self, mat_list):
    """Verbatim pre-port numpy version (as it stood before this session's port)."""
    threshold = (self.lame - self.lamb) * self._cur_task / self.total_sessions + self.lamb
    if len(self.feature_list) == 0:
        for activation in mat_list:
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold)
            self.feature_list.append(U[:, 0:max(r, 1)])
            self.project_type.append('remove' if r < (activation.shape[0] / 2) else 'retain')
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            if self.project_type[i] == 'remove':
                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1 ** 2).sum()
                act_hat = activation - np.dot(
                    np.dot(self.feature_list[i], self.feature_list[i].transpose()), activation)
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total
                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    continue
                Ui = np.hstack((self.feature_list[i], U[:, 0:r]))
                self.feature_list[i] = Ui[:, 0:Ui.shape[0]] if Ui.shape[1] > Ui.shape[0] else Ui
            else:
                assert self.project_type[i] == 'retain'
                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1 ** 2).sum()
                act_hat = np.dot(
                    np.dot(self.feature_list[i], self.feature_list[i].transpose()), activation)
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                accumulated_sval = sval_hat / sval_total
                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval >= (1 - threshold):
                        accumulated_sval -= sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    continue
                act_feature = self.feature_list[i] - np.dot(
                    np.dot(U[:, 0:r], U[:, 0:r].transpose()), self.feature_list[i])
                Ui, Si, Vi = np.linalg.svd(act_feature)
                self.feature_list[i] = Ui[:, :self.feature_list[i].shape[1] - r]

    for i in range(len(self.feature_list)):
        if self.project_type[i] == 'remove' and \
                (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0] / 2)):
            feature = self.feature_list[i]
            U, S, V = np.linalg.svd(feature)
            self.feature_list[i] = U[:, feature.shape[1]:]
            self.project_type[i] = 'retain'
        elif self.project_type[i] == 'retain':
            assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0] / 2)


def make_psd(rng, d, rank=6):
    # Low-rank-plus-noise structure -- mirrors real activation covariances (fast
    # eigenspectrum decay), unlike a flat-spectrum pure-random Gram matrix, which
    # would blow the 'retain' subspace past d/2 within a few tasks and trip the
    # reference's own (pre-existing, not port-introduced) invariant assertion.
    B = rng.standard_normal((d, rank)).astype(np.float64)
    latent = B @ B.T
    noise = rng.standard_normal((d, d)).astype(np.float64)
    noise = noise @ noise.T * 1e-3
    return latent + noise


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    d, n_layers, n_tasks = 64, 3, 5

    ref = _Mock(lamb=0.95, lame=1.0, total_sessions=n_tasks)
    new = InfLoRALearner.__new__(InfLoRALearner)   # skip __init__, only need attrs used
    new.lamb, new.lame, new.total_sessions = 0.95, 1.0, n_tasks
    new.feature_list, new.project_type = [], []

    max_diff = 0.0
    for t in range(n_tasks):
        ref._cur_task = t
        new._cur_task = t
        mats_np = [make_psd(rng, d) for _ in range(n_layers)]
        mats_torch = [torch.tensor(m, dtype=torch.float32) for m in mats_np]
        mats_np32 = [m.astype(np.float32) for m in mats_np]

        update_DualGPM_numpy_reference(ref, mats_np32)
        InfLoRALearner.update_DualGPM(new, mats_torch)

        assert len(ref.feature_list) == len(new.feature_list)
        for i in range(len(ref.feature_list)):
            assert ref.project_type[i] == new.project_type[i], \
                "task {} layer {}: project_type mismatch {} vs {}".format(
                    t, i, ref.project_type[i], new.project_type[i])
            a = ref.feature_list[i]
            b = new.feature_list[i].numpy()
            assert a.shape == b.shape, "task {} layer {}: shape mismatch {} vs {}".format(
                t, i, a.shape, b.shape)
            # basis vectors can differ by sign per-column; compare via subspace projector U U^T
            proj_a = a @ a.T
            proj_b = b @ b.T
            diff = np.abs(proj_a - proj_b).max()
            max_diff = max(max_diff, diff)
            assert diff < 1e-3, "task {} layer {}: subspace projector diff {:.2e}".format(t, i, diff)
        print("task {} OK (max subspace-projector diff so far: {:.2e})".format(t, max_diff))

    print("ALL CHECKS PASSED. max subspace-projector diff over all tasks/layers: {:.2e}".format(max_diff))


if __name__ == "__main__":
    main()
