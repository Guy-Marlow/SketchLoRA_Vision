"""Shared feature-extraction/persistence helpers for representation-drift
studies (originally inlined in models/sketchlora.py for the tsne_drift
campaign; factored out 2026-10-21 so models/olora.py can produce directly
comparable feature dumps for the olora_sketch_fd_divergence campaign without
duplicating this logic). Pure, dependency-free utility functions -- no model
state, no MRO/override concerns, safe to import from any Learner.

A caller gets a (features, labels) pair via extract_features(), using
whatever forward_fn produces that Learner's "features" key (the backbone's
pooled pre-classifier-head representation, matching tsne_drift's own usage),
then persists it via save_feature_group().
"""
import logging
import os

import numpy as np
import torch


@torch.no_grad()
def extract_features(net, loader, forward_fn, device):
    """forward_fn(net, inputs) -> output dict with a "features" key. Returns
    (features, labels) as numpy arrays, in loader order (no shuffle)."""
    net.eval()
    feats, labels = [], []
    for _, inputs, targets in loader:
        inputs = inputs.to(device)
        out = forward_fn(net, inputs)
        feats.append(out["features"].cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def save_feature_group(out_dir, fname, features, labels, **meta):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    np.savez(path, features=features, labels=labels, **meta)
    logging.info("[embed_drift] wrote {} ({} points)".format(path, features.shape[0]))
