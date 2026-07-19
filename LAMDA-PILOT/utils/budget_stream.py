"""Class-incremental streaming under a fixed per-adapter memory budget, decoupled
from the dataset's init_cls/increment task groups (Experiments_Timeline.pdf, image
headline result).  Design settled with the user 2026-07-16:

  * Stream order: class-CONTIGUOUS in the underlying DataManager's already-seeded
    class order (never interleaved across classes -- a strict reading of
    "class-incremental"). Within a class, images are drawn in a FIXED per-class
    seeded permutation (independent of ImageFolder/pickle incidental ordering),
    so which images fall in "the first half of class B" is reproducible.
  * Chunking: walk the flat class-contiguous stream, accumulating
    BYTES_PER_IMAGE (224x224x3 uint8 -- the agreed "model-input tensor bytes"
    accounting convention, NOT the actual normalized float32 runtime tensor)
    per image; close a chunk once the accumulated size >= the memory budget, or
    the stream ends (final chunk may be smaller).
  * A class's classifier-head slot grows on its FIRST appearance in a chunk. If
    that class's tail spills into the next chunk, get_task_size returns 0 NEW
    classes for the images that are pure continuation.
  * Training: 10 epochs per chunk (the project's normal LoRA-family convention),
    each epoch a fresh but SEEDED reshuffle of the chunk's fixed membership,
    IDENTICAL across every method run at the same (seed, budget, dataset) --
    achieved non-invasively by reseeding torch's global RNG once per chunk from
    the trainer.py loop (see _run_budget), so every Learner's un-modified
    `DataLoader(..., shuffle=True)` draws the same reproducible epoch sequence.
    No class reweighting: plain CE over the chunk's active class range, matching
    every existing method file.

This is a drop-in replacement for utils.data_manager.DataManager exposing only the
interface trainer.py's task loop and every Learner's incremental_train use
(nb_tasks, nb_classes, get_task_size, get_dataset) -- "tasks" become memory chunks.
TEST-source queries always delegate to the wrapped real DataManager (test sets stay
class-complete; only train-time exposure is memory-constrained), so TIL evaluation
(models/til_base.py) needs no changes: task ranges are still real, cumulative class
index ranges built from get_task_size's return values.
"""

import logging

import numpy as np
from torchvision import transforms

from utils.data_manager import DummyDataset

# uint8 model-input accounting: 224x224x3, the size actually agreed with the user
# for the memory-budget convention -- NOT the post-normalize float32 runtime tensor.
BYTES_PER_IMAGE = 224 * 224 * 3


class BudgetStreamManager:
    def __init__(self, data_manager, budget_mb, seed):
        self.args = data_manager.args
        self._dm = data_manager
        self.budget_mb = budget_mb
        self.budget_bytes = budget_mb * 1024 * 1024
        self.seed = seed
        self._nb_classes = data_manager.nb_classes
        self._active_chunk = None
        self._build_chunks()

    # ---- DataManager-compatible interface ----
    @property
    def nb_tasks(self):
        return len(self._chunks)

    @property
    def nb_classes(self):
        return self._nb_classes

    def get_task_size(self, task):
        self._active_chunk = task
        return self._new_classes_per_chunk[task]

    def ce_range(self, task):
        """The CE-loss local slice for chunk `task`: [min(chunk_targets),
        max(chunk_targets)+1). Distinct from (and generally WIDER than) the
        growth-based [known_classes, total_classes) range every Learner's
        _known_classes/_total_classes counters track: a chunk containing a
        carryover TAIL of the previous chunk's straddling class has raw labels
        BELOW that chunk's known_classes (that class's head slot already grew in
        an earlier chunk). Task-local CE using the growth-based range produces
        out-of-bounds targets for that carryover data (a real bug caught by a
        CUDA device-side assert during the first budget-mode smoke test,
        2026-07-16) -- the loss's class-competition set must match exactly what
        is actually present in the buffer, not the head-growth bookkeeping.
        Does NOT affect TIL's _task_ranges (models/til_base.py), which correctly
        stays growth-based -- that bookkeeping answers a different question
        ("which classes did this chunk introduce") than this one ("which classes
        are actually in this chunk's training buffer")."""
        _, targets = self._chunks[task]
        return int(targets.min()), int(targets.max()) + 1

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False, m_rate=None):
        if source == "test" or mode != "train":
            # test sets (and any non-train mode, e.g. "flip") stay class-complete:
            # delegate to the real DataManager unchanged. `indices` here IS a
            # real class-index range (e.g. TIL's per-task [lo,hi) from
            # get_task_size's cumulative bookkeeping), so this is correct as-is.
            return self._dm.get_dataset(indices, source, mode, appendent=appendent,
                                        ret_data=ret_data, m_rate=m_rate)
        # train: `indices` (a class-index range) does NOT apply -- a chunk may
        # hold only a FRACTION of a class's images, which class-index semantics
        # cannot express. Return the precomputed fixed membership of whichever
        # chunk was most recently named via get_task_size (called immediately
        # before this, once per task-loop iteration, by every Learner).
        assert appendent is None, "BudgetStreamManager: appendent unsupported in train mode"
        assert self._active_chunk is not None, \
            "get_dataset(train) called before get_task_size set the active chunk"
        data, targets = self._chunks[self._active_chunk]
        trsf = transforms.Compose([*self._dm._train_trsf, *self._dm._common_trsf])
        ds = DummyDataset(data, targets, trsf, self._dm.use_path)
        if ret_data:
            return data, targets, ds
        return ds

    # ---- chunk construction ----
    def _build_chunks(self):
        dm = self._dm
        x, y = dm._train_data, dm._train_targets
        # targets are ALREADY remapped into the DataManager's seeded class_order
        # index space (utils/data_manager.py::_setup_data -> _map_new_class_index),
        # so iterating 0..nb_classes-1 in numeric order IS class-order sequence.
        classes = list(range(self._nb_classes))

        flat_data, flat_targets = [], []
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                continue
            rs = np.random.RandomState((self.seed * 1_000_003 + c) % (2 ** 31 - 1))
            perm = rs.permutation(len(idx))
            flat_data.append(x[idx[perm]])
            flat_targets.append(y[idx[perm]])
        flat_data = np.concatenate(flat_data)
        flat_targets = np.concatenate(flat_targets)

        chunks, new_classes_per_chunk = [], []
        n = len(flat_targets)
        ptr = 0
        seen_classes = set()
        while ptr < n:
            start = ptr
            acc_bytes = 0
            while ptr < n and acc_bytes < self.budget_bytes:
                acc_bytes += BYTES_PER_IMAGE
                ptr += 1
            c_data = flat_data[start:ptr]
            c_targets = flat_targets[start:ptr]
            new_in_chunk = len(set(c_targets.tolist()) - seen_classes)
            seen_classes.update(c_targets.tolist())
            chunks.append((c_data, c_targets))
            new_classes_per_chunk.append(new_in_chunk)

        self._chunks = chunks
        self._new_classes_per_chunk = new_classes_per_chunk
        self._validate()
        logging.info(
            "[budget-stream] {}MB budget -> {} chunks, {} images/chunk (last={}), "
            "new-classes/chunk min={} max={} mean={:.2f}".format(
                self.budget_mb, len(chunks), len(chunks[0][1]), len(chunks[-1][1]),
                min(new_classes_per_chunk), max(new_classes_per_chunk),
                sum(new_classes_per_chunk) / len(new_classes_per_chunk)))

    def _validate(self):
        zero_chunks = [i for i, k in enumerate(self._new_classes_per_chunk) if k == 0]
        if zero_chunks:
            scenario = self.args.get("scenario", "cil")
            msg = (
                "BudgetStreamManager: {} chunk(s) introduce ZERO new classes (indices "
                "{}{}) at budget={:.0f}MB, dataset nb_classes={}. A chunk with zero new "
                "classes gives it an empty (lo, lo) TIL task range (models/til_base.py::"
                "_eval_til) -- meaningless under TIL, since TIL requires known task "
                "boundaries/counts that this memory-budget streaming setting doesn't "
                "have in the first place. Harmless for CIL: eval_task() only calls "
                "_eval_til() when scenario is 'til'/'both' (til_base.py:65-67), so an "
                "empty range is simply never read.".format(
                    len(zero_chunks), zero_chunks[:10],
                    "..." if len(zero_chunks) > 10 else "",
                    self.budget_bytes / 1024 / 1024, self._nb_classes))
            if scenario in ("til", "both"):
                raise RuntimeError(msg + " Report this (dataset, budget) combination "
                                          "-- do not silently skip/merge chunks.")
            logging.info(msg)
        total_new = sum(self._new_classes_per_chunk)
        assert total_new == self._nb_classes, (
            f"BudgetStreamManager: chunk class-introduction total {total_new} != "
            f"nb_classes {self._nb_classes}")
