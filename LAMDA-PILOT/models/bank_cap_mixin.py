"""Bounded bank memory: a byte-budget cap on a method's persistent,
non-classifier-head storage (adapters, prompts, subspace bases, gradient
snapshots -- whatever a method accumulates across tasks), distinct from
``bounded_memory_mixin.py``'s existing sample-*volume* budget (that one
limits how much training DATA a method sees per task; this one limits how
much it's allowed to keep afterward).

One shared byte budget per method (config key ``bank_cap_mb``, MB, unset =
feature off). Once one more admission would exceed it, the bank freezes
PERMANENTLY -- a strict one-way latch, never re-checked, never un-frozen,
even if a method's own reported bytes later dip (see the InfLoRA design
audit for why a re-checking gate would be unsafe: it would silently and
permanently drop whichever task's contribution happened to land during a
transient over-budget window). Each participating Learner supplies its own
``_bank_bytes()`` (current non-head bytes it wants counted) and calls
``_bank_admit(candidate_bytes)`` before growing anything -- the fallback
behavior once admission is refused is entirely method-specific and lives in
each method's own file, not here. This mixin only ever answers "may I grow?";
it never decides what a method does once the answer is no.

Design principle carried from the plan doc: prefer the simplest gate that
does not break the method, not the cleverest one that squeezes the most use
out of what's banked. This class is deliberately minimal -- a budget field,
one admission check, one latch flag -- so every method's fallback complexity
lives visibly in that method's own file, not hidden in shared plumbing.
"""


class BankCapMixin:
    def _bank_cap_init(self, args):
        cap_mb = args.get("bank_cap_mb")
        self._bank_cap_bytes = int(cap_mb * 1024 * 1024) if cap_mb is not None else None
        self._bank_frozen = False

    def _bank_bytes(self):
        """Current non-head bank bytes counted against the cap. Every
        participating Learner overrides this; the base implementation is
        intentionally absent (not a default of 0) so a method that mixes
        this in without overriding it fails loudly rather than silently
        never freezing."""
        raise NotImplementedError(
            "{} mixes in BankCapMixin but does not implement _bank_bytes()".format(type(self).__name__))

    def _bank_admit(self, candidate_bytes):
        """Call BEFORE growing a bank structure by candidate_bytes. Returns
        True if the growth is allowed (feature off, or room remains), False
        if it must be refused. Refusing latches self._bank_frozen=True
        permanently -- every subsequent call returns False regardless of
        candidate_bytes or of what _bank_bytes() reports later."""
        if self._bank_cap_bytes is None:
            return True
        if self._bank_frozen:
            return False
        if self._bank_bytes() + candidate_bytes > self._bank_cap_bytes:
            self._bank_frozen = True
            return False
        return True
