"""Round 2 §2.4: unit test disambiguating the two possible readings of
SketchLoRA's bounded-eviction admission rule (models/sketchlora.py) on
synthetic singular spectra -- no GPU, no dataset needed.

Reading A (conformant, already implemented in models/sketchlora.py):
    evict = min(r_residual, max(0, composite_rank - k_eps_keep))
    where k_eps_keep is the rank the energy threshold would keep if applied
    fresh to the composite (the plan's "k_epsilon" read as a KEEP count).
Reading B (rejected, implemented here only for the test):
    evict = min(r_residual, k_eps_keep)
    i.e. the plan's literal "t = min(r_residual, k_eps)" read with k_eps
    itself substituted directly as the eviction count.

Both readings structurally satisfy evict <= r_residual, so BOTH are
"monotone non-decreasing" in the narrow sense that new_rank >= prev_rank
always. The real distinguishing failure of Reading B is that it does not
correctly RESPOND to how aggressively the energy threshold wants to compress:
when k_eps_keep is small (composite spectrum is very compressible), Reading A
evicts up to the full residual contribution (real compression), while
Reading B evicts only k_eps_keep directions -- near-zero whenever the
threshold is being aggressive, which is exactly backwards.
"""


def reading_a(prev_rank, residual_total, k_eps_keep, rank_cap=None):
    composite = prev_rank + residual_total
    cap = rank_cap if rank_cap is not None else composite
    if composite > cap:
        evict = composite - cap
    else:
        evict = min(residual_total, max(0, composite - k_eps_keep))
    return max(1, composite - evict)


def reading_b(prev_rank, residual_total, k_eps_keep, rank_cap=None):
    composite = prev_rank + residual_total
    cap = rank_cap if rank_cap is not None else composite
    if composite > cap:
        evict = composite - cap
    else:
        evict = min(residual_total, k_eps_keep)
    return max(1, composite - evict)


def run_case(name, prev_rank, residual_total, k_eps_keep, rank_cap=None):
    a = reading_a(prev_rank, residual_total, k_eps_keep, rank_cap)
    b = reading_b(prev_rank, residual_total, k_eps_keep, rank_cap)
    a_ok = a >= prev_rank
    b_ok = b >= prev_rank
    a_evict = prev_rank + residual_total - a
    b_evict = prev_rank + residual_total - b
    print(f"{name}: prev_rank={prev_rank} residual={residual_total} k_eps_keep={k_eps_keep} "
          f"cap={rank_cap} -> A: new_rank={a} (evicted {a_evict}, monotone={a_ok}) | "
          f"B: new_rank={b} (evicted {b_evict}, monotone={b_ok})")
    return a_ok, b_ok


def main():
    print("=== monotonicity check across a sweep of synthetic (prev_rank, k_eps_keep) ===")
    all_a_ok, all_b_ok = [], []
    for prev_rank in [5, 20, 50, 100, 127]:
        for k_eps_keep in [1, 5, 10, 30, 60, 90, 127, 200]:
            a_ok, b_ok = run_case("sweep", prev_rank, 10, k_eps_keep, rank_cap=128)
            all_a_ok.append(a_ok)
            all_b_ok.append(b_ok)
    print(f"Reading A monotone in {sum(all_a_ok)}/{len(all_a_ok)} cases")
    print(f"Reading B monotone in {sum(all_b_ok)}/{len(all_b_ok)} cases")

    print()
    print("=== responsiveness check: aggressive threshold (small k_eps_keep) at large prev_rank ===")
    # This is the case that actually distinguishes the two readings: the energy
    # threshold says "you only need rank 2 to hit 99% energy" (highly
    # compressible spectrum) while the composite has grown to 110 (100 + 10
    # residual). A real "bounded eviction" rule should still evict a real
    # amount (up to the residual's own contribution) in this case, not
    # silently do almost nothing.
    a = reading_a(prev_rank=100, residual_total=10, k_eps_keep=2, rank_cap=128)
    b = reading_b(prev_rank=100, residual_total=10, k_eps_keep=2, rank_cap=128)
    print(f"prev_rank=100 residual=10 k_eps_keep=2 (highly compressible) "
          f"-> A new_rank={a} (evicted {110-a}) | B new_rank={b} (evicted {110-b})")
    print("Reading A evicts the full residual contribution (real compression, tracks the "
          "energy signal); Reading B evicts only 2 directions regardless of how compressible "
          "the composite actually is -- it fails to compress precisely when compression is "
          "most warranted, the opposite of the rule's stated purpose.")

    print()
    print("=== cap behavior (both readings identical once composite > cap) ===")
    a = reading_a(prev_rank=125, residual_total=10, k_eps_keep=50, rank_cap=128)
    b = reading_b(prev_rank=125, residual_total=10, k_eps_keep=50, rank_cap=128)
    print(f"prev_rank=125 residual=10 (composite=135 > cap=128) -> "
          f"A new_rank={a} | B new_rank={b} (both should be exactly 128)")
    assert a == 128 and b == 128, "cap enforcement broken in one of the readings"

    print()
    print("CONCLUSION: Reading A (implemented in models/sketchlora.py) is selected as the "
          "spec-conformant rule -- it is the only one that actually delivers eviction "
          "proportional to what the energy threshold requests. Reading B is documented here "
          "and rejected; it is never used in any production config.")


if __name__ == "__main__":
    main()
