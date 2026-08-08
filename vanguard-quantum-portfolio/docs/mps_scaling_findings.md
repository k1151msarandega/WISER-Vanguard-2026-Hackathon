# MPS vs. Partitioning: Scaling Ablation Findings

Week 3 preliminary work. Two independent scaling levers for the O-set QAOA
solve -- H/O/S partitioning (shrinks the problem before it reaches the
quantum solver) and MPS simulation (lets the solver handle more qubits than
statevector) -- tested separately, not just combined, to see which one is
actually doing the work. Code: `src/vqportfolio/validation/scaling_ablation.py`,
`run_ablation_at_size.py`.

**Caveat that applies to every result below:** all runs used the synthetic
price fallback (this sandbox has no internet access to real market data).
The covariance structure is not pure noise -- our synthetic generator uses a
block correlation model (0.55 within-asset-class, 0.15 across) -- so the
qualitative findings are plausible, but **this needs re-running with real
data before any number here is trusted for the actual submission.**

## Finding 1: Real (or realistically-structured) financial correlation is
## friendlier to MPS than adversarial random dense matrices

An early stress test used a **purely random** dense 20-qubit Hamiltonian
(every ZZ coefficient drawn independently, no structure) -- uncapped MPS
didn't finish in 2 minutes. Our actual QUBO's Hamiltonian, built from the
portfolio covariance matrix (also dense, but not random -- real assets have
correlation structure, even in our simplified synthetic version), ran
uncapped MPS in the same ballpark of qubits in a few seconds. The takeaway
isn't "MPS works" or "MPS doesn't work" -- it's that **entanglement
structure, not just qubit count or nominal density, determines whether MPS
is a good fit**, and financial covariance matrices (real ones especially,
via factor structure -- a handful of dominant risk factors) are likely
friendlier than the worst case.

## Finding 2: A real bug was caught and fixed mid-investigation

Comparing backends via sampled-bitstring decoding produced a nonsensical
result (bond≤8 MPS "beating" *uncapped* MPS, which is mathematically
impossible -- uncapped MPS is exact). Root-caused to a bit-ordering mismatch
between how different Aer simulation methods label measurement outcomes in
`get_counts()` -- confirmed via exact-statevector fidelity checks (fidelity
1.0, so the physics was always correct; only the bitstring *labels* differed
across backends). Fixed by comparing backends via Hamiltonian expectation
value (`save_expectation_value`) instead, which sidesteps bitstring decoding
entirely. This does not affect the production solver (`quantum/qaoa_solver.py`),
which always samples through one backend consistently and never compares
bitstrings cross-backend.

## Finding 3: Scaling curve, sizes 3-6 (9-21 qubits), p=1

| O size | Qubits | Statevector (expectation) | MPS bond≤16 | MPS uncapped | Uncapped matches SV? |
|---|---|---|---|---|---|
| 3 | 9  | 2.0947 | 2.0947 | 2.0947 | yes |
| 4 | 12 | 7.8306 | 7.8306 | 7.8306 | yes |
| 5 | 15 | 7.2140 | 7.2140 | 7.2140 | yes |
| 6 | 21 | 8.1975 | 8.1975 | 8.1975 | yes |

Correctness held at every size tested: uncapped MPS exactly reproduces
statevector's expectation value, as required mathematically. Bond≤16
truncation tracked the uncapped value closely at every size -- no dramatic
quality collapse from truncation in this range.

## Finding 4: A hard architectural ceiling at 27 qubits, not a soft one

`AerSimulator(method="statevector")` -- used internally for the QAOA angle
*optimization* step, even when comparing MPS at final sampling -- has a
default target that caps at **27 qubits**. 28 qubits fails outright with
`CircuitTooWideForTarget`, not a slowdown. This means the current
architecture cannot scale the O-set past this point without also switching
the optimization step itself to an MPS-based cost evaluation, not just the
final backend comparison. Not previously known; found empirically while
attempting to extend the scaling curve.

## Finding 5: MPS runtime scales sharply with circuit depth (p), independent
## of qubit count

Fixed size (15 qubits), varying only p (QAOA layers), same seed:

| p | Gap to exact | MPS uncapped runtime |
|---|---|---|
| 1 | 9.24 | 0.31s |
| 2 | **7.08** | 3.49s |
| 3 | 7.46 | 7.87s |

Two separate things here, not one:
- **Solution quality**: p=1 → p=2 is a real improvement (~23% smaller gap),
  consistent with the Fraunhofer benchmark cited in the Week 2 literature
  review (quality improves with depth). p=2 → p=3 plateaus -- but this is
  **confounded, not conclusive**: the classical optimization budget (2
  restarts × 60 iterations) was held fixed across all three depths, while
  the parameter count (2×p) grows with depth. We cannot cleanly separate
  "depth stops helping past p=2" from "the fixed budget stops being adequate
  for p=3's larger search space." Properly separating these was scoped as a
  stretch goal for the Week 4 buffer, not pursued further now -- open-ended
  (how many restarts is "enough" at each depth, multiple seeds needed to
  separate signal from the seed-variance we already know is large at this
  qubit count) and lower-priority than closing required-deliverable gaps
  (co-pilot demo, presentation) that were at zero progress.
- **MPS runtime cost**: unconfounded, real, and monotonic -- 0.31s → 3.49s →
  7.87s. Deeper circuits generate more entanglement per layer, which is
  exactly the mechanism MPS truncation is sensitive to. This is the more
  important finding for the "scalability" story: **MPS's qubit-count
  advantage and QAOA's depth requirement work against each other.** The more
  layers needed for solution quality, the more that erodes MPS's efficiency
  edge. Framing MPS as a simple, free scaling win would be misleading; this
  tension is the honest, more sophisticated version of that claim.

## Bottom line for the report

- Partitioning and MPS are genuinely independent levers; this ablation
  characterizes each rather than only reporting them combined.
- MPS is not a free scaling win for this problem class -- its benefit
  depends on entanglement structure (favorable) and erodes with circuit
  depth (unfavorable, and QAOA needs depth for quality).
- A real architectural ceiling (27 qubits) exists in the current statevector-
  based optimization step, independent of the MPS question.
- Everything above needs re-confirmation with real market data before being
  stated as a final result -- flagged, not hidden.

## Update: real-data run (Colab) -- the 27-qubit ceiling is a hardware wall,
## not just a soft software cap

Re-ran the qubit-count survey with real data. Sizes 3-5 behaved consistently
with the synthetic run (same qualitative pattern: MPS competitive or faster
at small sizes, statevector cost growing with qubit count). At size 6 (27
qubits, real data's specific asset selection at this size), a genuinely new
finding emerged:

- **Statevector**: completed, but took **748 seconds** just to compute one
  expectation value -- not a failure, but a two-order-of-magnitude jump from
  size 5's 7.5s.
- **MPS (both bond<=16 and uncapped)**: failed outright with
  `Insufficient memory to run circuit QAOA-384 using the matrix_product_state
  simulator. Required memory: 55297M, max memory: 12975M` -- MPS needed
  ~4.3x more RAM than Colab's free-tier instance provides.

This sharpens `mps_scaling_findings.md`'s original "soft ceiling" framing
into something more concrete and more useful for anyone planning
deployment: **27 qubits is not just where AerSimulator's default statevector
target happens to cap out -- it is also approximately where MPS's own
memory footprint exceeds commodity/free-tier hardware for this problem's
entanglement structure.** Both the "just switch to MPS" and "just raise the
qubit target" workarounds independently run into a real resource wall at
almost the same point, on real hardware most users would actually have
access to -- not a simulator configuration limit that a bigger `--target`
flag would fix.
