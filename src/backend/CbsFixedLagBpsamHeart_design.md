# CbsFixedLagBpsamHeart: Incremental Lag-Heart Refactor (Supervisor Track)

## Why legacy x/v/b-only lag logic is insufficient

The current stable CBS-heart path in `VioBackend` computes stale-tail eviction mainly from local Kimera state symbols (`x`, `v`, `b`) and then applies belief/orphan pruning as an augmentation step.

That keeps runtime bounded, but it is not yet a true fixed-lag definition over the full expanded BPSAM graph.

In a BPSAM-heart architecture, lag handling must reason about local-state ownership and belief-owned structures together, because stale local frames can leave attached belief factors and non-local keys disconnected or underconstrained.

## Expanded BPSAM state for Kimera CBS-heart

Expanded state/factor structure includes:

- local Kimera state keys (`x`, `v`, `b`) and their local factors,
- belief factors inserted with BPSAM Between semantics (`addBeliefs`/`beliefToFactor`),
- robot keys,
- consensus/pose-public keys,
- other GBP-side keys introduced by belief graph connectivity.

A lag policy over only local keys is therefore incomplete for true BPSAM-heart behavior.

## What must be evicted together at lag boundary

When stale local frames are evicted, eviction planning must explicitly account for:

- stale local keys and all incident factor slots,
- belief factors attached to stale local keys,
- disconnected belief components (orphan belief factors),
- orphan robot / GBP / consensus keys after planned factor removals.

## Temporary vs final handling

The existing `VioBackend` lag-boundary bridge (B1 anchor-style approximation) remains temporary and untouched in this phase.

This scaffold does **not** claim that the bridge is final marginalization logic.

## Lag-edge summary insertion point

`CbsFixedLagBpsamHeart` now computes `BoundarySeparatorCandidateSet` and exposes `SummaryInsertionPlan`.

This is the insertion point used in the current phase:

1. form separator/boundary over remaining active expanded graph,
2. build a LOCAL-only stale-side subgraph from removable stale-local factors,
3. linearize at current estimate and eliminate stale-side keys with partial multifrontal elimination,
3. inject summary factor/prior onto `summary_target_keys`.

The emitted factors are `LinearContainerFactor`s converted from the Gaussian
remainder graph after elimination, i.e. a kept-side Schur-style linear summary
for the selected boundary.

## Incremental lag state (new default inside PHASE-2 module)

The `CbsFixedLagBpsamHeart` module now keeps persistent incremental state:

- key timestamps (`incremental_key_timestamps_`),
- active local ownership by frame (`incremental_local_ownership_`),
- active belief-factor ownership (`incremental_belief_ownership_`),
- active robot/GBP/consensus ownership (inside `BeliefOwnedState`),
- newest/oldest active frame bookkeeping,
- cached separator/boundary + summary-target plan (`incremental_plan_cache_`).

Online usage shape now follows fixed-lag smoother ideas:

1. ingest local factors/values/timestamps incrementally (`ingestLocalKimeraPacket`),
2. ingest incoming beliefs incrementally with BPSAM Between semantics (`ingestIncomingBeliefs`),
3. advance lag window incrementally (`advanceLagWindowIncremental`),
4. consume incremental remove indices (`getIncrementalRemoveFactorIndices`),
5. prepare explicit future summary payload (`buildBoundarySummaryInput...`).

## Full recompute demoted to fallback/debug

`recomputeLagWindow(...)` remains in the class but is treated as:

- one-time safety seed / fallback path,
- debug/validation path,
- **not** the intended online hot path.

The intended runtime path uses persistent state + incremental advance.

## PHASE-2 lag-edge summary path

`CbsFixedLagBpsamHeart` now exposes `appendLagEdgeSummaryFactors(...)`:

- consumes the computed separator target set,
- selects LOCAL-only stale-side removable factors,
- builds a linearized Gaussian subgraph at the current estimate,
- eliminates stale-side keys (`eliminatePartialMultifrontal`),
- injects the kept-side remainder graph as `LinearContainerFactor`s.

Current summary target policy:

- `summary_target_keys` is the full LOCAL kept-separator set
  (`boundary_candidates.local_separator_keys`), not a bounded triplet.

Target coherence semantics (new):

- `requested_summary_target_keys`: all requested LOCAL kept separator targets
  from boundary planning.
- `realizable_summary_target_keys`: subset that actually appears in the
  selected LOCAL stale-side summary subgraph and has a valid estimate.
- requested targets absent from the selected LOCAL summary subgraph are dropped
  (recorded via dropped-target diagnostics) instead of hard-failing the epoch.
- `exact_schur` is reported only when:
  - realizable targets are non-empty,
  - emitted summary factors are non-empty,
  - no remaining internal target/subgraph inconsistency exists.

### Audit of prior exclusion points (kept as fallback path)

The previous LOCAL-only builder inside `appendLagEdgeSummaryFactors(...)` had
three explicit exclusion points:

1. `isBeliefFactor(factor)` factors were skipped.
2. any factor with at least one non-local key was skipped (`all_local` guard).
3. selected summary subgraph was therefore restricted to all-local stale-side
   factors only.

That LOCAL path remains in place as a validated fallback.

### First expanded-summary step (current implementation)

The heart now attempts an expanded crossing-factor summary first:

- candidate set: factors in `stale_local_factor_slots` that cross stale->kept
  boundary (touch stale local key and a kept-side key),
- included factor classes:
  - local non-belief crossing factors,
  - mixed-key non-belief crossing factors,
  - belief crossing factors,
- excluded (for now):
  - crossing factors with missing estimates for any involved key,
  - non-crossing stale factors.

Ownership / no-double-counting rule:

- only factors in `stale_local_factor_slots` are summarized, i.e. the same
  stale-side factors removed from the active graph this epoch.

Target semantics:

- `requested_expanded_summary_target_keys`: full kept separator set
  (`boundary_candidates.separator_keys`),
- `realizable_expanded_summary_target_keys`: requested targets that appear in
  the selected expanded crossing subgraph and have valid estimates.

Mode semantics:

- `exact_expanded_partial_scope`: expanded elimination succeeded and emitted
  kept-side summary factors.
- if expanded fails, heart falls back to the validated LOCAL exact path and
  reports fallback mode diagnostics explicitly.

Current limitation:

- this is still a partial expanded step, not yet a complete final lag-edge
  summary over every belief-owned/non-local structure in all cases.

Runtime gate:

- `--cbs_experimental_use_new_heart_summary_prior_bridge=false` (default):
  keep current stable B1 path.
- `--cbs_experimental_use_new_heart_summary_prior_bridge=true`:
  use incremental-heart + lag-edge summary-factor experiment in
  CBS-heart mode.

## Lightweight incremental diagnostics

The incremental heart now tracks/exports:

- incremental lag update time,
- stale local key count,
- stale belief-factor count,
- orphan key count,
- boundary candidate count,
- summary target count,
- summary crossing-factor count,
- summary factor count emitted,
- summary build time,
- summary mode / first failure reason,
- whether fallback full recompute was used.

This enables direct A/B comparison against the old full-recompute-per-boundary
bridge path.

## Reused conceptual building blocks

The scaffold reuses the same conceptual decomposition already validated in `VioBackend`:

- stale-tail local scan logic from `buildCbsFixedLagWindowState(...)`,
- belief/orphan connectivity pruning concept from `augmentCbsLagWindowWithBeliefOwnershipPruning(...)`,
- LOCAL-only covariance query concept (kept separate from fused semantics).

But it does so in a new dedicated module so the final architecture can replace
the heart cleanly later without destabilizing current runtime behavior.
