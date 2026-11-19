# Timing boundary independent audit

Declared before probe scoring. This audit reads artifacts and frozen checkpoints;
it does not fit probes, select checkpoints, modify original evidence, or alter
policy actions. The audit program and its declaration pin the protocol, source,
checkpoint, raw-stream, coefficient, and prediction files before verification.

1. Verify disjoint actor identities for training, validation, and test. Teacher
   streams are common to midpoint, first, interior, and random models. Compare
   policy streams against the same external test events, while retaining their
   own observations, receipts, eligibility, and boundary coverage.
2. Independently replay native streams from their independent origin. Compare
   every saved observation, event, private state, action receipt, and maturity
   pulse. Reconstruct receipts from action validity and public-history causal
   state as a separate arithmetic check.
3. Rebuild each request identity from actual NEW/REPLACE events, eligibility,
   issue time, and a later matching maturity pulse. Select post-warmup eligible
   requests without conditioning on completion, strict success, prediction, or
   content. Rebuild the last active, started pre-ready row and first active,
   started legal row, require they are consecutive and of the same actor, slot,
   request, version, and payload. Record all exclusions and retain the mature
   denominator. Offsets -8, -4, -2, -1, 0, +1, +2 are relative to first legal;
   retain an offset only while this same request is active and started.
4. Check every selected row's metadata and exact public observation against the
   raw stream. Private clocks, labels, offsets, request IDs, family, success,
   and maturity are used only for targets, selection, grouping, and audit.
5. Replay each frozen GRU continuously from zero at its independent origin,
   including the complete warmup. Compare hidden states after consuming the
   current observation and logits at every saved row. Independently run a
   chunked sequence forward and record its numerical discrepancy, then require
   bit-exact agreement from a separate complete tickwise native-GRU replay.
   The first vectorized verifier run (seed 0 teacher test) differed by up to
   3.28e-5 in logits near zero because float32 sequence and tickwise operations
   use different matrix multiplication paths. The bit-exact replay was added
   before audit declaration; pass tolerances were not relaxed. Independently
   recompute the original head logits from hidden states and frozen head
   weights. Check that checkpoint and raw file hashes remain unchanged.
6. Recompute every saved linear/logistic/ridge prediction directly from saved
   normalization, coefficients, and feature definitions without importing the
   fitting or scoring program. If a fixed nonlinear control is present, replay
   its saved affine layers and activation directly. Verify normalization comes
   from training rows only. Rebuild the fixed pair-label-flip training control;
   validation, test, and policy-history targets remain the real labels.
7. Recompute row accuracy, both-rows-correct pair accuracy, and within-pair score
   ranking separately. Ranking ties count one half; threshold ties use the
   frozen classifier rule. Recompute content correctness and conditional-eight
   head correctness separately from eleven-way action choice. Recompute time
   error on every retained offset with denominators by offset. Retain exact
   counts by model, seed, split/history, slot, family, and conflict status.
8. Keep probe decodability separate from native free-action completion, strict
   completion, idle opportunities and idle errors. Own-policy boundary coverage
   is action-dependent and must be reported alongside all mature requests; it
   is not a matched-cohort causal comparison with teacher histories.

Numerical checks require finite inputs, parameters, hidden states, logits,
coefficients and predictions. GRU/head checks use atol 5e-6, rtol 1e-5; saved
float64 probe calculations use atol 1e-9, rtol 1e-8. Exact counts and identities
must match exactly. A failed assertion prevents a passed audit artifact.
