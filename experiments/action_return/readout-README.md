# Frozen-state BEGIN action controller

`readout.py` is a new, externally supervised action-head experiment. It imports
the old model, world and autonomous evaluator unchanged; it does not retrain or
repair native output ports. The immutable protocol is
`data/action-return/readout-v1/protocol.json`.

Three assisted coverage endpoints at tick147456 supply the actual frozen model,
hidden state and world. Decoder training uses all tick/job positions from the
first64 assisted continuation blocks and actor lanes0..3. It continuously
replays the original public observations, including true historical action
receipts, through the endpoint-frozen backbone starting from CP131072 h. This
training feature trajectory differs from historical actual h, because original
training changed parameters. It is an off-policy input history; all free
evaluation forks instead start from the real CP147456 h/world and generate
their own actions and subsequent receipts.

Each seed/job has an affine three-way classifier, OTHER/BEGIN1/BEGIN2. All
teacher labels enter fitting. Original WAIT rows carry weight.1 and every
original nonWAIT row carries weight1, before WAIT and FINISH labels collapse to
OTHER. OTHER is anchored at logit0; fixed L2.001 penalizes standardized slopes.
For action choice, each BEGIN logit becomes the best existing native nonBEGIN
score plus its fitted log odds against OTHER. All other action scores remain
exactly native. No evaluation target, readiness, task metadata, gate or fitted
test threshold reaches the controller.

The primary feature set is the24 nodes outside all56 input ports. All80 state
nodes and the current56 public observation features are diagnostic arms with
the same fitting and action rule. They have different input dimensions and
parameter counts (both jobs together:100,324 and228), so they are source and
procedure controls, not parameter-count-matched controls.

For each arm/seed, the unchanged evaluator runs four context families times
short/long_both,2048 ticks each. Its original full-world results remain saved.
A read-only shadow of its actual `world.act` call adds midpoint FINISH label
counts, timing/action counts and matured request results split into decoder
fitting-heldout lanes4..7 and overlapping lanes0..3. The original backbone was
trained on all8 lanes. Every future request identity is checked against decoder
training identities. All arms must have identical matured primary case sets.
The native shadow rerun must reproduce the entire saved original endpoint
evaluation after JSON type normalization, including actions and state hashes.

Primary quantities are matured matched requests with an accepted autonomous
BEGIN, its feasible-correct subset, ordinary and strict completed requests, and
idle false BEGIN per1000 actual inactive opportunities. Repeated-position
timing sensitivity and precision are diagnostics, since a controller's own
BEGINs alter later labels. Legal FINISH opportunities, midpoint FINISH labels,
attempts and accepted FINISHs show whether changed action routing opens the
subsequent part of the loop. No FINISH decoder is trained here.

```powershell
.\.venv\Scripts\python.exe -m pytest experiments\action_return\test_readout.py -q
.\.venv\Scripts\python.exe experiments\action_return\readout.py prepare
# Run these in three single-thread processes; each command fixes all thread
# environments and CUDA_VISIBLE_DEVICES=-1 itself.
.\.venv\Scripts\python.exe experiments\action_return\readout.py run --seed 0
.\.venv\Scripts\python.exe experiments\action_return\readout.py run --seed 1
.\.venv\Scripts\python.exe experiments\action_return\readout.py run --seed 2
.\.venv\Scripts\python.exe experiments\action_return\readout.py summary
```

`prepare` and each seed's `run` refuse to overwrite existing results. Per-seed
`train_dataset.pt.gz` contains `groups[job]` with h, current obs, original11-way
label and request identity set; `fits.pt` contains frozen coefficients and
normalization; `fit.json` binds their SHA256s. `result.json` holds original
evaluator records and per-lane shadow counts. `summary.json`/`summary.csv` sum
family counts within each seed/condition/lane group; independent seeds remain
the replication unit.

## Post-result version-score swap

`readout_swap.py` is a separately predeclared diagnostic added after observing
the non_input24 controller's many inappropriate attempts. It refits nothing.
It swaps its two final BEGIN scores, asserting that BEGIN versus nonBEGIN
choice at each given actual obs/h remains unchanged, while unique selected
BEGIN scores choose the opposite version. Subsequent own trajectories and
gate choices can diverge through different actual actions and receipts.

The independent protocol is
`data/action-return/readout-swap-v1/protocol.json`. It reruns all24 original
non_input24 forks with complete evaluator/shadow equality checks and adds24
swapped forks. A read-only scorer partitions *matched matured feasible-BEGIN
requests* by whether their accepted BEGIN occurred at that slot's current
NEW/REPLACE cue or without a current command pulse (NONE). It separately
reports actual accepted events, active correct/wrong-version attempts and idle
actions. NONE still includes public previous-action/result receipts, so delayed
acceptance does not alone establish remembered version information.

```powershell
.\.venv\Scripts\python.exe -m pytest experiments\action_return\test_readout_swap.py -q
.\.venv\Scripts\python.exe experiments\action_return\readout_swap.py prepare
.\.venv\Scripts\python.exe experiments\action_return\readout_swap.py run --seed 0
.\.venv\Scripts\python.exe experiments\action_return\readout_swap.py run --seed 1
.\.venv\Scripts\python.exe experiments\action_return\readout_swap.py run --seed 2
.\.venv\Scripts\python.exe experiments\action_return\readout_swap.py summary
```
