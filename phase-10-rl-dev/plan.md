# Universal target-specific RL development plan

## Decision boundary

The eight receptors in `complexes/dev-panel.txt` are the complete development
panel. One algorithm, one set of hyperparameters, and one adaptive stopping rule
must pass all eight. Only then is the protocol frozen and copied verbatim to the
five-receptor benchmark. A receptor-specific patch is not an acceptable outcome.

## Evidence carried forward

The archived `phase-10` work established four useful facts.

1. Target-specific RL is real: the original three-stage protocol improved minimum,
   median, and poor-tail docking scores for all eight development and all five
   historical benchmark receptors, while sharply reducing positive scores.
2. Fixed update counts are not universal. Slow targets such as `1nyx` and `5ek0`
   had thousands of elite molecules but needed more concentration updates.
3. Rewarding increasingly extreme docking scores permits chemistry exploitation:
   low QED, high formal charge, excessive aromaticity, and near-zero fraction-Csp3.
   QED and charge guards helped, but did not prevent planar fused-ring solutions.
4. Cached fast scores are unsafe checkpoint evidence. In the `1err` rescue, one
   molecule occupied 63.55% of draws; its cached fast score was -11.010 but its
   independent fast score was -9.693. The online gate passed while the final fast
   qualified rate collapsed to 27.94%.

## Candidate protocol

The new design keeps the existing iGen3 model/sampler and existing iGenVS/Uni-Dock
path. It adds only RL-specific controls:

- Uni-Dock `balance` first establishes a robust target-specific region. A final
  short `fast` stage refines the same policy against a separately generated fast
  base reference. This is a single fixed four-stage recipe for every receptor.
- A bounded percentile reward first shifts the whole distribution; it has no
  unbounded extreme-score bonus.
- The balance stages use a binary reward for the chemistry-qualified base top-1%
  region; the final fast refinement uses the top 0.5% to give its noisier search a
  fixed margin. Every qualifying molecule is equally good, so a noisy extreme
  cannot earn more.
- At most 32 occurrences of one canonical molecule contribute to a batch policy
  gradient. Repeats remain in raw metrics and the KL term, but cannot dominate the
  reward gradient.
- Every checkpoint evaluation is freshly docked and bypasses the training score
  cache. Two consecutive 4,096-draw evaluations must pass.
- Reward qualification requires iGen3 Lipinski, QED >= 0.40, absolute formal charge
  <= 1, fraction-Csp3 >= 0.05, and at most four aromatic rings. Archived base
  references contain qualifying top-1% examples for every one of the 13 targets.
- The original base model remains the immutable KL prior.

## Acceptance contract

An independent 10,000-draw test retains invalids, repeats, failures, non-elites,
and positive scores in every denominator. For both Uni-Dock `fast` and `balance`,
each target must have at least 90% raw chemistry-qualified top-1% draws, at least
200 distinct qualified elites, at most 1% positive scores, and gains over the
matched base of at least 1 kcal/mol for best-10 mean and at least 2 kcal/mol for
median and p95. The single minimum remains reported but is not an acceptance gate:
it compares one unstable extreme from each independent sample and rewards a model
with more unique lottery tickets. It must also have at least 500 distinct valid
molecules, no single molecule above 20% of raw draws, and at least 90% chemistry
pass. All eight development targets must pass; no averaging can hide a failure.

## First-candidate decision

The balance-only candidate completed prospectively on all eight receptors. Every
target passed balanced concentration, best-10, median, p95, positive-score,
chemistry, and diversity requirements. Fast qualified concentration was 79.27% to
87.04% on five targets and at least 92.87% on the other three. A fresh binary-elite
fast stage then passed 7/8 independent endpoints. `4f8h` passed two online fast
checks but reproduced at 84.57%; overlapping molecules had only 0.579 fast-score
Spearman agreement between the checkpoint and endpoint docking batches. The last
development revision therefore uses the same bounded binary reward at the deeper
base top-0.5% cutoff. This supplies a fixed margin around the noisy top-1% boundary
without reinstating an unbounded docking reward. It still requires two consecutive
4,096-draw evaluations at or above 93%. At that decision point, the five
benchmark receptors were still untouched. The predeclared contingency was to
revise only from aggregate eight-target evidence and rerun the full development
panel; it was not needed after this candidate passed 8/8.

## Final outcome

The top-0.5% fast-refinement candidate was the final development revision. It
passed every predeclared endpoint on all 8/8 development receptors and was frozen
before benchmark access. The byte-exact freeze is recorded in `freeze.json`, and
the complete development metrics are in `REPORT.md`.

The unchanged operational recipe then passed all 5/5 held-out benchmark
receptors. This supports one receptor-agnostic training protocol that produces a
separate target-tuned iGen3 model for each receptor. It is not evidence for one
shared receptor-independent checkpoint, nor a guarantee for every receptor.
