# Pre-registration: Final Experimental Batch

Mohammed Al-Khudhair, 31482 Honours Project

Registered on: 25/09/2026, 23:00

This records, in advance, what the final batch is expected to show, how
each expectation will be tested, and how the results will be treated. The
methods chapter commits to pre-registration as part of construct validity;
the predictions come from the three development batches (pilot, 3M, 6M),
which are kept separate from the final results.


## 1. Design

- Controllers: ORCA/RVO, rule-based heuristic, MAPF (prioritised
  cooperative A*), CTDE PPO.
- 6 layouts x 2 demand levels x 2 human densities = 24 conditions.
- Training: 10 seeds per policy (10 to 19, unused in development),
  10M steps, imitation anchor fading over the first 2M steps, v2 reward.
- Evaluation: 30 ten-minute episodes per condition per controller (3 per
  PPO seed), from a held-out block of episode seeds never used in
  development. Every controller faces the same episodes.
- Primary metrics: orders per hour, congestion time proportion,
  collisions per 100 timesteps, separation violations per interaction.
  Secondary: metres driven per delivery.


## 2. Analysis

- Unit of analysis: a seed unit, the mean of one trained seed's episodes
  in one condition (baselines grouped identically on the same episodes).
- Aggregates: interquartile mean over all condition x seed units, with 95%
  stratified bootstrap confidence intervals.
- Per-condition tests: two-sided Mann-Whitney U on seed units, alpha 0.05,
  Holm-Bonferroni correction across all conditions, metrics and baselines
  within a suite.
- Probability of improvement (Agarwal et al., 2021) with 95% bootstrap
  confidence intervals, averaged over conditions.


## 3. Confirmatory Hypotheses

A hypothesis is supported when the aggregate 95% confidence intervals do
not overlap in the predicted direction and the probability of improvement
interval lies entirely on the predicted side of 0.5. Per-condition counts
are reported alongside.

| | Hypothesis | Basis in development runs |
|---|---|---|
| H1 | PPO has fewer collisions per 100 timesteps than ORCA and than MAPF. | 6M: 5.10 against 8.05 and 7.68 |
| H2 | PPO has fewer separation violations per interaction than ORCA and than MAPF. | 6M: 0.57 against 0.69 and 0.70 |
| H3 | PPO has lower throughput than ORCA and than MAPF. | 6M: 288 against 303 and 301 |
| H4 | PPO's throughput gap to ORCA is smaller than at 6M (5.1%). | Gap fell from 10% (3M) to 5.1% (6M); curves still rising |
| H5 | PPO has higher throughput and lower congestion than the heuristic. | 6M: 288 against 261; 0.01 against 0.13 |
| H6 | The heuristic has fewer separation violations than PPO. | 6M: 0.44 against 0.57 |
| H7 | PPO with the v2 reward has higher throughput than with the v1 reward. | 6M: 334 against 143 |
| H8 | Safety-only PPO has lower throughput than full-reward PPO. | 6M: 0 against 334 |
| H9 | Throughput-only PPO has more collisions than full-reward PPO. | 6M: 7.65 against 5.37 |
| H10 | PPO's separation-violation advantage over ORCA holds at every sensitivity level. | 6M: significant at all five levels |


## 4. Exploratory Questions (no direction predicted)

- Robot density stress (150% robots): how each controller's metrics change.
- Demand transfer: how standard-demand-only policies compare with
  mixed-demand policies and ORCA at peak demand.
- Curriculum: how the curriculum policy compares with the directly trained
  policy and ORCA on Large 1.
- Learning progress: how the main policies change between the 3M, 6M and
  10M snapshots.
- Cross-layout transfer and 150% human density.
- Throughput-only PPO against ORCA on throughput (a tie at 6M).


## 5. Rules for Reporting

1. Every hypothesis and exploratory result is reported, whether or not it
   supports the expectation.
2. No training setting, reward weight or evaluation choice is changed after
   the final results are seen. The only exception is a genuine bug; if one
   is found, it is described, fixed, and every affected result is rerun and
   reported alongside the original.
3. Development-batch results are reported as development, never pooled
   with the final batch.
4. Any deviation from this plan is listed in the thesis with its reason.
