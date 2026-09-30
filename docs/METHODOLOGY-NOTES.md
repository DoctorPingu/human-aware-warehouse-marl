# Methodology notes

This file records every change from the methodology proposed in the Introduction and Methods chapters (Technology Research Methods, Autumn 2026), with the reason for each. The proposal allowed for documented changes ("Note on Scope and Changes"); these notes are the record behind that section of the thesis.


## 1. Problems found in the proof of concept

1. **The collision metric could not count robot collisions.** Its threshold was 0.44 m, but the physics never lets a robot and a person get closer than 0.45 m (their radii added), or two robots closer than 0.50 m. Only two people touching (0.40 m) ever crossed it, so every recorded "collision" in the proof-of-concept log (about 16.5 per episode) was between two people, which no policy can influence. This is why that curve stayed flat. The proof-of-concept collision numbers are therefore not robot collisions and are not reported as such.
2. **The environment ignored its random seed.** Every episode was identical, so 30 repeated runs would have produced the same number 30 times. Seeding now controls people's start points, speeds and pauses, and robots' start jitter and staggered departures.

Both are fixed in the current environment.


## 2. Changes to the method

1. **Shared actor, per-robot critic** (MAPPO; Yu et al., 2022). One policy network is shared by all robots, so a policy works with any number of robots and can be tested on other layouts. The critic scores each robot from the global state plus that robot's observation. Without this, the generalisation tests would have been impossible.
2. **Imitation warm start** (Damani et al., 2021; Rajeswaran et al., 2018). PPO starts from a policy trained to copy ORCA, with an imitation term that fades out. Pure PPO learned too slowly for the timeline. This also sharpens the research question: what does learning add on top of a reactive controller?
3. **Reward v2.** A collision is penalised when it starts, plus a small cost per step of continued contact, instead of a large penalty for every contact step. The safety penalty grows with how far a robot intrudes into the 0.8 m zone. The time penalty was reduced because it added a constant of about −200 to every episode and taught nothing. Evidence: under the original per-step collision penalty, the policy learned to stop moving (throughput fell from 351 to about 50 orders per hour in development).
4. **MAPF as prioritised cooperative A\*** (Silver, 2005) rather than conflict-based search. Conflict-based search solves one-shot problems; here robots receive tasks continuously and must replan, for which prioritised planning is the standard approach.
5. **Six layouts instead of nine.** The proposed 3 × 3 × 2 × 2 design gives 36 conditions, not the 24 stated; six layouts × 2 demand levels × 2 densities gives exactly 24.
6. **Training mixes both demand levels,** because both are evaluated. Out-of-distribution tests are unseen layouts, denser crowds, more robots, changed human behaviour, and a separate standard-demand-only training arm (item 19).
7. **ORCA/RVO2 runs underneath every controller** as the physics layer, so all four controllers are compared as decision layers on the same substrate.
8. **Simulation platform.** The simulation is a custom 2D Gymnasium environment with RVO2; it does not run inside Habitat 3.0. Habitat was the initial development environment, and 3D simulation is future work.
9. **Episode length: 10 simulated minutes** (14,400 steps at 24 steps per second) instead of 83 seconds (2,000 steps). The pilot showed that short episodes could not measure throughput: robots completed only 4 to 8 deliveries, so orders per hour moved in steps of 43.2 and all four controllers tied. Over 10 minutes the differences appear. The same test showed the imitation-only starting policy jamming over long episodes (68 orders per hour against ORCA's 348 on Small 1), which PPO training removes.
10. **Sensitivity analysis:** human walking speed 20% lower and higher, and a wider human avoidance radius (0.30 m and 0.40 m against a 0.20 m body radius), applied at evaluation time only, on Small 1 and Large 1 at high density. The nominal preferred speeds are 0.85 to 1.05 m/s, varied by up to 15% per episode and capped at 1.2 m/s.
11. **"Thirty independent runs per condition"** is implemented as 30 evaluation episodes per condition, spread over 10 independently trained policies (3 episodes each), with statistics computed per trained seed (item 15).
12. **PPO hyperparameters.** The proposal lists the single-environment proof-of-concept values (rollout 512, 4 epochs, minibatch 128). The toolkit uses 8 parallel environments × 256 steps, 10 epochs and minibatch 512, chosen for vectorised training and unchanged across all development and final runs. Learning rate (3e-4 annealed to 1e-5), clip (0.2) and network (128-128-64, Tanh) match the proposal.
13. **Robot density stress test:** 150% robots (3 become 5, 4 become 6), evaluation only, at standard human density. Each extra robot starts in the staging area between two existing robots and follows one of their validated routes. The critic's global state covers at most 4 robots, which does not matter at evaluation, where only the actor runs.


## 3. How the final batch was designed

14. **Development batches.** Three development rounds preceded the final batch: a pilot (300k steps, 83-second episodes) that exposed the episode-length problem; a 3M-step batch (3 seeds) whose training curves showed PPO held near its imitation start for half of training; and a 6M-step batch (3 seeds) that halved the throughput gap to ORCA while improving safety, but had not converged. Results are in `development/pilot`, `development/3m` and `development/6m`.
15. **Seed-level statistics** (Agarwal et al., 2021). Each trained seed's episodes in a condition are averaged into one seed unit, and every test, interval and figure uses these units, so the unit of evidence is a trained policy rather than an episode. Baselines are grouped the same way, on the same evaluation episodes. Probability of improvement is reported alongside significance tests.
16. **Final batch settings, fixed before it ran:** 10M steps (chosen by a convergence rule set in advance); the imitation anchor fades over a fixed 2M steps instead of half of training (at 10M, half would have held the policy near the imitation for 5M steps); 10 fresh training seeds (10 to 19) never used in development; a fresh held-out block of evaluation episodes; and pre-registered hypotheses (`PRE-REGISTRATION.md`).
17. **Resume fidelity.** The imitation demonstrations are saved to disk, so a run paused or interrupted during its first 2M steps resumes with the imitation term intact. A resumed run is statistically equivalent to an uninterrupted one but not bit-for-bit identical.
18. **Curriculum arm:** Small 1 at standard demand (3M steps, with the imitation warm start), then Medium 1 (3M), then Large 1 (4M), both with mixed demand, each stage starting from the previous stage's policy and with its own learning-rate schedule. Compared on Large 1 against the policy trained there directly for 10M steps.
19. **Demand transfer arm:** policies trained on standard demand only (Small 1 and Large 1), tested at both demand levels against the main mixed-demand policies and ORCA.
20. **Learning progress:** the main policies are also evaluated at snapshots taken at 3M and 6M steps. The learning-rate schedule spans the full 10M steps, so a snapshot shows a run part-way through, not a policy trained with a smaller budget.
21. **Holm correction family.** The analysis script applies Holm's correction within each metric, whereas the pre-registration specified correction across all conditions, metrics and baselines of a suite. The thesis reports per-condition counts recomputed with the stricter pre-registered correction. Hypothesis outcomes are decided on aggregate confidence intervals and probability of improvement, so none is affected.


## 4. Metric definitions

- **Orders per hour:** deliveries in a 10-minute simulated shift, scaled to one hour.
- **Congestion time proportion:** share of a robot's active time spent moving slower than 0.3 m/s when it should be moving.
- **Collisions per 100 timesteps:** contact steps (any robot-robot or robot-human pair touching) per 100 timesteps.
- **Separation violations per interaction:** an interaction is a robot and a person coming within 2 m; a violation is that pair then entering the 0.8 m ISO zone. Both count entries, so one long encounter counts once.
- **Metres per delivery** (secondary, travel efficiency): metres driven by all robots per delivery; seed units with no deliveries are excluded and counted.


## 5. References added beyond the proposal

- Agarwal, R. et al. (2021). Deep reinforcement learning at the edge of the statistical precipice. NeurIPS.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics.
- Rajeswaran, A. et al. (2018). Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. RSS.
- Silver, D. (2005). Cooperative pathfinding. AIIDE.
- Yu, C. et al. (2022). The surprising effectiveness of PPO in cooperative multi-agent games. NeurIPS.
