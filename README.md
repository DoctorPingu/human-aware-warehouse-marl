# Warehouse RL: how to finish the project

Mohammed Al-Khudhair, 31482 Honours Project

This folder contains everything needed to run Stages 2 to 4 of the thesis,
from training to the final figures. It replaces the earlier `warehouse_env_v4.py`
and `INTEGRATION.md` I sent; ignore those. Your old `training/` folder stays
untouched, so the proof-of-concept remains reproducible.


## 1. Where it goes

Copy the whole `warehouse_rl` folder into your project so it sits next to
`training`:

```
E:\Honours Project\habitat-lab\py scripts\
    training\          <- old files, leave alone
    warehouse_rl\      <- this folder
```


## 2. The files, in plain terms

| File | What it is | Do you edit it? |
|---|---|---|
| `run_all.py` | The one command you run. Runs every stage in order, in parallel, shows progress and time estimates, and can pause and resume. | No |
| `layouts.py` | The six warehouse maps as data: racks, docks, and the routes robots and humans follow. Checks itself for mistakes. | Only to change a map |
| `warehouse_env.py` | The simulation. Robots, humans, physics, rewards and the four thesis metrics. | No |
| `policies.py` | The four things being compared: ORCA, heuristic, MAPF and your trained PPO. | No |
| `networks.py` | The neural network shapes (same sizes as your proof-of-concept). | No |
| `curriculum.py` | Runs the curriculum arm: three training stages in a row, each starting from the last. `run_all.py` calls it for you. | No |
| `train.py` | Trains one PPO policy. `run_all.py` calls it for you. | No |
| `evaluate.py` | Plays 30 test episodes per algorithm per condition and records the results. | No |
| `analyze.py` | Turns results into tables, significance tests and thesis figures. | No |
| `make_video.py` | Side-by-side comparison videos for the pitch. | No |
| `planner.py` | Tool that generated safe routes for the maps. Only needed if you change a map. | No |
| `render_layouts_3d.py` | Optional 3D pictures of the six maps for slides. Does not affect results. | No |


## 3. One-time setup (10 minutes)

In your Ubuntu terminal:

```bash
cd "/mnt/e/Honours Project/habitat-lab/py scripts/warehouse_rl"
conda activate habitat
pip install pandas scipy matplotlib
sudo apt install ffmpeg            # only needed for videos
python run_all.py check
```

`check` validates the maps, runs the simulation, does a tiny training run and
a tiny evaluation, and prints how fast your machine is. If it says
"All checks passed", you are ready. If not, send me the output.

Why CPU and not your GPU: the networks are small enough that moving data to
the GPU costs more than it saves. The speed comes from running many training
jobs at once on your 16 CPU cores instead.


## 4. The seven-day plan

Training runs in the background, so you write while the computer works.

### Day 1: start the final batch
1. Do the setup above.
2. Start everything and leave it running:
   ```bash
   nohup python run_all.py all --jobs 12 --runs-dir runs_final > run_final.log 2>&1 &
   ```
   The defaults are the final configuration (section 8): 10 seeds (10 to
   19), 10 million steps, snapshots at 3M and 6M. It trains 140 policies
   and training stages, runs all nine evaluation suites, analyses them and
   renders one comparison video per layout. Expect about 27 hours; `status`
   gives a live estimate. Keep the Ubuntu window open (minimised is fine):
   on WSL, closing the last window can shut Linux down even with `nohup`.
   Do not let the PC sleep, and pause Windows Update for the weekend.
3. Watch it (see section 4a below).

### Days 2 and 3: look at the results
Open `results/analysis/summary.md` first, then the figures and videos. Also
open `results/analysis/training_main.png`: if the curves are still clearly
rising at the end, the policies needed longer. Rerun with
`--timesteps 6000000` if time allows: finished runs are extended from
their last checkpoint rather than restarted. One thing to report if you
do this: the learning-rate schedule is recalculated for the new total, so
the rate steps back up (from 1e-5 to 1.5e-4 when going from 3M to 6M)
before annealing again. That is a warm restart, not the same as a single
6M run, so say so in the methods chapter. Evaluation notices the
retrained policies and re-runs them; delete `videos/` first if you want
the videos redone too.

Stages can also be run one at a time: `train`, `calibrate`, `ablation`,
`evaluate`, `analyze`, `video`.

### 4a. Watching, pausing and resuming

| What | Command |
|---|---|
| Progress, per-run speed, time left and finish time | `python run_all.py status` |
| The same, refreshing every 30 seconds | `watch -n 30 python run_all.py status` |
| The running log (a progress line every minute) | `tail -f run.log` (Ctrl+C only stops watching) |
| One training run in detail | `tail -f logs/main_large_2_robotic_grid_seed0.log` |
| Pause: everything saves and stops within about a minute | `python run_all.py pause` |
| Continue after a pause, crash, closed window or power cut | `nohup python run_all.py resume >> run.log 2>&1 &` |

What an unexpected stop costs: training checkpoints about every 100,000
steps (roughly a minute), so each running job repeats at most that much.
Finished evaluation jobs are cached, and a video is only kept once it is
complete. Status tells you if the run stopped unexpectedly and how to
continue. Starting a second run while one is going is refused, because
two runs writing to the same folders would corrupt each other.

One honest caveat for the thesis: a run that was paused or interrupted
and resumed is statistically equivalent to an uninterrupted one, but not
bit-for-bit identical (the simulated environments restart their current
episodes on resume). Say so if any of your reported runs were resumed.

The time estimates use each run's measured speed and simulate the job
queue, so they settle down after the first 10 to 15 minutes. The clock
times come from Ubuntu's clock; if they look off by some hours, WSL's time
zone differs from Windows, and the durations are still right.

### Days 4 and 5: write the results and discussion
Use the tables and figures directly. Suggested structure:
1. Main comparison (Stage 3): the four metrics across 24 conditions.
2. Reward calibration: v1 against v2.
3. Ablation: what each part of the reward does.
4. Generalisation (Stage 4): unseen layouts and 150% density.
5. Discussion: where PPO helped, where it did not, and why.

### Day 6: conclusion, videos, pitch slides
The comparison videos are already in `videos/` from Day 1. For a custom
one (different length, density or algorithms), for example:
```bash
python make_video.py --layout medium2 --algos orca,ppo --seconds 60
```
Write the conclusion, limitations and future work. The rubric wants a
recommendation for extending the project to a research degree, so say
what you would do with more time and compute.

### Day 7: record the pitch, proofread, buffer
Keep this day free. Something always takes longer than planned.


## 5. What the results will and will not show

Be ready for any outcome, because any of them is a legitimate finding.

A test run (one seed, 500k steps, 10 episodes) came out statistically level
with ORCA: similar throughput, slightly fewer collisions, not significant.
The full runs are six times longer with three seeds each, so expect clearer
separation, but I cannot promise PPO will beat every baseline everywhere.
If it does not, that is still a result: it tells you where learning adds
value and where a simple reactive controller is already good enough. An
honest negative result, well analysed, scores better than an overclaimed one.


## 6. Changes to write up in the thesis

Your methodology chapter already allows for changes ("Note on Scope and
Changes"). These need a short paragraph each. Most make the study more
rigorous, which examiners like.

### 6.1 Two bugs in the proof-of-concept numbers (important)
The collision count in the proof-of-concept was not measuring robot
collisions at all. The threshold was 0.44 m, but the physics never lets a
robot and a human get closer than 0.45 m (their radii added), or two robots
closer than 0.50 m. So robot collisions could never register. Only two humans
bumping into each other (0.40 m) ever crossed it. Every one of the ~16.5
"collisions" per episode in your training log was human-human, which is why
that curve stayed flat.

Separately, the old environment had no randomness, so every episode was
identical and 30 repeated runs would have given the same number 30 times.

Both are fixed. Do not quote the old collision numbers as robot collisions.
Say what was found and that the new environment corrects it. Finding and
fixing this yourself is evidence of rigour, not a weakness.

### 6.2 Other changes
1. Shared actor and per-robot critic (MAPPO, Yu et al., 2022), so one policy
   works across layouts with different robot counts. Without this, the
   generalisation stage was impossible.
2. Imitation warm start: PPO starts from a copy of the ORCA controller, held
   near it early on and then released (Damani et al., 2021, already in your
   references; Rajeswaran et al., 2018). Pure PPO learned too slowly for the
   timeline. This also sharpens the research question: what does learning
   add on top of a reactive controller?
3. Reward v2: a collision is penalised when it happens, not for every step
   agents stay touching; the safety penalty grows with how far a robot
   intrudes into the ISO zone; the time penalty was shrunk because it added
   a constant -200 to every episode and taught the agent nothing. Measured
   evidence: with the old per-step collision penalty, the policy learned to
   stop moving (throughput fell from 351 to about 50 orders per hour).
4. MAPF is implemented as prioritised cooperative A* (Silver, 2005) rather
   than conflict-based search. CBS solves one-shot problems; your robots get
   new tasks continuously, which needs replanning. Prioritised planning is
   the standard approach for that.
5. Six layouts, not nine. The original 3 x 3 x 2 x 2 design gave 36
   conditions, not the 24 your chapter states; six layouts gives exactly 24.
6. Training mixes both demand levels, because both are evaluated. The
   out-of-distribution tests are unseen layouts and 150% agent density.
7. ORCA/RVO2 runs underneath every controller as the physics layer, so the
   comparison is between decision-making layers on the same substrate.
8. The simulation is a custom 2D Gymnasium environment with RVO2. It does
   not run inside Habitat 3.0. Your methods chapter currently says the
   infrastructure is "built on Habitat 3.0"; soften that to say Habitat
   was the development environment and 3D integration is future work,
   otherwise an examiner who reads the code will notice.

9. Episodes last 10 simulated minutes (14,400 steps), not 83 seconds
   (2,000 steps). The pilot showed the short episodes could not measure
   throughput: robots completed only 4 to 8 deliveries, so orders per
   hour moved in steps of 43.2 and all four controllers tied. Over 10
   minutes the differences appear (for example the heuristic's stand-offs
   build up and cut its throughput by about a quarter in Medium 1). The
   same test found that the imitation-only starting policy jams over long
   episodes (68 orders per hour against ORCA's 348 on Small 1) and that
   PPO training steadily removes this, which is worth a sentence in the
   results as evidence of what learning adds.

10. Sensitivity analysis (the methods chapter promises one): human
    walking speed 20% lower and higher, and a wider human avoidance
    radius (0.30 m and 0.40 m against the 0.20 m body radius), applied at
    evaluation time only, on Small 1 and Large 1 at high density. Check
    these levels against the ATC figures you cite and adjust
    `PERTURBATIONS` in evaluate.py if the documented range differs.
11. Two planned tests were replaced, and need a sentence each: the
    small-to-large training curriculum (each layout now has its own
    policy, which is what makes the full cross-layout transfer matrix
    possible), and standard-to-peak demand transfer (training mixes both
    demand levels, see item 6).
12. "Thirty independent runs per condition" is implemented as 30
    evaluation episodes per condition, spread over 3 independently
    trained policies (10 each). Word it that way.

13. Development batches. Three rounds of development runs preceded the
    final batch and belong in the methods chapter as the process that
    shaped it: a pilot (300k steps, 83-second episodes) that exposed the
    episode-length problem (item 9); a 3M-step batch (3 seeds) whose
    training curves showed PPO held near its imitation start for half of
    training; and a 6M-step batch (3 seeds) that halved the throughput gap
    to ORCA while improving safety, but had not converged. Their results
    are kept in `results_3m` and `results_6m`.
14. Final batch design (section 8), decided before it ran: 10M steps by a
    convergence rule set in advance; the imitation anchor fades over a
    fixed 2M steps instead of half of training (at 10M, half would have
    held PPO near the imitation for 5M steps); 10 fresh training seeds
    never used in development; a fresh held-out set of evaluation
    episodes; and pre-registered hypotheses (`PREREGISTRATION.md`).
15. Seed-level statistics (Agarwal et al., 2021). Each seed's episodes in a
    condition are averaged into one "seed unit", and every test, interval
    and figure uses those units, so the unit of evidence is a trained
    policy, not an episode. Baselines are grouped the same way, on the
    same evaluation episodes as the matching PPO seed. Probability of
    improvement is reported alongside the significance tests.
16. PPO hyperparameters differ from the proof-of-concept values in the
    methods chapter (rollout 512, 4 epochs, minibatch 128): the toolkit
    uses 8 parallel environments x 256 steps, 10 epochs, minibatch 512,
    chosen for vectorised training and kept unchanged across all
    development and final runs. Learning rate (3e-4 to 1e-5), clip (0.2)
    and the network (128-128-64, Tanh) match the chapter.
17. Robot density stress test (Objective 3, FAQ): 150% robots (3 become 5,
    4 become 6), evaluation only, at standard human density. Each extra
    robot starts in the staging area midway between two existing robots
    and follows one of their validated routes, so extra traffic uses the
    same safe paths. The critic's global state only covers 4 robots, which
    does not matter at evaluation time, where only the actor runs.
18. Curriculum arm: Small 1 at standard demand (3M steps, with the
    imitation warm start), then Medium 1 (3M), then Large 1 (4M), both
    with both demand levels, each stage starting from the previous stage's
    policy. Each stage has its own learning-rate schedule. Compared on
    Large 1 against the policy trained there directly for 10M steps.
19. Demand transfer arm: policies trained on standard demand only (Small 1
    and Large 1), tested at both demand levels against the main
    mixed-demand policies and ORCA.
20. Resume fidelity: the imitation demonstrations are saved to disk, so a
    run paused or interrupted during its first 2M steps resumes with the
    anchor intact.

### 6.3 Metric definitions for the methods chapter
- Orders per hour: deliveries in a 10-minute simulated shift, scaled to one hour.
- Travel efficiency (secondary): metres driven by all robots per
  delivery; episodes with no deliveries are excluded and counted.
- Congestion time proportion: share of a robot's active time spent moving
  slower than 0.3 m/s when it should be moving.
- Collisions per 100 timesteps: contact steps (any robot pair or
  robot-human pair touching) per 100 timesteps.
- Separation violations per interaction: an interaction is a robot and a
  human coming within 2 m; a violation is that pair then entering the 0.8 m
  ISO zone. Both count entries, so one long encounter counts once.

### 6.4 References to add
- Yu, C. et al. (2022). The surprising effectiveness of PPO in cooperative
  multi-agent games. NeurIPS.
- Rajeswaran, A. et al. (2018). Learning complex dexterous manipulation with
  deep reinforcement learning and demonstrations. RSS.
- Silver, D. (2005). Cooperative pathfinding. AIIDE.
- Agarwal, R. et al. (2021). Deep reinforcement learning at the edge of the
  statistical precipice. NeurIPS.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  Scandinavian Journal of Statistics.


## 8. The final batch at a glance

| | |
|---|---|
| Training seeds | 10 to 19 (development used 0 to 2) |
| Length | 10M steps; snapshots kept at 3M and 6M |
| Imitation anchor | fades over the first 2M steps |
| Policies | main (6 layouts), v1 calibration, throughput-only and safety-only ablations (Small 1), standard-demand arm (Small 1, Large 1), curriculum (Small 1 to Medium 1 to Large 1) |
| Evaluation episodes | 30 per condition, 3 per seed, from a fresh held-out seed block |
| Suites | main, calibration, ablation, generalisation (unseen layouts, 150% humans), sensitivity, robot_stress, demand, curriculum, progress |

## 7. If something goes wrong

- A training run failed: read the last lines of its file in `logs/`, then
  `python run_all.py resume`. Finished runs are skipped.
- It stopped (closed window, crash, restart): `python run_all.py status`
  confirms it, then `nohup python run_all.py resume >> run.log 2>&1 &`.
- Everything is slow: lower `--jobs`. On WSL, copying the folder into your
  Linux home (`cp -r ... ~/warehouse_rl`) and running it there is faster than
  running from `/mnt/e`; copy `results/` back when done.
- You edited a map: run `python layouts.py` before anything else. It will
  tell you exactly which route hits a rack.
- Anything else: send me the error text.
