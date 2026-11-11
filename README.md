# Human-Aware Multi-Agent Reinforcement Learning for Warehouse Coordination

Deep reinforcement learning for coordinating warehouse robots that share their aisles with people.

Mohammed Al-Khudhair - Bachelor of Computing Science (Honours), University of Technology Sydney
Honours Project 31482, Spring 2026 · Supervisor: Dr Mingshan Jia

> Status: complete. Research code, results and videos accompanying the Honours thesis submitted in November 2026.


## Overview

Autonomous mobile robots increasingly work alongside people in warehouses. Classical coordination methods were largely designed for robot-only floors, and around people they tend either to stop and idle or to come too close.

This project trains a fleet coordination policy with multi-agent deep reinforcement learning, using centralised training with decentralised execution (CTDE) and Proximal Policy Optimisation (PPO), and a reward that penalises entering the 0.8 m safety zone around people (informed by ISO 10218 and ISO/TS 15066). It compares the learned policy against three classical controllers across six warehouse layouts and 24 experimental conditions, and tests how well it holds up beyond its training conditions.


## Key results

Main comparison: 24 conditions, 10 independently trained seeds, 2,880 ten-minute evaluation episodes. Interquartile means over seed units.

| Controller | Orders per hour | Collisions per 100 steps | ISO-zone entries per interaction | Congestion | Metres per delivery |
|---|---|---|---|---|---|
| ORCA / RVO | 302 | 8.31 | 0.69 | 0.03 | 55.1 |
| Rule-based heuristic | 257 | 5.07 | 0.44 | 0.15 | 56.2 |
| MAPF (prioritised A*) | 300 | 7.73 | 0.70 | 0.03 | 55.5 |
| CTDE PPO (this work) | 279 | 4.48 | 0.52 | 0.02 | 62.8 |

- Against ORCA, the learned policy has 46% fewer collisions and 25% fewer ISO-zone entries, for 7.5% lower throughput. It has the fewest collisions and the lowest congestion of all four controllers.
- The safety advantage holds with denser crowds, more robots than in training, and changed human behaviour (walking speed and avoidance radius).
- The reward sets the operating point: rewarded for throughput alone, the same algorithm beats ORCA (364 against 350 orders per hour on Small 1).
- Throughput was still rising at 10M training steps (220, 249 and 279 orders per hour at 3M, 6M and 10M), so the policies had not converged.
- Nine of ten pre-registered hypotheses were supported (see [docs/PRE-REGISTRATION.md](docs/PRE-REGISTRATION.md)).

Full tables, significance tests and figures for every suite are in [results/analysis/summary.md](results/analysis/summary.md). Side-by-side videos of all four controllers on each layout are in [videos/](videos).


## Repository structure

```
.
├── src/
│   ├── run_all.py              Runs every stage: check, train, evaluate, analyse, video; with status, pause and resume
│   ├── warehouse_env.py        The simulation: robots, people (RVO2), rewards and metrics (Gymnasium)
│   ├── layouts.py              The six warehouse layouts as data, with a self-check for route clearances
│   ├── policies.py             The four controllers: ORCA/RVO, heuristic, MAPF and the trained PPO policy
│   ├── networks.py             Actor and critic networks
│   ├── train.py                Trains one CTDE PPO policy
│   ├── curriculum.py           Runs the three-stage curriculum arm
│   ├── evaluate.py             Runs the evaluation suites on held-out episodes
│   ├── analyze.py              Seed-level statistics, tables and figures
│   ├── make_video.py           Side-by-side comparison videos
│   ├── planner.py              Route generator used to build the layouts
│   └── render_layouts_3d.py    Illustrative 3D renders of the layouts
├── results/                    Final results: one CSV per suite, plus results/analysis (tables and figures)
├── development/                Earlier work, kept as a record of how the method evolved
│   ├── proof_of_concept/       The first environment, trainer and demo scripts
│   ├── pilot/                  Pilot batch results
│   ├── 3m/                     3M-step development batch results
│   └── 6m/                     6M-step development batch results
├── videos/                     Side-by-side comparison videos of the four controllers, one per layout
├── docs/
│   ├── PRE-REGISTRATION.md     Hypotheses and analysis plan, registered before the final run
│   └── METHODOLOGY-NOTES.md    Every change from the original methodology, with reasons
└── requirements.txt
```

All commands are run from the repository root, and training runs, logs and videos are written there too. Trained policies and training logs are not stored in the repository because of their size. The six final comparison videos are included in `videos/`.


## Setup

Tested on Ubuntu (WSL 2) with Python 3.9.

```bash
conda create -n warehouse python=3.9
conda activate warehouse
pip install -r requirements.txt
```

Two dependencies are installed separately:

1. RVO2 Python bindings, built from source from [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2):
   ```bash
   pip install "cython<3" cmake
   git clone https://github.com/sybrenstuvel/Python-RVO2.git
   cd Python-RVO2
   python setup.py build
   python setup.py install
   ```
   With CMake 4 or newer, the build may need `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` added to the CMake call in `setup.py`.
2. ffmpeg, only needed for videos: `sudo apt install ffmpeg`.

Then confirm everything works:

```bash
python src/run_all.py check
```

This validates the layouts, runs the simulation, performs a short training run and evaluation, and reports how fast your machine trains. The networks are small, so training runs on the CPU, with many jobs in parallel.


## Reproducing the results

The complete final experiment (140 training runs, nine evaluation suites, analysis and videos):

```bash
nohup python src/run_all.py all --jobs 12 --runs-dir runs_final > run_final.log 2>&1 &
```

The defaults are the final configuration: 10 seeds (10 to 19), 10M steps per run, snapshots at 3M and 6M. On an Intel i9-12900K this took about 20.5 hours. Set `--jobs` to about half your CPU threads.

| To | Run |
|---|---|
| See progress, speed and time left | `python src/run_all.py status` |
| Pause (everything saves within about a minute) | `python src/run_all.py pause` |
| Continue after a pause, crash or restart | `nohup python src/run_all.py resume >> run_final.log 2>&1 &` |
| Run one stage | `python src/run_all.py train`, `evaluate`, `analyze` or `video` |
| Regenerate tables and figures from the saved CSVs | `python src/analyze.py` |

Every stage is resumable: finished training runs are skipped, evaluation jobs are cached, and all randomness is seeded.


## Experimental design

| | |
|---|---|
| Layouts | 6: parallel-aisle, L-shaped, twin-block, fishbone, fulfilment centre, robotic storage grid |
| Conditions | 24 = 6 layouts × 2 demand levels × 2 human densities |
| Agents | 3 to 4 robots and 2 to 7 people; people and robots avoid each other (RVO2) |
| Controllers | ORCA/RVO, rule-based heuristic, MAPF (prioritised cooperative A*), CTDE PPO |
| Learning | Shared actor across robots, centralised critic during training only, imitation warm start from ORCA fading over 2M steps |
| Training | 10 seeds × 10M steps per policy |
| Evaluation | 30 ten-minute episodes per condition, from seeds never used in training |
| Metrics | Orders per hour, congestion time, collisions per 100 steps, ISO-zone entries per interaction; metres per delivery |
| Statistics | Seed-level interquartile means with stratified bootstrap 95% CIs, Mann-Whitney U tests with Holm correction, probability of improvement (Agarwal et al., 2021) |

Beyond the main comparison, the suites cover reward calibration, reward ablations, unseen layouts, denser crowds, more robots, sensitivity to human behaviour, a standard-demand-only training arm, a small-to-large curriculum, and learning progress at 3M, 6M and 10M steps.


## Development history

The final design came out of three development batches, whose results are kept in `development/`:

1. A pilot showed that 83-second episodes were too short to measure throughput, so episodes became ten simulated minutes.
2. A 3M-step batch showed the policy staying close to its imitation start for half of training.
3. A 6M-step batch halved the throughput gap to ORCA while improving safety, but had not converged.

The development work also found and fixed a collision metric in the proof of concept that could not register robot collisions. [docs/METHODOLOGY-NOTES.md](docs/METHODOLOGY-NOTES.md) records every change from the original methodology and why it was made.


## Limitations

- Training had not converged at 10M steps.
- The safety level is set through reward weights, not a safety budget an operator can state.
- The simulation is 2D, with simulated people, and has not been tested on real robots.


## Citation

```
Al-Khudhair, M. (2026). Deep Reinforcement Learning for Multi-Agent Coordination
in Mixed Human-Robot Warehouse Environments. Honours thesis, University of
Technology Sydney.
```


## Licence

Released under the MIT Licence. See [LICENSE](LICENSE).


## Acknowledgements

Supervised by Dr Mingshan Jia, University of Technology Sydney. Built on [RVO2](https://gamma.cs.unc.edu/RVO2/) (van den Berg et al.), [Gymnasium](https://gymnasium.farama.org/) and [PyTorch](https://pytorch.org/).
