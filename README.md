# W2S: When-and-What to Schedule

Code, trained models and benchmark instances for

> K. Xie, J. Liu, Q. Liu. *Beyond Non-Delay Dispatching with a Factorized Deep Reinforcement Learning Policy for Flexible Job Shop Scheduling.* Submitted to Swarm and Evolutionary Computation.

W2S is a deep reinforcement learning policy for the flexible job shop scheduling problem (FJSP). At every step it first selects a
start time from the candidate start times that are earlier than the earliest completion time (timing head), and then an
operation–machine pair that can start at that time (pairing head). The decision process generates only active schedules and can
reach every strongly active schedule, hence an optimal one.

## Installation

```bash
pip install -r requirements.txt
```

The experiments in the paper used Python 3.12 and PyTorch 2.5.1; the released models were also checked with PyTorch 2.14.

## Repository layout

| Path | Content |
|---|---|
| `env/` | FJSP environment (factorized timing–pairing MDP) and the generator of training instances |
| `network/` | heterogeneous graph encoder, timing head, pairing head and critic |
| `train/`, `train.py` | PPO training |
| `evaluate.py` | greedy (-G), augmented (-A) and sampling (-S) inference |
| `utils/` | instance parser and order-preserving processing-time transforms |
| `config.yaml` | network, PPO and training settings |
| `checkpoints/` | trained models of the four sizes |
| `instances/` | synthetic instances of Song et al. (`song/`) and the public benchmarks |

## Evaluation

Greedy decoding (W2S-G), e.g. on Brandimarte with the 10×5 model:

```bash
python evaluate.py --checkpoint checkpoints/W2S_10x5.pt --benchmark instances/brandimarte --mode greedy --no_augment
```

- **W2S-A**: drop `--no_augment` (eight order-preserving processing-time transforms, best of the eight greedy solutions).
- **W2S-S*k***: `--mode sampling --num_samples k --no_augment`.

As in the paper, the public FJSP benchmarks and the Taillard and Demirkol JSP benchmarks are solved with the 10×5 model, the Behnke
instances with the 20×10 model, and each synthetic set with the model of its own size.

## Training

```bash
python train.py config.yaml
```

The training size and the instance distribution are set in `config.yaml` (`env.training_sizes`, `data_gen`), and the
validation set in `training.test_data_dir` / `training.val_subdir`.

The run directory `checkpoints/YYYY-MM-DD-NNN/` contains the configuration, the training log, TensorBoard files and
`best_test_model.pt`, the model with the best validation makespan.

## Instances

`instances/README.md` lists the instance sets, their sources and the file formats.

## License

The code is released under the MIT License (`LICENSE`). The benchmark instances keep the licenses of their sources
(`instances/README.md`).
