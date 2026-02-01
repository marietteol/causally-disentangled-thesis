# Causal Bottlenecks for Demographic-Invariant Speaker Representations

This repository implements a causal bottleneck framework for disentangling
demographic attributes from speaker representations.

## Method
- CNN encoder with statistics pooling
- Linear causal bottleneck
- Adversarial demographic removal (GRL)
- Orthogonality regularization

## Structure
- `models/` – encoders, bottlenecks, probes
- `training/` – training loops
- `evaluation/` – speaker verification and leakage metrics
- `experiments/` – experiment drivers

## Running Experiments
```bash
python experiments/run_experiments.py --config configs/default.yaml
