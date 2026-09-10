# GSF-GNN

Implementation for the paper [*Global structure-aware and feature-augmented graph neural network for heterophilic graphs*](https://dl.acm.org/doi/full/10.1145/3775057), published in *ACM Transactions on Information Systems*.

![Overview of the GSF-GNN architecture](assets/gsf-gnn-overview.png)

*GSF-GNN combines structure-based global propagation with a feature-augmented compensatory update. Figure adapted from the associated paper.*

## What is included

- Native PyTorch graph propagation; DGL, PyTorch Geometric, compiled CUDA extensions, and graph-framework packages are not required.
- The 21 migrated NPZ datasets in `data/`. See [DATASETS.md](DATASETS.md) for exact filenames, expected fields, and integrity checking.
- A no-download smoke test, a dataset validator, and a configurable training entry point.

Each dataset file contains `node_features`, `node_labels`, `edges`, `train_masks`, `val_masks`, and `test_masks`.

## Installation

The verified environment is Windows, Python 3.10, PyTorch 2.7.0+cu128, NumPy 1.24.3, and tqdm 4.67.3 (tested on an RTX 5090).

```bash
git clone <YOUR_REPOSITORY_URL>
cd GSF-GNN
python -m pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 12.8 PyTorch build. For a CPU-only machine, install `requirements-cpu.txt` instead. `requirements-dev.txt` adds test/lint tools; `environment.yml` is the equivalent Conda entry point.

## Verify the checkout

```bash
python smoke_test.py
python verify.py
python train.py --dataset synthetic --device cpu --num-steps 2 --num-layers 1 --hidden-dim 8 --num-heads 2
```

`smoke_test.py` checks forward and backward passes without data downloads. `verify.py` validates every bundled NPZ file before training.

## Train

Short GPU run on Actor:

```bash
python train.py --dataset actor --device cuda --num-layers 2 --hidden-dim 64 --num-heads 8 --num-steps 200 --seed 42
```

Standard ten-run Actor command:

```bash
python train.py --dataset actor --device cuda --num-layers 2 --hidden-dim 64 --num-heads 8 --num-steps 200 --num-runs 10 --seed 42 --output-dir experiments/actor-10runs
```

The command selects the validation-best evaluation from each split and writes a JSON summary to `<output-dir>/<dataset>_results.json`. On the verified machine, the ten-run Actor command produced `0.3711 ± 0.0077` test accuracy (mean ± population standard deviation). Exact values may vary with PyTorch, CUDA, and GPU kernels.

`--device auto` uses CUDA when available. Avoid CPU for large datasets: GSF-GNN constructs a dense node-similarity matrix with O(N²) memory. Use `--disable-filters`, `--disable-combinations`, and `--no-global-node` for the three ablations.

## Reproducibility

`--seed` controls Python, NumPy, and PyTorch RNGs. Multi-run experiments use `seed + run_index` and rotate through the dataset’s supplied train/validation/test masks. Saved JSON records every selected step, validation score, and test score.

## Citation

If you use this implementation, please cite the GSF-GNN paper:

```bibtex
@article{liu2025global,
  title={Global structure-aware and feature-augmented graph neural network for heterophilic graphs},
  author={Liu, Huijie and Ruan, Shulan and Liu, Qi and Cheng, Mingyue and Huang, Zhenya and Liu, Yu and Chen, Enhong and He, You},
  journal={ACM Transactions on Information Systems},
  volume={44},
  number={2},
  pages={1--28},
  year={2025},
  publisher={ACM New York, NY}
}
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## Release and data notice

The implementation was cleaned from the supplied local research codebase. Dataset archives are redistributed as provided and retain their original provenance; check [DATASETS.md](DATASETS.md) and the upstream dataset terms before redistribution or commercial use.

The supplied local codebase contains a Yandex Research MIT notice, but the upstream GSF-GNN repository does not clearly license this cleaned standalone implementation. Do not attach a new code license or publish it as an open-source release until the relevant rights holder confirms the intended license.
