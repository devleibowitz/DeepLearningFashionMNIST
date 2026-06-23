# Deep Learning on FashionMNIST

This repository presents a set of deep learning assignments centered on the **FashionMNIST** dataset, with a focus on comparing learning strategies across supervised and semi-supervised settings.

The project is designed as a practical exploration of how different model families and regularization techniques affect representation quality and classification performance on grayscale fashion images.

## Project Goals

- Build and train core deep learning models on FashionMNIST.
- Compare training behavior under different regularization methods.
- Explore latent-space learning for low-label (semi-supervised) regimes.
- Provide reproducible notebooks, saved models, and visual outputs.

## Implemented Algorithms

### 1) LeNet-5 for Supervised Classification

Implemented in the `HW1` section of the repository, this track benchmarks the classic **LeNet-5** architecture on FashionMNIST.

Model variants include:
- **Baseline LeNet-5** (no explicit regularization)
- **LeNet-5 + Dropout**
- **LeNet-5 + Weight Decay (L2 regularization)**
- **LeNet-5 + Batch Normalization**

Purpose:
- Evaluate how regularization choices influence generalization, overfitting, and convergence.

### 2) Semi-Supervised Learning with Deep Generative Models (M1-style)

Implemented in `HW3`, this track follows the M1 idea from Kingma et al.:

- Train a **Variational Autoencoder (VAE)** to learn latent representations of FashionMNIST images.
- Train an **SVM classifier** on the learned latent features, using limited labeled data.

Purpose:
- Study feature learning in low-label settings.
- Measure performance across multiple labeled sample budgets.

## Repository Structure

- `HW1/` — Supervised FashionMNIST experiments with LeNet-5 and regularization variants.
- `HW3/` — Semi-supervised FashionMNIST experiments using VAE-based representations.
- `FINAL_PROJECT/` — Additional deep learning project work beyond FashionMNIST.

## Typical Workflow

1. Open the notebook for the relevant task (`HW1` or `HW3`).
2. Run training cells to reproduce experiments.
3. Review saved outputs (plots, checkpoints, and reported metrics).
4. Optionally load pretrained artifacts for evaluation/inference.

## Tech Stack

- Python
- PyTorch
- scikit-learn
- Jupyter Notebooks

## Results (High-Level)

Across the assignments, the repository demonstrates:
- Strong supervised baselines on FashionMNIST with LeNet-5.
- Observable regularization trade-offs between stability, speed, and test performance.
- Effective semi-supervised classification gains from VAE-based feature learning when labels are scarce.

## References

- [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Y. LeCun et al. - Gradient-Based Learning Applied to Document Recognition (LeNet)](https://ieeexplore.ieee.org/document/726791)
- [D. P. Kingma et al. - Semi-Supervised Learning with Deep Generative Models](https://arxiv.org/abs/1406.5298)

---

If you are visiting this repo for a quick start, begin with `HW1/lenet5.ipynb` for supervised experiments, then continue to `HW3/problem3_semisupervised.ipynb` for semi-supervised learning.
