# DPImageBench

Differentially Private Synthetic Image Generation Benchmark.

A unified benchmark for training and evaluating differentially private image synthesizers, supporting 12+ methods across multiple datasets.

## Supported Methods

| Method | `--method` | Description |
|---|---|---|
| DP-MERF | `DP-MERF` | Random Fourier Features |
| DP-NTK | `DP-NTK` | Neural Tangent Kernel |
| DP-Kernel | `DP-Kernel` | Kernel-based |
| GS-WGAN | `GS-WGAN` | Wasserstein GAN with GS |
| DPGAN | `DPGAN` | DP Generative Adversarial Network |
| DPDM | `DPDM` | DP Diffusion Model |
| PDP-Diffusion | `PDP-Diffusion` | Pre-trained DP Diffusion |
| DP-LDM-SD | `DP-LDM-SD` | Latent Diffusion with Stable Diffusion |
| DP-LDM | `DP-LDM` | DP Latent Diffusion Model |
| DP-LoRA | `DP-LORA` | DP Low-Rank Adaptation |
| PE | `PE` | Private Ensemble |
| **DPIS-SQ** | `DPIS-SQ` | DP Image Synthesis with Semantic Query |
| **DPIS-MI** | `DPIS-MI` | DP Image Synthesis with Mode Images |

## Datasets

| Dataset | Resolution | Classes | `--data_name` |
|---|---|---|---|
| MNIST | 28 | 10 | `mnist_28` |
| Fashion-MNIST | 28 | 10 | `fmnist_28` |
| CIFAR-10 | 32 | 10 | `cifar10_32` |
| CIFAR-100 | 32 | 100 | `cifar100_32` |
| EuroSAT | 32 | 10 | `eurosat_32` |
| CelebA (Male) | 32/64/128 | 2 | `celeba_male_32` |
| Camelyon17 | 32 | 2 | `camelyon_32` |

## Installation

```bash
# Clone and set up conda environment
git clone <repo-url>
cd DPImageBench
conda create -n dpimagebench python=3.9
conda activate dpimagebench
bash install.sh
```

## Data Preparation

Preprocess datasets into the internal zip format:

```bash
python data/preprocess_dataset.py \
    --data_name mnist fmnist cifar10 \
    --data_dir ./dataset
```

This downloads raw data, resizes images, packs them into `train_<res>.zip` / `test_<res>.zip`, and computes `fid_stats_<res>.npz`.

For DPIS-SQ / DPIS-MI, ImageNet pretrained models are required:

```bash
# Place these files in models/pretrained_models/
# - imagenet_classifier_ckpt.pth   (ResNet50 classifier)
# - imagenet64_cond_270M_250K.pt   (EDM diffusion checkpoint)
```

## Usage

### Quick Demo

```bash
bash scripts/demo_dpis.sh
```

Runs both DPIS-SQ and DPIS-MI on MNIST-28 with ε=1.0 (minimal epochs). Edit `DEMO_PARAMS` in the script to adjust speed vs. quality.

### Training

```bash
# Single method
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    run.py setup.run_type=torchrun \
    -m DPIS-SQ -dn mnist_28 -e 10.0 -ed my_experiment

# With custom parameters
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    run.py setup.run_type=torchrun \
    pretrain.n_epochs=3200 train.n_epochs=150 \
    -m DPIS-SQ -dn cifar10_32 -e 10.0 -ed full_train
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `-m`, `--method` | `DP-LDM` | Method name |
| `-dn`, `--data_name` | `cifar10_32` | Dataset |
| `-e`, `--epsilon` | `10.0` | Privacy budget |
| `-ed`, `--exp_description` | `""` | Experiment tag |
| `pretrain.n_epochs` | 3200 | Pretrain epochs |
| `train.n_epochs` | 150 | DP-SGD epochs |
| `train.dp.max_grad_norm` | 0.001 | Gradient clipping norm |
| `gen.data_num` | 60000 | Number of synthetic images to generate |

All config YAML keys can be overridden via CLI (OmegaConf dotlist format).

### Resume Training

```bash
python run.py -re <experiment_dir_name> -m DPIS-SQ -dn mnist_28 -e 10.0
```

## Pipeline

Each training run executes 6 phases:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. Config    │───▶│ 2. Data      │───▶│ 3. Pretrain  │
│    Parsing   │    │    Loading   │    │ (public data)│
└──────────────┘    └──────────────┘    └──────────────┘
                                                │
┌──────────────┐    ┌──────────────┐    ┌───────▼──────┐
│ 6. Evaluate  │◀───│ 5. Generate  │◀───│ 4. DP-SGD    │
│ (Acc + FID)  │    │ (syn images) │    │    Train     │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Method Comparison

| Phase | Standard Methods | DPIS-SQ | DPIS-MI |
|---|---|---|---|
| Public data | None or full dataset | Semantically-filtered ImageNet | Central (mean/mode) images |
| Pretrain | None | Diffusion on public subset | Diffusion on central images |
| DP Training | DP-SGD from scratch | DP-SGD fine-tuning | DP-SGD fine-tuning |

### DPIS-SQ Semantic Query

DPIS-SQ uses a pretrained ResNet50 to bridge the semantic gap between sensitive and public data:

```
MNIST image → ResNet50(ImageNet) → Top-5 ImageNet classes
                                              ↓
                        Gaussian noise (σ=50) added to histogram
                                              ↓
                        50 unique ImageNet classes selected
                        (5 per MNIST digit, no overlap)
```

### DPIS-MI Central Images

DPIS-MI creates differentially private "prototype" images from sensitive data:

- **Mean mode**: DP mean image per class, downsampled + noisy
- **Mode mode**: DP histogram mode per pixel, per class

These serve as a privacy-preserving bridge for pretraining.

## Evaluation Metrics

After generation, the evaluator computes:

| Metric | Description |
|---|---|
| **Accuracy** | Train ResNet/WRN/ResNeXt on synthetic images, test on real data |
| **FID** | Fréchet Inception Distance (InceptionV3 pool3 features) |

Accuracy uses DP-aware model selection: Laplace noise is added to validation accuracy to simulate private hyperparameter tuning.

## Project Structure

```
DPImageBench/
├── run.py                  # Entry point
├── eval.py                 # Standalone evaluation
├── install.sh              # Dependency installation
├── configs/                # YAML configs per method/dataset
│   ├── DPIS-SQ/            # DPIS-SQ params
│   └── DPIS-MI/            # DPIS-MI params
├── models/                 # Model implementations
│   ├── DPIS_SQ/            # Semantic query (ResNet50 classifier)
│   ├── DPIS_MI/            # Mode image query
│   ├── DP_Diffusion/       # EDM diffusion model
│   └── dpsgd_diffusion.py  # DP-SGD diffusion training
├── data/                   # Data loading and preprocessing
│   ├── dataset_loader.py   # Runtime data pipeline
│   ├── preprocess_dataset.py
│   └── SpecificImagenet.py # Semantic class filtering
├── evaluation/             # Evaluator (accuracy + FID)
├── opacus/                 # Modified Opacus with DPDDP
├── utils/                  # Config parsing, distributed launcher
├── scripts/                # Shell scripts for experiments
│   └── demo_dpis.sh        # Quick demo
├── plot/                   # Plotting scripts for figures
└── exp/                    # Experiment outputs (generated)
```

## Output

Each run creates a directory under `exp/<method>/<dataset>_eps<epsilon><desc>-<timestamp>/`:

```
exp/dpis-sq/mnist_28_eps10.0demo-2026-05-21-01-46-08/
├── stdout.txt              # Full training log
├── pretrain/
│   ├── checkpoints/
│   └── samples/
├── train/
│   ├── checkpoints/
│   └── samples/
└── gen/
    ├── gen.npz              # Synthetic images (x) and labels (y)
    └── sample.png           # Sample grid
```

## Known Issues

### NCCL + NVIDIA RTX 5090 (Blackwell)

The built-in NCCL in PyTorch 2.7.1 has a compatibility issue with RTX 5090 GPUs. The workaround is using the `gloo` backend instead of `nccl` (already applied in `utils/utils.py`). Gloo uses CPU-side communication and is slower but functionally correct.

When NCCL is fixed in a future PyTorch release, revert `dist.init_process_group("gloo")` back to `dist.init_process_group("nccl")` in `utils/utils.py` for better multi-GPU performance.

## License

MIT License. See [LICENSE](LICENSE) for details.
