# DPImageBench

Differentially Private Synthetic Image Generation Benchmark.

## Methods

| `--method` | Description | Config Dir |
|---|---|---|
| `DP-MERF` | Random Fourier Features | `configs/DP-MERF/` |
| `DP-Kernel` | Kernel-based | `configs/DP-Kernel/` |
| `DPGAN` | DP Generative Adversarial Network | `configs/DPGAN/` |
| `DPDM` | DP Diffusion Model | `configs/DPDM/` |
| `PDP-Diffusion` | Pre-trained DP Diffusion | `configs/PDP-Diffusion/` |
| `DPIS-SQ` | DP Image Synthesis with Semantic Query | `configs/DPIS-SQ/` |
| `DPIS-MI` | DP Image Synthesis with Mode Images | `configs/DPIS-MI/` |

DPIS-SQ and DPIS-MI are the primary methods; the rest are baselines with varying levels of code completeness.

## Datasets

All preprocessed data stored as zip archives under `dataset/`:

| Dataset | `--data_name` | Resolution | Channels | Classes | Preprocessed |
|---|---|---|---|---|---|
| MNIST | `mnist_28` | 28 | 1 | 10 | train/test + fid_stats |
| Fashion-MNIST | `fmnist_28` | 28 | 1 | 10 | train/test + fid_stats |
| CIFAR-10 | `cifar10_32` | 32 | 3 | 10 | train/test + fid_stats |
| CIFAR-100 | `cifar100_32` | 32 | 3 | 100 | train/test + fid_stats |
| EuroSAT | `eurosat_32` | 32 | 3 | 10 | train only |
| CelebA (Male) | `celeba_male_32/64/128` | 32/64/128 | 3 | 2 | train/test + fid_stats |
| Camelyon17 | `camelyon_32` | 32 | 3 | 2 | train/test/val + fid_stats |

Each dataset directory contains `train_<res>.zip`, `test_<res>.zip`, and `fid_stats_<res>.npz`.

## Installation

```bash
conda create -n dpimagebench python=3.9
conda activate dpimagebench
bash install.sh
```

## Data Preparation

```bash
# Preprocess datasets into zip format + compute FID stats
python data/preprocess_dataset.py \
    --data_name mnist fmnist cifar10 cifar100 \
    --data_dir ./dataset

# Preprocess ImageNet to target resolution
python data/process_imagenet.py \
    --data_dir /path/to/ImageNet_ILSVRC2012/train \
    --new_dir dataset/imagenet/imagenet_32 \
    --image_size 32
```

## Quick Start

```bash
# Minimal demo: DPIS-SQ then DPIS-MI on MNIST, ε=1.0, 1 epoch each
bash scripts/demo_dpis.sh
```

## Usage

```bash
# Standard invocation
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    run.py setup.run_type=torchrun \
    -m DPIS-SQ -dn mnist_28 -e 10.0 -ed my_run
```

### CLI Arguments

| Flag | Default | Description |
|---|---|---|
| `-m`, `--method` | `DP-LDM` | Method name |
| `-dn`, `--data_name` | `cifar10_32` | Dataset |
| `-e`, `--epsilon` | `10.0` | Privacy budget |
| `-ed`, `--exp_description` | `""` | Experiment tag appended to output dir |

### Config Overrides

All YAML keys can be overridden via CLI using OmegaConf dotlist syntax:

```bash
python run.py -m DPIS-SQ -dn mnist_28 -e 10.0 \
    pretrain.n_epochs=100 \
    train.n_epochs=50 \
    train.dp.max_grad_norm=0.01 \
    gen.data_num=10000
```

### Resume

```bash
python run.py -re <experiment_dir_name> -m DPIS-SQ -dn mnist_28 -e 10.0
```



### Phase Details

**1. Config Parse** — Loads `configs/<method>/<dataset>_eps<X>.yaml`, merges CLI overrides.

**2. Data Load** — Loads sensitive data from zip archives. For DPIS-SQ/DPIS-MI, also loads public data:

- **DPIS-SQ**: Runs semantic query — ResNet50 classifier maps sensitive images to semantically related ImageNet classes (50 out of 1000), filtered via `SpecificImagenet`. Labels are remapped from ImageNet IDs to contiguous 0-9.
- **DPIS-MI**: Creates differentially private central (mean/mode) images per class as public data.

**3. Pretrain** — Standard diffusion training on public data (DDP, Adam, EDM loss, EMA). No differential privacy.

**4. DP-SGD Train** — Fine-tunes on sensitive data with differential privacy:

| Component | Role |
|---|---|
| `DPDDP` | Distributed data parallel with per-sample gradient hooks |
| `PrivacyEngine` | Gradient clipping (max_grad_norm) + Gaussian noise + RDP accounting |
| `BatchMemoryManager` | Virtual large batch via gradient accumulation (n_splits) |
| `noise_multiplicity` | Multiple noise levels per image (32× for better score matching) |

**5. Generate** — EDM/DDIM sampling to produce synthetic images, saved as `gen.npz`.

**6. Evaluate** — Two metrics:

- **Accuracy**: Train ResNet/WRN/ResNeXt classifiers on synthetic images, test on real data. Uses DP-aware model selection (Laplace noise on validation accuracy).
- **FID**: Fréchet Inception Distance between real and synthetic InceptionV3 feature distributions.

## Project Structure

```
DPImageBench/
├── run.py                     # Training entry point
├── install.sh                 # Dependency installation
├── requirements.txt
├── configs/
│   ├── DPIS-SQ/               # 17 configs (all datasets × eps10/eps1)
│   ├── DPIS-MI/               # 17 configs
│   └── <baseline>/            # 5 baseline method config dirs
├── models/
│   ├── DPIS_SQ/               # Semantic query: ResNet50 classifier
│   │   ├── resnet.py
│   │   └── classifer_trainer.py
│   ├── DP_Diffusion/          # EDM diffusion model
│   │   ├── denoiser.py        # EDMDenoiser (preconditioning)
│   │   ├── score_losses.py    # EDMLoss, VPSDELoss, etc.
│   │   ├── samplers.py        # DDIM, EDM samplers
│   │   ├── generate_base.py
│   │   └── model/
│   │       ├── ncsnpp.py      # Song U-Net backbone
│   │       ├── layerspp.py
│   │       └── ema.py
│   ├── dpsgd_diffusion.py     # DP_Diffusion: pretrain + DP-SGD train + generate
│   ├── model_loader.py        # Method → model dispatch
│   ├── DP_MERF/               # Random Fourier Features utils
│   ├── DP_GAN/                # DP-GAN utils
│   ├── DP_LDM/                # Latent Diffusion peft utils
│   ├── pretrained_models/     # Checkpoints
│   └── synthesizer.py         # Base class
├── data/
│   ├── dataset_loader.py      # Runtime data pipeline
│   ├── preprocess_dataset.py  # Raw → zip + fid_stats
│   ├── process_imagenet.py    # ImageNet resize
│   ├── SpecificImagenet.py    # Semantic class filtering + label remapping
│   └── stylegan3/dataset.py   # Zip-based ImageFolderDataset
├── evaluation/
│   ├── evaluator.py           # Accuracy + FID evaluation
│   ├── ema.py                 # Exponential Moving Average
│   └── classifier/            # ResNet, WRN, ResNeXt
├── opacus/                    # Modified Opacus (DPDDP, PrivacyEngine)
├── utils/utils.py             # Config parsing, distributed launch
├── scripts/
│   └── demo_dpis.sh           # Quick demo for DPIS-SQ + DPIS-MI
├── dataset/                   # Preprocessed datasets
│   ├── mnist/                 # train_28.zip, test_28.zip, fid_stats_28.npz
│   ├── fmnist/
│   ├── cifar10/
│   ├── cifar100/
│   ├── eurosat/
│   ├── celeba/
│   ├── camelyon/
│   └── imagenet/              # imagenet_32/ directory + imagenet_32.zip
└── exp/                       # Experiment outputs

```

## Output

Each run creates:

```
exp/<method>/<dataset>_eps<eps><desc>-<timestamp>/
├── stdout.txt                 # Full log with metrics
├── pretrain/checkpoints/      # Pretrain snapshots
├── train/checkpoints/         # DP-SGD training snapshots
└── gen/
    ├── gen.npz                 # Synthetic images + labels
    └── sample.png              # Sample grid (8 per class)
```

## Key Configuration Parameters

### Training

| Parameter | Default (DPIS-SQ) | Description |
|---|---|---|
| `pretrain.n_epochs` | 3200 | Pretrain epochs on public data |
| `pretrain.batch_size` | 1024 | Pretrain batch size |
| `train.n_epochs` | 150 | DP-SGD epochs |
| `train.batch_size` | 4096 | Logical batch size (must be large for DP) |
| `train.max_physical_batch_size` | 8192 | Max physical batch before gradient step |
| `train.n_splits` | 32 | Virtual micro-batch count |
| `train.dp.max_grad_norm` | 0.001 | Per-sample gradient clipping norm |
| `train.dp.epsilon` | 10.0 | Target privacy budget |
| `gen.data_num` | 60000 | Synthetic images to generate |

### Model

| Parameter | MNIST | CIFAR-10 | Description |
|---|---|---|---|
| `model.network.image_size` | 28 | 32 | Input resolution |
| `model.network.num_in_channels` | 1 | 3 | Input channels |
| `model.network.ch_mult` | [2,2] | [2,2,2] | Channel multipliers per resolution |
| `model.network.attn_resolutions` | [14] | [16] | Resolutions with attention |
| `model.sampler.num_steps` | 50 | 50 | DDIM sampling steps |

### Semantic Query (DPIS-SQ)

| Parameter | Default | Description |
|---|---|---|
| `public_data.name` | `imagenet` | Public dataset |
| `public_data.n_classes` | 1000 | Public dataset classes |
| `public_data.selective.ratio` | 0.05 | Fraction of public classes to select |
| `public_data.selective.sigma` | 50 | Gaussian noise scale for DP |

