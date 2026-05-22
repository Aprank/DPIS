## 环境配置

```bash
bash install.sh
```

## 数据集

敏感数据集预处理为 zip 归档，存放于 `dataset/` 下：

```bash
sh data_preparation.sh
```

## 基本运行

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    run.py setup.run_type=torchrun \
    -m DPIS-SQ -dn mnist_28 -e 10.0 -ed my_run
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-m`, `--method` | — | 方法：`DPIS-SQ`、`DPIS-MI`、`DP-MERF`、`DP-Kernel`、`DPGAN`、`DPDM`、`PDP-Diffusion` |
| `-dn`, `--data_name` | — | 数据集名称（见上表） |
| `-e`, `--epsilon` | `10.0` | 隐私预算 |
| `-ed`, `--exp_description` | `""` | 实验标签，附加在输出目录尾部 |

### 配置覆盖

所有 YAML 配置项均可通过命令行覆盖：

```bash
python run.py -m DPIS-SQ -dn mnist_28 -e 10.0 \
    pretrain.n_epochs=100 train.n_epochs=50 \
    train.dp.max_grad_norm=0.01 gen.data_num=10000
```


### 快速测试

```bash
bash scripts/demo_dpis.sh
```

## 输出

```
exp/<方法>/<数据集>_eps<隐私预算><描述>-<时间戳>/
├── stdout.txt              # 训练日志，含评估指标
├── pretrain/checkpoints/
├── train/checkpoints/
└── gen/
    ├── gen.npz              # 合成图像与标签
    └── sample.png           # 各类别样本
```

