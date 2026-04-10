# openpi

该代码仓库基于 [openpi](https://github.com/Physical-Intelligence/openpi) 面向灵御双臂机器人，包含开源模型和用于微调训练的程序

提供一个基础 VLA 模型权重，用于微调训练。 

| 基础模型        | 用途    | 描述                                                                                                 | 检查点路径                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | 微调 | [$\pi_0$ 基础模型](https://www.physicalintelligence.company/blog/pi0)                | `gs://openpi-assets/checkpoints/pi0_base`      |


## 系统要求

要运行本仓库中的模型，需要配备至少以下规格的 NVIDIA GPU。当前训练脚本尚不支持多节点训练。

| 模式               | 所需内存 | 示例 GPU        |
| ------------------ | --------------- | ------------------ |
| 推理          | > 8 GB          | RTX 4090           |
| 微调（完整） | > 70 GB         | A100 (80GB) / H100 |


## 环境安装

克隆本仓库时，请确保更新子模块。

```bash
git clone https://github.com/zhou-yh19/openpi.git
```

使用 [uv](https://docs.astral.sh/uv/) 来管理 Python 依赖。
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者如果没有 curl，可以使用 wget
wget -qO- https://astral.sh/uv/install.sh | sh
```

安装 uv 后，运行以下命令来设置环境。

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

注意：`GIT_LFS_SKIP_SMUDGE=1` 是为了拉取 LeRobot 作为依赖项所必需的。


## 模型微调

### 1. 下载 base_model

提前将 `pi0_base` 放入到 `~/.cache/openpi/openpi-assets/checkpoints/` 目录中。


### 2. 把 遥操作数据 转换为 LeRobot 数据集

使用代码库 [rosbag_to_lerobot](https://github.com/dexteleop/rosbag_to_lerobot) 将 rosbag 转换为 LeRobot 数据集。


### 3. 计算训练集归一化参数

要在自己的数据上微调基础模型，需要定义数据处理和训练的配置。在下方为 LIBERO 提供了带有详细注释的示例配置，可以根据自己的数据集进行修改。

在开始训练之前，需要计算训练数据的归一化统计信息。使用训练配置名称运行以下脚本。

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_teleavatar
```
生成的 norm_stats.json 会放在数据集文件夹路径下。


### 4. 配置训练参数

编辑 [`TrainConfig`](src/openpi/training/config.py) 来调整训练参数。

```python
TrainConfig(
    name="pi0_teleavatar",
    # Here is an example of loading a pi0 model for LoRA fine-tuning.
    model=pi0_config.Pi0Config(
        action_dim=32,  # Keep 32 to match pi0_base pretrained weights
        action_horizon=50 # VLA produces 30 action steps at a time
    ),
    checkpoint_base_dir="~/checkpoints",
    data=LeRobotTeleavatarDataConfig(
        repo_id="~/organize_desk",  # 需要更换为lerobotdataset所在的本地路径
        base_config=DataConfig(
            prompt_from_task=True,
            action_sequence_keys=("action",),
        ),
        use_delta_joint_actions=False,  # Use end-effector representation
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    batch_size=64,
    num_train_steps=20_000,
    wandb_enabled=True,
    overwrite=False, # Overwrite existing checkpoints when fine-tuning with the same configuration
    resume=True,
),
```

将检查点保存到 `checkpoints` 目录，还可以在 Weights & Biases 仪表板上监控训练进度。


### 5. 运行训练脚本

现在可以使用以下命令启动训练。

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_teleavatar --exp-name=my_experiment
```

--exp-name: 实验名称，用于区分在不同设置下的 $π_0$ 微调后的权重保存路径。若按照上面命令来微调模型，微调后的模型权重保存的位置为 `~/checkpoints/pi0_teleavatar/my_experiment`。

为了最大化使用 GPU 内存，在运行训练前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` —— 这允许 JAX 使用高达 90% 的 GPU 内存（相比默认的 75%）。


### 6. 下载归一化文件

将第3步生成的 `norm_stats.json` 文件，先从数据集复制到模型文件中的 asset/inference 下，再下载模型文件。这个归一化文件在模型推理时会用到。
