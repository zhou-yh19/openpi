# Teleavatar 末端执行器控制说明

本文档说明如何使用末端执行器位姿（而非关节角度）来训练和推理 OpenPI 模型。

## 📋 概览

**关键变化：**

- 🔧 **表示方式**：从 16 维关节角度 → 16 维末端执行器位姿
- 📊 **状态维度**：从 48 维 → 62 维（增加了末端执行器位姿信息）
- 🎯 **动作格式**：7D 左臂末端位姿 + 1D 左夹爪力 + 7D 右臂末端位姿 + 1D 右夹爪力

## 📁 新增文件

1. **`src/openpi/policies/teleavatar_policy_endeffector.py`**
   - 末端执行器表示的策略转换
   - 从 62 维状态中提取末端执行器信息

2. **`examples/teleavatar/ros2_interface_endeffector.py`**
   - ROS2 接口，订阅末端执行器位姿 topic
   - 发布末端执行器位姿命令

3. **`examples/teleavatar/env_endeffector.py`**
   - 环境包装器，使用末端执行器表示

4. **`examples/teleavatar/main_endeffector.py`**
   - 主程序，用于推理控制

## 📊 数据格式说明

### 状态向量（62 维）

```python
observation/state[0:16]   # 关节位置（16维）
observation/state[16:32]  # 关节速度（16维）
observation/state[32:48]  # 关节力矩（16维）
observation/state[48:55]  # 左臂末端位姿 (x,y,z,qx,qy,qz,qw)
observation/state[55:62]  # 右臂末端位姿 (x,y,z,qx,qy,qz,qw)
```

### 动作向量（16 维）

```python
action[0:7]    # 左臂末端位姿 (x,y,z,qx,qy,qz,qw)
action[7]      # 左夹爪力矩
action[8:15]   # 右臂末端位姿 (x,y,z,qx,qy,qz,qw)
action[15]     # 右夹爪力矩
```

## 🔧 配置修改

在 `src/openpi/training/config.py` 中，`pi0_teleavatar_low_mem_finetune` 配置已修改为：

```python
TrainConfig(
    name="pi0_teleavatar_low_mem_finetune",
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=32,
        action_horizon=10
    ),
    data=LeRobotTeleavatarEndEffectorDataConfig(  # 使用新的配置
        repo_id="left_dataset",
        base_config=DataConfig(
            prompt_from_task=False,
            action_sequence_keys=("action",)
        ),
        use_delta_ee_actions=False,  # 末端执行器表示
    ),
    # ... 其他配置
)
```

## 🚀 使用步骤

### 1. 准备数据集

确保您的数据集包含 62 维的状态信息，其中索引 48-61 包含末端执行器位姿。

数据集应使用 LeRobot 格式，并包含以下字段：

- `observation/state`: 62 维状态向量
- `observation/images/left_color`: 左相机图像
- `observation/images/right_color`: 右相机图像
- `observation/images/head_camera`: 头部相机图像
- `action`: 62 维动作向量（训练时会自动提取所需的 16 维）

### 2. 训练模型

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_teleavatar_low_mem_finetune
```

训练将使用末端执行器表示：

- ✅ 自动从 62 维状态中提取末端执行器信息
- ✅ 使用 LoRA 进行低内存微调
- ✅ action_horizon=10（每次推理生成 10 步动作）

### 3. 配置 ROS2 Topics

⚠️ **重要**：您需要配置机器人发布末端执行器位姿信息。

在 `ros2_interface_endeffector.py` 中，默认订阅的 topics 为：

```python
# 订阅（读取传感器数据）
/left_arm/ee_pose          # 左臂末端位姿
/right_arm/ee_pose         # 右臂末端位姿

# 发布（发送动作命令）
/left_arm/ee_target        # 左臂末端位姿目标
/right_arm/ee_target       # 右臂末端位姿目标
```

**请根据您的实际 ROS2 系统修改这些 topic 名称！**

### 4. 启动策略服务器

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar_low_mem_finetune \
    --policy.dir=pi0_teleavatar_low_mem_finetune/pi0_lora_with_joint_positions_and_gripper_efforts/29999
```

### 5. 运行机器人控制

```bash
python examples/teleavatar/main_endeffector.py \
    --remote-host 192.168.1.100 \
    --prompt "pick a toy and put it in the basket using left gripper"
```

## 🔄 推理频率

根据 `main_endeffector.py` 的默认配置：

- `action_horizon=10`：策略服务器每次推理生成 10 步动作
- `open_loop_horizon=8`：执行 8 步后重新推理
- `control_frequency=20.0`：控制频率 20 Hz

**推理频率计算：**

- 每 8 步重新推理一次
- 20 Hz 控制频率 → 每步 0.05 秒
- **推理间隔 = 8 × 0.05 = 0.4 秒（2.5 Hz）**

可以通过修改 `open_loop_horizon` 参数来调整：

```bash
python examples/teleavatar/main_endeffector.py \
    --open-loop-horizon 5  # 改为每 5 步推理一次（4 Hz）
```

## ⚠️ 注意事项

1. **ROS2 Topics 配置**
   - 请确保在 `ros2_interface_endeffector.py` 中配置正确的 topic 名称
   - 使用 `ros2 topic list` 查看可用的 topics
   - 使用 `ros2 topic echo /left_arm/ee_pose` 测试数据

2. **坐标系**
   - 确保训练数据和推理时使用相同的坐标系
   - PoseStamped 消息中的 `frame_id` 需要正确设置

3. **四元数归一化**
   - 末端执行器姿态使用四元数表示 (qx, qy, qz, qw)
   - 确保四元数是归一化的（模长为 1）

4. **数据同步**
   - 确保末端执行器位姿和图像数据是同步的
   - ROS2 接口会等待所有传感器数据就绪

## 🐛 故障排查

### 问题：无法接收末端执行器位姿数据

```bash
# 检查 topic 是否存在
ros2 topic list | grep ee_pose

# 检查 topic 数据
ros2 topic echo /left_arm/ee_pose --once

# 检查 topic 频率
ros2 topic hz /left_arm/ee_pose
```

### 问题：四元数不合法

确保四元数归一化：

```python
import numpy as np
quat = np.array([qx, qy, qz, qw])
quat = quat / np.linalg.norm(quat)
```

### 问题：推理速度太慢

- 检查网络延迟：`ping 192.168.1.100`
- 减少 batch_size 加速推理
- 考虑使用本地推理而非远程服务器

## 📝 与关节角度版本的区别

| 特性 | 关节角度版本 | 末端执行器版本 |
|------|------------|--------------|
| 状态维度 | 48 | 62 |
| 动作表示 | 14 关节角 + 2 夹爪力 | 14 末端位姿 + 2 夹爪力 |
| 策略文件 | `teleavatar_policy.py` | `teleavatar_policy_endeffector.py` |
| 环境文件 | `env.py` | `env_endeffector.py` |
| ROS2 接口 | `ros2_interface.py` | `ros2_interface_endeffector.py` |
| 主程序 | `main.py` | `main_endeffector.py` |

## 📚 相关文档

- [OpenPI 训练配置说明](../../docs/training_config.md)
- [LeRobot 数据格式](https://github.com/huggingface/lerobot)
- [ROS2 Geometry消息](https://docs.ros.org/en/rolling/p/geometry_msgs/interfaces/msg/PoseStamped.html)
