# 五子棋 AlphaZero（C++）

15×15 五子棋引擎，AlphaZero 式强化学习训练。核心为纯 C++17：神经网络前向/反向、
Adam 优化器、无锁 MCTS、自对弈流水线全部手写，无框架依赖；装有 PyTorch 时可选用
LibTorch/CUDA 后端加速推理与训练。

## 特性

- **纯 C++ 核心**：前向/反向/Adam 全部手写，只链接 `pthread`。
- **CPU 推理快**：AVX2 卷积（寄存器分块）+ int8 量化（AVX-VNNI 点积指令），运行时检测 CPU 指令集。
- **无锁 MCTS**：竞技场分配节点、缓存行对齐原子操作、虚拟损失、评估缓存（约 30% 的 playout 跳过网络）。
- **批量自对弈**：多局叶子任务合并成大 batch 推理（`GOMOK_BATCH_WAIT_US` 控制等待上界）。
- **可选 CUDA**：自动探测 PyTorch，批量推理 + autograd 训练 + TF32；`--cpu` 强制走 CPU。
- **数值已验证**：C++ 前向输出与 ONNX 导出一致到 ~1e-5。

## 目录

```
├── src/            # 引擎：game、network、gradient、selfplay、train、logger
├── tools/          # 模型导出工具
├── tests/          # ctest 单元测试 + 基准工具
└── build/          # CMake 产物（Git 忽略）
```

## 编译

```sh
# CUDA 版本（使用当前 Python 环境里的 LibTorch）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DGOMOK_ENABLE_CUDA=ON -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

# 无依赖 CPU 版本
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
    -DGOMOK_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu -j
```

默认在 Python PyTorch 中检测到 `libtorch_cuda.so` 时启用 CUDA 后端；没有可用
CUDA 设备时程序自动回退到 CPU。Release 构建默认启用 `GOMOK_ENABLE_IPO=ON`。

## 使用

```sh
# 79 批 × 每批 64 局 = 5056 局（默认使用 CUDA）
GOMOK_MCTS_THREADS=8 ./build/gomoku_train train \
    --init best_policy_cpp_bin --games 79 --playout 400 --batch-games 64 \
    --check-freq 10 --threads 8

# 同上，纯 CPU；INT8 推理需要 AVX-VNNI
GOMOK_MCTS_THREADS=8 ./build/gomoku_train train \
    --init best_policy_cpp_bin --games 79 --playout 400 --batch-games 64 \
    --check-freq 10 --threads 8 --int8 --cpu

# 自对弈吞吐基准
GOMOK_PROFILE=1 GOMOK_SEED=1 ./build/gomoku_train selfplay \
    --model best_policy_cpp_bin --playout 400 \
    --games 64 --batch 64 --threads 8

# 终端对战
./build/gomoku_train human --model best_policy_cpp_bin --playout 400
```

### 训练参数

`train --games` 表示**训练批次数**，不是总对局数。例如
`--games 79 --batch-games 64` 会生成 5056 局；如果写成 `--games 5000`，
程序会生成 32 万局。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--init DIR` | 随机初始化 | 初始16文件模型目录 |
| `--games N` | 1000 | 自对弈/训练批次数 |
| `--playout N` | 400 | 每手棋的 MCTS 模拟次数 |
| `--batch-games N` | 64 | 每批同时进行的对局数 |
| `--batch-size N` | 512 | 每次优化使用的样本数 |
| `--epochs N` | 5 | 每批自对弈后的优化次数 |
| `--buffer N` | 10000 | 经验回放容量 |
| `--check-freq N` | 50 | 每隔多少批保存检查点 |
| `--threads N` | 自动，最多8 | CPU 推理 worker 数量 |
| `--int8` | 关闭 | AVX-VNNI 推理，仅 CPU |
| `--cpu` | 关闭 | 即使有 CUDA 也强制使用 CPU |
| `--mix PREFIX` | 无 | 可选外部训练数据 |
| `--mix-ratio R` | 0.5 | 每批优化中外部数据比例 |
| `--tag NAME` | `cpp` | 输出目录后缀：`best_policy_<tag>_bin` |

### 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `GOMOK_MCTS_THREADS` | min(8, 可用核心数) | MCTS 树 worker 数量 |
| `GOMOK_BATCH_WAIT_US` | 100 | 攒推理 batch 的最大等待时间（µs） |
| `GOMOK_LEAF_BATCH` | CUDA：总 batch 约128；CPU：每局 min(playout,128) | 每局推理队列上限 |
| `GOMOK_CACHE_ENTRIES` | 32768 | 评估缓存容量 |
| `GOMOK_PROFILE` | 关 | 打印每批的 eval/调用/缓存统计 |
| `GOMOK_SEED` | 时钟 | 固定自对弈随机种子 |

### 模型格式

权重存放在目录下的 16 个 `.bin` 文件里（`conv1_weight.bin`、`act_fc1_bias.bin` 等），
每文件为 `[u32 数量][float32 数据...]`。C++ 引擎读写同一格式，
`tools/export_model.py` 可从 PyTorch state_dict 生成；
`tools/export_onnx.py` 导出单文件 ONNX（输入 `state` [B,4,15,15]；输出
`log_policy` [B,225]、`value` [B,1]）。

## 网络结构

```
state [B,4,15,15]                      # 己方、对方、最后一手、执子方
  conv3x3 4->32 relu
  conv3x3 32->64 relu
  conv3x3 64->128 relu
policy: conv1x1 128->4 relu, fc 900->225, log_softmax   -> [B,225]
value:  conv1x1 128->2 relu, fc 450->64 relu, fc 64->1, tanh -> [B,1]
```

## 测试

- `gomoku_smoke` — 优化卷积 vs 朴素参考、快/量化推理 vs 慢推理一致性、胜负/悔棋、自对弈全流程。
- `gomoku_game_test` — 棋盘规则：四方向五连、悔棋、状态编码。
- `gomoku_gradient_test` — 手写反向传播 vs 数值差分。

```sh
ctest --test-dir build --output-on-failure
```

## 性能

以下数据实测于 2026-08-15，均为墙钟时间，不是理论 FLOPS。

### 测试环境

| 项目 | 版本 |
|---|---|
| 系统 | Arch Linux，kernel 7.1.4，x86-64 |
| CPU | Intel Core i7-14650HX，16核/24线程 |
| GPU | NVIDIA GeForce RTX 5060 Laptop，8151 MiB |
| 驱动 | NVIDIA 610.43.03 |
| 编译器 | GCC 16.1.1、CMake 4.4.0 |
| PyTorch | 2.12.0+cu130，CUDA runtime 13.0 |
| 构建 | Release、IPO 开启、MCTS/CPU 线程数8 |

### 完整单批训练

CPU 与 GPU 使用同一个导出的5000局检查点，参数如下：

```text
GOMOK_SEED=1，GOMOK_PROFILE=1
games=1，batch-games=64，playout=400
batch-size=512，epochs=5，check-freq=10，不使用 mix 数据
```

以下命令使用独立 tag，不会覆盖输入检查点：

```sh
# CUDA
GOMOK_PROFILE=1 GOMOK_SEED=1 ./build/gomoku_train train \
    --init best_policy_cpp_bin --games 1 --playout 400 --batch-games 64 \
    --batch-size 512 --epochs 5 --check-freq 10 --threads 8 \
    --tag bench_cuda

# CPU
GOMOK_PROFILE=1 GOMOK_SEED=1 ./build/gomoku_train train \
    --init best_policy_cpp_bin --games 1 --playout 400 --batch-games 64 \
    --batch-size 512 --epochs 5 --check-freq 10 --threads 8 \
    --int8 --cpu --tag bench_cpu
```

墙钟时间包含进程启动、自对弈、5次参数优化和最终模型保存。

| 后端 | 墙钟时间 | 网络评估数 | 样本数 | 推理 batch |
|---|---:|---:|---:|---:|
| RTX 5060 CUDA FP32/TF32 | **8.34秒** | 884,039 | 3,001 | 9,053次（平均97.7，最大128） |
| CPU INT8自对弈 + FP32训练 | **55.17秒** | 919,553 | 3,087 | 1,237次（平均743.4） |

本轮 GPU 端到端约快6.6倍。连续运行2批时总墙钟为14.19秒，其中完成预热后的
第二批为5.8秒。根据这些结果，79批（5056局）预计 CUDA 需要约
**9～11分钟**，CPU 约 **70～80分钟**。

即使 playout 相同，平均局长不同也会显著改变总工作量。比较两个版本时应同时观察
`evals`、`samples` 和 `moves/game`，不能只比较总秒数。

### 内核基准

| 负载 | RTX 5060 CUDA | i7-14650HX CPU（8线程） |
|---|---:|---:|
| 训练 B=512 × 5步，已预热 | **0.069秒** | **1.84～1.91秒** |
| 纯前向 B=128 | **169.5k eval/s** | — |
| 纯前向 B=512 | **131.2k eval/s** | — |
| 纯前向 B=768 | **120.7k eval/s** | — |
| 纯前向 B=4096 | **119.2k eval/s** | — |

这个15×15小网络在 CUDA batch 约128时吞吐最高，因此自对弈默认把 CUDA
总推理 batch 限制在128附近；在这张显卡上继续增大 batch 反而会降低吞吐。

## 数据

- `mix_data*` — 可选混合训练样本（自对弈 + 棋谱），训练时用 `--mix` 加载。
- `data/` — renju-net 风格棋谱数据（`.npz`）。

## 许可证

MIT。
