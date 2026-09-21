# 五子棋 AlphaZero（C++）

15×15 五子棋引擎，AlphaZero 式强化学习训练。核心为纯 C++17：神经网络前向/反向、
Adam 优化器、无锁 MCTS、自对弈流水线全部手写，无框架依赖；装有 PyTorch 时可选用
LibTorch/CUDA 后端加速推理与训练。

## 特性

- **纯 C++ 核心**：前向/反向/Adam 全部手写，只链接 `pthread`。
- **CPU 推理可用**：原生前向（批归一化折合进卷积 + 残差），无 CUDA 时仍可对弈，并与 LibTorch 前向做数值对照。
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

# 同上，纯 CPU（只做推理；训练需要 LibTorch 后端）

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
|---|---|---:|---|
| `--init DIR` | 随机初始化 | 初始模型目录（旧 3 层检查点不兼容） |
| `--games N` | 1000 | 自对弈/训练批次数 |
| `--playout N` | 400 | 每手棋的 MCTS 模拟次数 |
| `--batch-games N` | 64 | 每批同时进行的对局数 |
| `--batch-size N` | 512 | 每次优化使用的样本数 |
| `--epochs N` | 5 | 每批自对弈后的优化次数 |
| `--buffer N` | 10000 | 经验回放容量 |
| `--check-freq N` | 50 | 每隔多少批保存检查点 |
| `--threads N` | 自动，最多8 | 推理 worker 数量 |
| `--cpu` | 关闭 | 强制走原生 CPU 路径（**仅推理**，无法训练） |
| `--mix PREFIX` | 无 | 可选外部训练数据 |
| `--mix-ratio R` | 0.5 | 每批优化中外部数据比例 |
| `--label-smooth F` | 0.1 | 人类棋谱 one-hot 目标的标签平滑量 |
| `--tag NAME` | `cpp` | 输出目录后缀：`best_policy_<tag>_bin` |

**学习率**（替代原先基于伪 KL 的反馈控制）：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--lr F` | 2e-3 | 基础学习率 |
| `--lr-warmup N` | 2 | 线性预热批次数 |
| `--lr-final-fraction F` | 0.1 | 余弦衰减终点（× `--lr`） |
| `--weight-decay F` | 1e-4 | L2 正则（CPU/CUDA 统一） |
| `--grad-clip F` | 2.0 | 梯度范数裁剪上限 |

**自对弈与搜索**：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--c-puct F` | 5.0 | PUCT 探索常数 |
| `--temp-high F` / `--temp-low F` | 1.0 / 0.5 | 温度调度两端 |
| `--temp-switch N` | 30 | 第 N 手起切换到低温 |
| `--dirichlet-eps F` | 0.25 | 根节点 Dirichlet 噪声强度（0 关闭） |
| `--dirichlet-alpha F` | 0.3 | 根节点 Dirichlet 浓度 |
| `--max-candidates N` | 60 | 每节点最大扩展子节点数 |
| `--virtual-loss F` | 8.0 | 虚拟损失强度（实测最优，减少线程碰撞重试） |
| `--unseen-smooth F` | 0.05 | 留给"未访问合法着法"的概率质量 |
| `--augment 0\|1` | 1 | 棋盘 8 重对称数据增强 |
| `--inference-batch N` | 32 | CUDA 单次批量推理的目标样本数 |

**棋力评估**（唯一反映强弱的指标）：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--eval-games N` | 0（关闭） | 每轮评估对局数 |
| `--eval-freq N` | 50 | 每隔多少批评估一次 |
| `--eval-playout N` | 400 | 评估时的 MCTS 模拟次数（与部署一致） |
| `--baseline DIR` | 无 | 对手检查点目录 |

**评估已批量化**：`play_match` 同时推进最多 32 局，把各局叶子按走子方分组后
合并成批量推理。实测 400 playout 下 **3.05 s/局**（单局串行是 7.85 s/局），
比自对弈（6.26 s/局）还快，因此可以用与部署一致的搜索预算做验收评估。

> ⚠️ **无禁手规则下先手必胜**。实测同一模型自我对战，黑棋在 playout 32/64/200
> 下都是 **20:0** 全胜（先手胜率 1.00）。配合交替颜色，任何两个模型的对局都会
> 呈现 50% 的比分，**无法反映棋力差异**。
> 实测对照：训练过的模型 vs 随机初始化模型——
> playout 1/4/16/32 时训练方 20:0 全胜（浅搜索下策略网络主导）；
> playout 64/100/128 时变成 10:10（先手优势主导）。
> 日志现在会打印先手胜率，超过 0.95 时给出提醒。
> **要比较两个模型的强弱，需要用连珠禁手规则或受限开局**；在此之前，
> 有意义的比较只能停留在浅搜索（此时衡量的是策略先验质量）。

`--int8` 已随新架构移除（AVX-VNNI 量化路径与新主干不兼容），
传入会给出提示并被忽略。


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

目录有**两种形式**，由是否存在 `training.meta` 区分：

| 形式 | 内容 | 用途 |
|---|---|---|
| **推理导出** | 折合后的 conv weight/bias + 全连接层 | MCTS、`human`、部署、CPU 推理 |
| **训练 checkpoint** | 原始 conv weight/bias + BN gamma/beta/running stats + 全连接层 + batch 序号 + `training.meta` | 续训 |

折合 BN 对推理是精确等价的仿射改写，但会改变参数化（BN 的四个张量不再存在），
因此**折合形式无法忠实续训**。`tools/export_onnx.py --bin-dir` 读的是推理形式，
`gomoku_train` 保存检查点时同时写出两者，`load` 按 `training.meta` 自动识别。

权重按张量存放在目录下（`stem_weight.bin`、`res0_a_weight.bin`、
`act_fc1_bias.bin` 等），每文件为 `[u32 数量][float32 数据...]`。

**批归一化已折合进卷积**，这是磁盘上的唯一格式，CUDA 与 CPU 两条路径读同一份
文件。折合对推理是精确等价的仿射改写，且是幂等的：加载时把
`running_var` 设为 `1 - eps`，使折合系数恰好为 1，因此反复保存/加载不会
累积漂移（由 `gomoku_backend_parity` 的往返检查覆盖）。

`tools/export_onnx.py --bin-dir <目录>` 导出单文件 ONNX（输入
`state` [B,4,15,15]；输出 `log_policy` [B,225]、`value` [B,1]）。

## 网络结构

结构集中定义在 `src/net_config.h`，其余代码由它派生：

```
state [B,4,15,15]                      # 己方、对方、最后一手、执子方
  stem:  conv3x3 4->192, bn, relu
  trunk: 12 x [ conv3x3 192->192, bn, relu,
                conv3x3 192->192, bn, 残差相加, relu ]
policy:  conv1x1 192->2, bn, relu, fc 450->225, log_softmax  -> [B,225]
value:   conv1x1 192->1, bn, relu, fc 225->64, relu,
         fc 64->1, tanh                                      -> [B,1]
```

共 **8,100,525** 参数，主干占 98.6%，感受野覆盖整个棋盘。

改动理由（均为实测）：

| 项目 | 旧 | 新 |
|---|---|---|
| 参数量 | 325,964 | 8,100,525 |
| 主干占比 | 28.7% | 98.6% |
| 感受野 | 7x7（判定五连需 9x9，**看不见长程威胁**） | 25 层卷积，覆盖全盘 |
| 主干死通道 | conv3 实测 15.6%（20/128） | 残差 + BN |
| 策略头 | `fc 900->225`，202,500 参数、无空间权重共享 | `1x1->2 + fc 450->225`，101,475 参数 |

**训练只走 LibTorch/CUDA 后端**；原生 CPU 路径只做推理（MCTS 用），
用来在没有 CUDA 时仍可对弈，并作为与 LibTorch 的数值对照。

## 测试

- `gomoku_smoke` — 棋盘四方向五连与悔棋、网络输出归一化、自对弈目标分布合理性。
- `gomoku_game_test` — 棋盘规则：四方向五连、悔棋、状态编码。
- `gomoku_gradient_test` — 手写卷积/全连接反向传播 vs 数值差分。
- `gomoku_backend_parity` — 同一权重下原生 CPU 前向 vs LibTorch 前向必须选出同一步棋
  （实测 0/8 argmax 不一致，log 空间最大偏差 1.7e-03）。

### Sanitizer

`-DGOMOK_SANITIZER=` 可开启 ASan / UBSan / TSan。选中 `thread` 时 CMake 会自动
关闭 LibTorch 后端（CUDA 运行时与 TSan 不兼容）：

```sh
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DGOMOK_ENABLE_CUDA=OFF -DGOMOK_SANITIZER=thread
cmake --build build-tsan -j
ctest --test-dir build-tsan --output-on-failure
```

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

# 纯 CPU 只保留推理路径
GOMOK_PROFILE=1 GOMOK_SEED=1 ./build/gomoku_train selfplay \
    --model best_policy_cpp_bin --playout 400 --games 64 --batch 64 \
    --threads 8 --cpu
```

墙钟时间包含进程启动、自对弈、5次参数优化和最终模型保存。

> **注意：下表是旧 3 层网络（325,964 参数）的数据，仅作历史参考。**
> 当前 12 层残差主干（8,100,525 参数）慢得多，见下方实测。

| 后端 | 墙钟时间 | 网络评估数 | 样本数 | 推理 batch |
|---|---:|---:|---:|---:|
| ~~RTX 5060 CUDA FP32/TF32~~ | ~~8.34秒~~ | 884,039 | 3,001 | 9,053次（平均97.7，最大128） |
| ~~CPU INT8自对弈 + FP32训练~~ | ~~55.17秒~~ | 919,553 | 3,087 | 1,237次（平均743.4） |

### 当前架构实测（RTX 5060 Laptop）

```
GOMOK_PROFILE=1 GOMOK_SEED=1 ./build/gomoku_train selfplay \
    --model <ckpt> --playout 400 --games 32 --batch 32 --threads 8
```

| 指标 | 数值 |
|---|---:|
| 墙钟 | 233.6 秒（32 局） |
| 每局 | 7.30 秒（46.7 手/局） |
| 网络评估数 | 501,216 |
| 推理调用 | 4,468 次（平均 batch 112.2，最大 128） |
| 评估缓存命中 | 69,757（13.9%） |

按此推算，79 批 × 64 局（约 5,056 局）在 CUDA 上约需 **4～6 小时**。
比旧网络慢一个数量级，这是把参数量从 32.6 万提到 810 万的直接代价。

评估缓存命中率从旧架构的 10.3% 提升到 13.9%（虚拟损失修正减少了
无效重试，更多 playout 落在已评估节点上）。

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
