# GomokuZero — C++17 / CUDA 实现的 15×15 AlphaZero 五子棋

[![CI](https://github.com/LangLang03/GomokuZero/actions/workflows/ci.yml/badge.svg)](https://github.com/LangLang03/GomokuZero/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **本项目是 vibe coding 的产物。** 引擎与本文档均在人工指导下由 AI 编码代理生成，并对照下文列出的命令、日志与产物逐条复核。README 里每个测量值都绑定到可复现的东西，就是为了让人能去核对，而不是照着信 —— 代码也请同样对待。

从零实现的 **15×15 自由规则五子棋** AlphaZero 风格引擎（五连或更长即胜，无连珠禁手、无交换开局）。棋盘、PUCT/MCTS、批量自对弈、数据管线、训练器、检查点、竞技场与 CLI 全部是 C++17；神经网络训练走 LibTorch/CUDA，推理走原生「BN 折合」CPU 路径。Python 工具链负责人类棋谱数据集处理、ONNX 导出、量化以及浏览器/终端对弈服务。

起点是著名的 Python 项目 [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)。在下文这台笔记本上，一次同等规模的 15×15 训练，使用当前 **16 块 × 256 通道**网络 —— 250 个自对弈批次、每批 64 局、每手 400 次搜索 —— 在那里花了**约 7 天**，在这里花了**约 14 小时** —— 哪些是实测、哪些是机制说明见[结果速览](#结果速览)和[为什么快：优化清单](#为什么快优化清单)。

| | |
|---|---|
| 规则 | 15×15 自由规则五子棋，五连及以上即胜，黑先 |
| 网络 | 残差主干，**16 块 × 256 通道**，19,025,709 参数，主干感受野 67×67 |
| 输入 / 输出 | `[B,4,15,15]` 平面 → `log_policy [B,225]` + `value [B,1]`（tanh） |
| 搜索 | 批量 PUCT/MCTS，默认每手 400 次搜索，虚拟损失、子树复用、战术精确值 |
| 训练 | 人类棋谱监督预训练 → 搜索蒸馏 → 自对弈强化学习，Adam + 余弦调度，可选 BF16，按价值加权的检查点选择 |
| 技术栈 | C++17、CMake、LibTorch/CUDA 训练、原生 CPU 推理、ONNX Runtime 部署 |
| 数据 | RenjuNet XML → 4,218 局 / 129,588 局面，按整局划分并剔除 D4 泄漏，另加 3,000 局搜索蒸馏自对弈 / 51,755 局面 |

参考网络用 `-DGOMOK_NET_CHANNELS=256 -DGOMOK_NET_RES_BLOCKS=16` 构建；编译期默认仍是旧的 12 块 × 192 通道形状，那是另一种检查点格式，不能与它混用。

---

## 目录

- [结果速览](#结果速览)
- [基准设备](#基准设备)
- [快速开始](#快速开始)
- [为什么快：优化清单](#为什么快优化清单)
- [网络与搜索](#网络与搜索)
- [训练流程](#训练流程)
- [数据集](#数据集)
- [强度测评](#强度测评)
- [部署：ONNX、量化、浏览器与终端对弈](#部署onnx量化浏览器与终端对弈)
- [仓库结构](#仓库结构)
- [测试与 CI](#测试与-ci)
- [环境变量](#环境变量)
- [代码导览](#代码导览)
- [口径、来源与限制](#口径来源与限制)

---

## 结果速览

基准训练：[`tools/train_256x16_quality.sh`](tools/train_256x16_quality.sh) —— 当前 256×16 流水线，分三个可续训阶段：

1. **搜索蒸馏自对弈数据** —— 从最强的 192×12 检查点做 3,000 局、每手 400 次搜索 → 51,755 个带真实自由规则胜负的局面（价值监督开启），每局 17.3 个局面。
2. **监督预训练** —— 在 116,781 个人类局面上做 6 轮策略模仿（批 256、BF16、按验证最优），再在蒸馏集上做 2 轮（绝对目标 8）。
3. **自对弈强化学习** —— **250 批 × 64 局 = 16,000 局**，每手 400 次搜索，每批 16 个全新优化器 minibatch（4,000 次优化更新），回放缓冲 50,000，每批 35 % 人类棋谱样本，在留出的自对弈探针上做按价值加权的检查点选择，每 50 批跑一次 80 局竞技场。

| | 参考实现 junxiaosong/AlphaZero_Gomoku（PyTorch） | 本仓库（C++17 + LibTorch/CUDA） |
|---|---:|---:|
| 同机、同等 15×15 训练量 | **约 7 天** | **约 14 小时** |
| 加速比 | — | **约 12×** |

这约 14 小时的构成（本机，`build-256x16/`）：

| 阶段 | 实测 | 依据 |
|---|---:|---|
| 自对弈 RL，第 50 批 → 第 250 批，从第 50 批检查点续训（200 批、4 次 80 局竞技场），每批 64 局、每手 400 次搜索 | 11.1 h 墙钟 | 流水线锁 `2026-09-29 01:29` → 最终检查点 `12:35` |
| 用这段速率外推到全部 250 批 | 约 13.9 h | 11.1 h × 250/200。同一条日志的稳态速率是每批 211 s，其 `[progress]` 行给出的全程投影是 16.3 h（`elapsed` + **剩余** `ETA`：第 49 批 `2:53:16 + 13:23:38`，第 56 批 `3:35:53 + 12:40:07`） |
| 监督预训练，6 轮人类 + 2 轮蒸馏，批 256，BF16 | 约 17 分钟 | 3,148 次更新（457 × 6 + 203 × 2），按该次运行实测的 807 样本/秒，另加 8 次验证 |
| 蒸馏数据，3,000 局、每手 400 次搜索（192×12 生成器） | 约 2 h | 生成脚本自己给出的 2.2 s/局估计 |

对比行用的是 RL 阶段，因为两种实现的差别就在这里。一次干净的运行落在这两个估计之间 —— 按续训段的速率是 13.9 h，按第一次（被中断的）运行自己的投影是 16.3 h —— 因此标题写「约 14 小时」。约 2 h 的蒸馏批次是一次性数据生成步骤，每个网络家族只跑一次。

那次 250 批运行实际达到的水平（记录在 `runs/q256_rl/metrics.csv` 和 `runs/train_256x16.log`）：

| 测量项 | 结果 |
|---|---|
| RL 验证，第 0 批 → 第 250 批，2,048 个抽样局面 | 策略 CE 2.047 → **1.807**，top-1 46.3 % → **48.9 %**，价值 MSE 1.091 → **0.751** |
| 竞技场，第 50 批 对 蒸馏后的监督检查点，80 局，400 次搜索，4 手配对开局 | `q256_rl`（第 50 批）41 胜 / 37 负 / 2 和，得分 0.5250，开局对 6 胜 / 4 负 / 30 和，先手胜率 0.795，1,058 s |

同一次训练、同一个 256×16 网络的支撑实测：

| 测量项 | 结果 |
|---|---|
| ONNX Runtime 对 原生 C++ 引擎，30 手复盘、每手 200 次搜索 | 343.8 ms/move 对 348.2 ms/move，30 个局面全部选择相同着法，各自 4,798 次网络调用 |
| 前向，batch 1（CUDA） | C++ 2.138 ms 对 ONNX Runtime 1.752 ms |
| 前向，batch 32（CUDA） | C++ 22.896 ms（1,398 pos/s）对 ONNX Runtime 23.358 ms（1,370 pos/s） |
| 72.5 MiB 导出的 FP16 / INT8 ONNX 变体 | 38.0 MB / 19.2 MB，与 FP32 的 top-1 一致率 100.00 % / 93.23 %，价值 MAE 0.0003 / 0.0674 |
| C++ 测试套件，12×192 生产构建 | `ctest` **11/11，71.0 s** |
| 数据集载入 + 校验 + 按整局划分 | 129,588 局面 0.394 s，峰值 RSS 570 MiB |

在较旧的 12×192 网络上做的测量保留在下面并标注为旧形状；它们不会被悄悄套用到新形状上。这里的每个数字都绑着一条命令、一行日志或一个产物；没有任何一个是 Elo，也没有任何一个是关于所有对手的结论。

---

## 基准设备

本 README 的所有耗时都产自这一台笔记本，基线精确如下。请把它们当作「单机证据」，而不是与硬件无关的承诺。

| 组件 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8,151 MiB，计算能力 12.0（sm_120），驱动 610.57.04，CUDA UMD 13.3 |
| CPU / 内存 | Intel Core i7-14650HX（24 逻辑核），15 GiB RAM |
| 系统 / 内核 | Arch Linux，内核 7.1.9-arch1-2，glibc 2.44 |
| 编译器 / 构建 | GCC 16.2.1，CMake 4.4.2，`-DCMAKE_BUILD_TYPE=Release`，C++17 |
| C++ 训练后端 | LibTorch **2.12.0+cu130**（CUDA 13.0，cuDNN 9.24），由 CMake 通过 `/usr/bin/python3.14` 发现 |
| Python 工具 | `.venv/` 内 Python 3.14：torch 2.14.0+cu130、numpy 2.5.3、onnx 1.23.0、onnxruntime-gpu 1.28.0 |
| 部署 | onnxruntime-gpu 1.28.0（CUDA 13 / cuDNN 9），CUDA 库来自 pip 的 `nvidia-*` wheel |

**不需要 nvcc**：CUDA 后端就是 LibTorch，CMake 只需要一个能 `import torch` 的 Python 解释器和 `libtorch_cuda.so`。机器上装了多个环境时用 `-DPython3_EXECUTABLE=...` 指定（本机同时存在系统 torch 2.12 与 venv torch 2.14）。

256×16 网络把 12×192 的参数量提高了 2.35×（19.0 M 对 8.1 M），每次前向的算术量大约翻倍，都跑在同一块 8 GiB 笔记本 GPU 上；上面的训练数字就是它在这里的代价。

---

## 快速开始

### 1. 构建与测试

```sh
# 依赖：CMake >= 3.18、C++17 编译器、带 CUDA 的 torch
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'   # 例如 2.12.0+cu130 13.0

# 参考的 256x16 形状（tools/train_256x16_quality.sh 就按它构建）：
cmake -S . -B build-256x16 -DCMAKE_BUILD_TYPE=Release \
  -DGOMOK_ENABLE_CUDA=ON -DBUILD_TESTING=ON \
  -DGOMOK_NET_CHANNELS=256 -DGOMOK_NET_RES_BLOCKS=16 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build-256x16 -j2
ctest --test-dir build-256x16 --output-on-failure
```

也支持无 CUDA 构建，用于推理、搜索和测试；**没有 LibTorch/CUDA 后端时训练会被拒绝**：

```sh
cmake -S . -B build-cpu -DGOMOK_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu -j2 && ctest --test-dir build-cpu --output-on-failure
```

两个 `-DGOMOK_NET_*` 都不给就是旧的 12×192 形状。形状是编译期常量，所以检查点、导出和基准都只属于某一个形状。

### 2. 对弈、观察自对弈、跑对战

```sh
./build-256x16/gomoku_train human --model best_policy_q256_rl_val_bin --playout 400   # 输入 r,c（0-14）

GOMOK_PROFILE=1 ./build-256x16/gomoku_train selfplay \
  --model best_policy_q256_rl_val_bin --games 8 --batch 8 --playout 400 --seed 1

./build-256x16/gomoku_train arena --model best_policy_q256_rl_val_bin \
  --baseline best_policy_q256_distill_bin --games 80 --playout 400 \
  --opening-moves 4 --seed 1
```

检查点（`best_policy_*_bin/`）是本地产物，不入版本控制：先自己训一个，或用 `./build-256x16/gomoku_train export --output my_model_bin` 导出随机权重。

### 3. 准备数据并训练

```sh
python3 tools/parse_renjunet.py data/renjunet.xml --bin-prefix data/gomoku_v2
python3 tools/audit_dataset.py data/gomoku_v2 --state-overlap   # 审计 JSON 输出到 stdout

# 监督预训练（绝对目标 6 轮；--games 0 表示只做监督）
./build-256x16/gomoku_train train --mix data/gomoku_v2 --pretrain 6 --games 0 \
  --batch-size 256 --epochs 1 --precision bf16 --seed 1 \
  --val-samples 2048 --val-freq 1 --kl-freq 0 --early-stop-patience 2 --tag q256_pre

# 从验证最优检查点开始自对弈 RL（绝对目标：第 250 批）
./build-256x16/gomoku_train train --init best_policy_q256_pre_val_bin --mix data/gomoku_v2 \
  --pretrain 6 --games 250 --batch-games 64 --batch-size 256 --epochs 16 --precision bf16 \
  --playout 400 --buffer 50000 --mix-ratio 0.35 --seed 1 \
  --val-samples 2048 --val-freq 5 --kl-freq 5 --selfplay-val-every 10 \
  --select-value-weight 1 --check-freq 25 \
  --eval-games 80 --eval-freq 50 --eval-playout 400 --tag q256_rl
```

带蒸馏阶段、可续训阶段、GPU 锁和竞技场的参考训练是 [`tools/train_256x16_quality.sh`](tools/train_256x16_quality.sh)；更早的 12×192 5,000 局训练仍是 [`tools/train_quality_5000.sh`](tools/train_quality_5000.sh)。`--games` 与 `--pretrain` 都是**绝对目标**而非增量：从第 100 批续训并写 `--games 250`，只会再生成 150 批。

### 4. 导出、量化与部署

```sh
./build-256x16/gomoku_train export --model best_policy_q256_rl_val_bin --output model_export_256x16
python3 tools/export_onnx.py --bin-dir model_export_256x16 256x16.onnx     # 76.0 MB
python3 tools/quantize_onnx.py --src 256x16.onnx --out-dir runs/quantized  # fp16 + int8 + drift report

python3 -m pip install 'onnxruntime-gpu[cuda,cudnn]'
python3 tools/server_onnx.py --model 256x16.onnx --playout 400 --device cuda \
  --host 127.0.0.1 --port 8000        # 浏览器打开 http://127.0.0.1:8000
```

---

## 为什么快：优化清单

参考实现是单进程 PyTorch 训练器：一次只跑一局、每次前向只评估一个叶子、FP32、每手都重建搜索树。下面每一条都在替换其中一种做法。「基线」始终指那种参考做法，每一行都附证据。

本节的内核与微基准行是在 **12×192** 网络上测的（仓库默认构建的形状）；端到端训练行是 256×16。优化本身与形状无关，常数不是。

### A. 自对弈并行与搜索调度

| # | 优化 | 基线做法 | 本实现 | 证据 / 收益 |
|---|---|---|---|---|
| A1 | **批量自对弈**：一次调用推进 `--batch-games` 局（默认 64） | 一轮一局 | 64 棵树同时在飞，互相独立 | `src/selfplay.cpp` `BatchedSelfPlay::run_batch`；256×16 的 RL 训练每批生成 64 局 |
| A2 | **无锁 MCTS worker 池**（条件变量按轮启动） | 单线程搜索 | `min(8, 物理核)` 个 worker，钳制 1–32 | `src/selfplay.cpp` `SearchWorkerPool`；`GOMOK_MCTS_THREADS`、`--threads` |
| A3 | **CPU 亲和性：把最快的核留给 GPU 协调线程** | 不绑核，搜索线程与推理线程互相抢核 | 物理核按最大频率排序，跳过最快的 `inference_threads` 个核 | `src/selfplay.cpp`（拓扑 + 读取 `cpufreq`） |
| A4 | **批量叶子评估 + 收集窗口**：CUDA 下自旋最多 `GOMOK_BATCH_WAIT_US`（100 µs）后提交一次 GPU 调用 | 一个叶子一次前向，GPU 在两次调用之间空转 | 目标推理批 32；所有对局的叶子合并成一批 | 在 12×192 上批大小扫描 16…512 → **32 最优**（2,846 eval/s，128 只有 2,588），而且把目标从 128 降到 32 还让碰撞重试减少 32%；在 256×16 上吞吐仍能涨到 batch 32（1,398 pos/s，batch 1 只有 468），所以目标仍是 32 |
| A5 | **定点虚拟损失**，作用在在飞路径上 | 无虚拟损失，多线程挤向同一子节点 | visits 用整数原子、价值和用 Q20 定点，默认 8；叶批为 1 时关闭 | 在 12×192 上扫描：虚拟损失 3 → 8 让碰撞重试减少 **27%**（每次评估 720 → 523），**每局快 6%**（1.66 → 1.56 s） |
| A6 | **碰撞重试不消耗搜索预算** | 重试算一次搜索 | 命中 `NODE_EXPANDING` 返回 `Retry`，不递减已发放的搜索次数 | `src/selfplay.cpp` 重试循环，预算用 CAS 钳制 |
| A7 | **跨手子树复用**（`GOMOK_TREE_REUSE=1`，默认开） | 每手 `reset_tree()`，400 次搜索从零开始 | 提根：所选子节点及其后代搬进备用竞技场，已完成的访问计入下一手预算 | 小模型 CPU 对照：8 局 **9.40 s → 6.44 s**；固定棋路对照：搜索循环 0.040 s → 0.012 s。256×16 训练中已启用，但未在该形状上隔离实测 |
| A8 | **跨手局面评估缓存**：Zobrist 键 + FIFO 环，32,768 条，一个批次内 64 局共享 | 每个叶子都重新评估 | 12×192 生产自对弈命中率 0.27–0.41；同一批内的重复叶子也会去重 | `GOMOK_CACHE_ENTRIES`；profile 输出 `cache_hits`/`cache_hit_rate` |
| A9 | **战术与终局短路** | 每个叶子都问网络 | 一步必胜 → 精确 +1 且不推理；唯一必防点把展开收敛为单子节点；终局与已解节点永不进网络 | `src/selfplay.cpp`；256×16 复盘里每手网络调用随胜负明朗逐手塌缩：200、169、147、146、99、98、93、121、120，最后三手 **4、1、0** |
| A10 | **预分配节点竞技场与 scratch**（节点 32 字节对齐，搜索期零分配） | 每个节点一个 Python 对象 | 每局每批一个竞技场，worker scratch 跨手复用 | `alignas(32) GNode`、`SearchWorkerScratch`，缓冲按 `run_batch()` 一次性分配 |
| A11 | **竞技场批量对战** | 一次一局 | 最多 32 局同步推进，每轮恰好两次批量调用（每个模型一次） | `play_match`；上文 80 局 256×16 竞技场在 400 次搜索下用 1,058 s 跑完 |

### B. 精度与 GPU 内核

| # | 优化 | 基线做法 | 本实现 | 证据 / 收益 |
|---|---|---|---|---|
| B1 | **TF32 同时作用于 cuDNN 与 cuBLAS**，RAII 作用域、默认开 | 纯 FP32 内核 | `GOMOK_TF32` / `GOMOK_INFERENCE_TF32`（默认都是 1） | 12×192：严格 FP32 → TF32 推理 172.8 → 41.9 ms（**4.1×**），训练 354.6 → 152.0 ms（**2.3×**） |
| B2 | **BF16 训练 autocast**，参数/损失/Adam 动量保持 FP32，头部最后两个非线性强制 FP32 | 只有 FP32 | `--precision bf16` | 12×192：训练 152.0 → 76.6 ms（相对 TF32 **1.98×**）；端到端监督阶段 460.4 s → 260.5 s（**1.77×**）。256×16 的两个监督阶段和 RL 阶段都跑 BF16 |
| B3 | **融合优化器与融合梯度运算** | 逐参数 Python 循环 | `at::_fused_adam_`、`_foreach_norm` 求全局范数、`_foreach_mul_` 裁剪、5 个标量合并为一次回读 | `GOMOK_FUSED_ADAM=1`；单独 A/B 只有约 0.5–1 %（留着是为了减少 kernel 启动） |
| B4 | **BN 折合进卷积**，每次权重换代只折一次 | 每次前向都跑 BN kernel | 推理图里只剩 conv + bias + ReLU | `src/network_cuda.cpp` `fold_bn`；256×16 导出是 19,025,709 个训练参数中的 19,008,807 个折合参数 |
| B5 | **`torch::InferenceMode` + `eval()` + channels-last** 的卷积权重/激活 | 带 autograd 图的 FP32 NCHW | 无图、无版本计数、对 Tensor Core 友好的内存布局 | `src/network_cuda.cpp` |
| B6 | **故意关闭 cuDNN autotune** | `cudnn.benchmark = True` | MCTS 批大小不断变化，autotune 是净亏（需要确定性时用 `GOMOK_DETERMINISTIC=1`） | `src/network_cuda.cpp` 注释 |
| B7 | **冗余卷积 bias 不参与优化** | 所有参数都更新 | 冻结卷积 bias（BN 使其冗余，且其舍入噪声会被 Adam 放大），同时缩小 Adam 状态 | 卷积 bias 上 `requires_grad(false)` |
| B8 | **无监督时价值分支根本不进计算图** | `policy + 0 * value` 仍会生成价值梯度，被 Adam 的权重衰减毁掉 | 关闭价值监督时价值项直接不存在，价值参数使用独立的 Adam 步数 | 修复了一次实测到的价值头塌缩：旧写法下价值头稠密层权重衰减到约 1e-37 |
| B9 | **原生 CPU 推理路径**（折合 BN、8 路展开卷积、线程本地 scratch、按样本并行） | — | 没有 CUDA 也能跑 MCTS、CLI 和测试；CPU 训练被明确拒绝而非降级 | 本机 256×16 实测：**每前向 3,374.8 ms**（单线程），而 FP16 ONNX 导出在 CPU 上是 12 ms —— 可用于测试和小的默认形状，没有 GPU 时不能用来跑 256×16 模型 |

### C. 数据、IO 与检查点

| # | 优化 | 基线做法 | 本实现 | 证据 / 收益 |
|---|---|---|---|---|
| C1 | **数据集 mmap + 只存索引的划分** | 全量读进内存当张量 | `mmap(PROT_READ, MAP_PRIVATE)` + 逐条校验；训练/验证只是行索引，不复制数据 | 129,588 局面校验并划分 **0.394 s**，峰值 RSS 570 MiB |
| C2 | **在线 D4 增强**（8 种对称，状态与策略同一变换） | 预展开或无增强 | 每次优化更新重新抽样**并**重新选对称 | 8× 样本效率、1× 存储；`--augment 1` |
| C3 | **每次更新都换新 minibatch** | 同一 minibatch 反复更新 | 每步重新抽回放样本与外部样本，日志记录 `replay_draws`/`external_draws` | 256×16 的 RL 训练每批 16 次更新，每次抽 2,672 个回放 + 1,424 个外部样本 |
| C4 | **流式、带校验的数据工具** | 纯内存 numpy 管线 | `npz_to_bin.py` 逐成员 mmap；`parse_renjunet.py` 逐局校验 + D4 去重；临时文件 + rename 写出 | 159,000 条 XML 记录 → 4,218 局接受，并留有淘汰台账 |
| C5 | **按整局划分 + 剔除 D4 泄漏** | 按局面随机划分 | 用 `splitmix64(id ^ seed)` 对局 ID 做哈希，再剔除与训练集 D4 等价的留出局面；C++ 与 Python 同一算法并有跨语言黄金测试 | 随机划分会泄漏约 88.9 % 的对局；整局划分后剩 619/12,807 = 4.83 % D4 重叠，剔除后 → **12,188 个独立验证局面**。3,000 局蒸馏集用同样方式审计（按行划分预期泄漏 80.4 %） |
| C6 | **检查点原子换代 + 恢复合约** | 只存/读权重 | `.staging` → rename，带 `.previous` 回滚；网络 + BN + Adam + 回放 + 批号 + 调度视野 + 种子 + 选项/数据指纹，全部校验后才使用 | 已完成的训练不会被悄悄从头重跑；`--games`/`--pretrain` 在续训后仍是绝对目标。256×16 的 RL 训练被中断过一次，从第 50 批检查点续上 |
| C7 | **诚实的 ETA 与分离计时** | 每轮墙钟时间 | 自对弈与更新分开计时，按近期批次平滑，计划中的竞技场单独计入 | `[train]`、`[progress]` 日志；上文约 14 h 就是从它们读出来的 |
| C8 | **按价值加权的检查点选择** | 最后一批获胜，或只看策略损失 | `--select-value-weight W` 在留出回放缓冲的自对弈探针上（`--selfplay-val-every N`）按 `policy + W × value` 给快照打分，而不是在人类数据上打分 —— 后者的胜负不是本引擎的价值标签 | `src/train.cpp`；没有它，256×16 流水线曾保留过一个从未在价值头上被评估过的快照 |

### D. 工程

| # | 优化 | 理由 |
|---|---|---|
| D1 | 网络宽度、块数、头大小的唯一来源（`src/net_config.h`），C++ 与 Python 共用 | 过去形状散落在四处，网络、导出器和测试可能互相不一致。本 README 中每个随形状变化的数字（折合 19,008,807 / 训练 19,025,709、67×67）都由它推导 |
| D2 | `gomoku_core` 只编译一次的静态库，所有可执行文件与测试链接它 | 避免每个目标都重编昂贵的 LibTorch 翻译单元 |
| D3 | 默认开 LTO，但 LibTorch 那个 TU 单独用 `-O1 -fno-lto` | 既保留发布优化，又约束编译时间与内存 |
| D4 | CI 用缩小主干跑 ASan + UBSan，另有一个生产形状的编译任务 | 并发与内存缺陷能被抓到，又不必让每个任务都付完整 16×256 网络的代价 |
| D5 | 严格 CLI 解析（未知/重复/缺值直接失败），模型加载失败绝不静默换成随机权重 | 杜绝「训错了实验却毫无察觉」这类失败 |
| D6 | 一个 Python 搜索的 C++ 镜像（`tests/play_benchmark.cpp` 对 `tools/gomoku.py`） | 让「C++ 引擎 vs ONNX Runtime」成为对同一个后端的受控测量，而不是两种实现的比较；两个驱动器都被检查过会选择相同着法 |

### 没赚到的尝试（保留但默认关闭）

| 尝试 | 结果 |
|---|---|
| `GOMOK_PINNED_INFERENCE=1`：pinned 主机暂存 + 非阻塞拷贝 | RTX 5060 上无收益；推理 API 每批仍要同步，拷贝无法重叠。默认 0 |
| `GOMOK_PERSISTENT_WORKERS=1`：跨手保留搜索线程与 scratch | RTX 5060 上无收益。默认 0 |
| 融合 Adam | 单独 A/B 只有 0.5–1 %，在噪声范围内；仅为减少 kernel 启动而保留 |
| 生产 CUDA 路径上的子树复用 | 目前只在小型 CPU 模型上隔离测过（整局短 31 %、每手评估 −2.9 %）；CUDA 上已启用但未隔离 |
| cuDNN autotune | 故意关闭（见 B6） |
| 原生 CPU 训练 | 设计上拒绝：`train --cpu` 直接报错退出，而不是假装可以 |
| 256×16 形状的原生 CPU 推理 | 实测每前向 3,374.8 ms；可用于测试，不能用于对弈（见 B9） |

**约 12× 的归因。** 可以单独实测的部分是：精度（TF32/BF16 内核约 2–4×，一个监督阶段端到端 1.77×）、批量 GPU 评估（一次 32 个叶子而不是 1 个；批大小扫描说明 batch 1 时 GPU 是延迟受限而非吞吐受限）、子树复用与共享评估缓存（每手网络调用更少）、战术短路（已定胜负的局面不推理），以及把 Python/GIL 开销移出热路径（C++ 单进程）。只有精度倍数和批大小扫描是隔离实测；其余是机制解释加上约 14 小时的端到端结果。这个倍数比同类内核对比要小，是因为参考实现是 FP32 一次一局，而本次训练跑的是 2.35× 的参数和每手 400 次搜索。

---

## 网络与搜索

### 状态编码

四个平面，全部以「当前行棋方」视角编码（与 `Board::encode_state` 及 Python 工具一致）：己方棋子、对方棋子、最后一手、「是否黑方走」（常量平面）。网络因此看不到绝对颜色。

### 网络结构

```
state [4,15,15]
  stem      conv3x3 4 -> 256, BN, ReLU
  trunk     16 x [ conv3x3 256->256, BN, ReLU, conv3x3 256->256, BN, add, ReLU ]   （感受野 67x67）
  policy    conv1x1 256->2, BN, ReLU, flatten(450), fc 450->225, log_softmax
  value     conv1x1 256->1, BN, ReLU, flatten(225), fc 225->64, ReLU, fc 64->1, tanh
```

| 部件 | 16×256（参考） | 12×192（默认构建） |
|---|---:|---:|
| stem 3×3 4→C | 9,472 | 7,104 |
| 主干，16 / 12 块（占模型 99.3 % / 98.5 %） | 18,882,560 | 7,967,232 |
| 策略头（1×1 卷积 + 450→225 全连接） | 101,989 | 101,861 |
| 价值头（1×1 卷积 + 225→64 + 64→1） | 14,786 | 14,722 |
| **推理导出合计（BN 已折合）** | **19,008,807** | **8,090,919** |
| 训练检查点可学习参数（含 BN γ/β） | 19,025,709 | 8,100,525 |

19,025,709 这个数字就是训练器启动时打印的（`[net] CUDA backend enabled (19025709 parameters)`），也是 `runs/model_export_256x16/` 逐个文件重建出来的；折合导出恰好少了那 16,902 个 batch-norm γ/β 值。

用 `-DGOMOK_NET_CHANNELS=N -DGOMOK_NET_RES_BLOCKS=N` 可以重建其它形状，不同形状的检查点互不兼容。策略头是 450→225 的稠密层（1×1 卷积后跨棋盘位置共享权重），而不是旧的 900→225：旧层占全部参数的 62 %，且没有跨棋盘位置的权重共享。

### 搜索

- PUCT/MCTS：`c_puct 5`、`--playout 400`（参考训练的数据生成和 RL 都用 400），候选上限 60。
- 叶子按自身行棋方视角评估，回传时交替符号；终局按精确值处理。
- 一步必胜的叶子回传精确值且不推理；唯一必防点能存活到剪枝之后，并把展开收敛为单子节点。
- 虚拟损失只在同一棵树有多个叶子在飞时生效（串行的竞技场搜索会关闭它）。
- 根节点 Dirichlet 噪声（`ε 0.25`、`α 0.3`）在剪枝前对**所有**合法着法抽取；自对弈的策略标签用归一化的原始访问次数，温度只影响实际采样。
- 子树复用把已完成的访问与价值带到下一手；根噪声重抽，已访问的子节点保留，并补上当前的前排候选。
- 竞技场用相同开局交换执色，并同时报告单局 W/L/D 与开局对结果 —— 自由五子棋先手优势明显，只看总分会被误导。

---

## 训练流程

三个阶段，由一个 `train` 命令加上自对弈生成器驱动：

1. **监督预训练**（人类棋谱）—— 默认只用策略交叉熵。RenjuNet 各来源的开局/长连规则不同，其记录的胜负不能当作自由五子棋的价值标签：数据集元数据把 `value_weight = 0`，此时价值分支真正不进计算图。
2. **搜索蒸馏** —— 由已训练引擎生成的自对弈局面带真实自由规则胜负，因此其元数据把 `value_weight = 1`，它们**确实**监督价值头。256×16 流水线先训 6 轮人类数据，再训 2 轮蒸馏数据（用蒸馏数据集时即 `--pretrain 8`）。
3. **自对弈强化学习** —— 批量自对弈产局，回放环缓冲与外部人类棋谱样本按 `--mix-ratio` 混合，每批执行 `--epochs` 次全新优化更新。

| 方面 | 实现 |
|---|---|
| 优化器 | Adam（β 0.9/0.999，ε 1e-8），`--weight-decay 1e-4` 做 L2，全局范数裁剪 2.0 |
| 学习率 | 预热 `base·(0.1+0.9·(step+1)/warmup)`，随后余弦衰减到 10 % 下限；base 2e-3，RL 预热 2 批 / 监督预热 20 步 |
| 损失 | 策略交叉熵 + `value_loss_weight` × 加权价值 MSE（逐样本置信度按批内权重和归一化） |
| 批组成 | `--batch-size`（默认 512；256×16 训练用 256），其中 `--mix-ratio` 比例是外部样本，每个样本现场抽一个 D4 变换 |
| 回放 | FIFO 环（256×16 训练 `--buffer 50000`，旧训练 20000），落盘到 `trainer.replay` 以支持续训 |
| 精度 | `--precision bf16`（autocast）或 `fp32`；两者默认都启用 TF32 内核 |
| 检查点 | `best_policy_<tag>_bin`（最新）与 `best_policy_<tag>_val_bin`（选择得分最优）；`best` 前缀是历史命名，不代表最强 |
| 选择 | `--select-value-weight W --selfplay-val-every N` 在留出回放缓冲的自对弈探针上按 `policy + W · value` 给每个保留快照打分；`0 / 0` 恢复旧的只看策略行为 |
| 指标 | `runs/<tag>/metrics.csv` —— 固定训练/验证损失、价值误差、top-1、熵与选择权重；留出的自对弈探针与由此得到的选择分数打在 `[val]` 日志行上，不进 CSV |
| 续训 | 校验批号、种子、选项/环境合约、数据集路径/大小/mtime、回放内容与 Adam 状态；保留已保存的学习率视野，延长 `--games` 后停在 LR 下限 |
| 早停 | 监督阶段 `--early-stop-patience`；patience 会随检查点恢复；不会自动回滚到验证最优检查点 |
| 训练中的竞技场 | `--eval-games/--eval-freq/--eval-playout/--baseline`；256×16 训练每 50 批对自己那个监督检查点打 80 局 |

耗时日志不粉饰：`[train]` 分别打印 `selfplay_seconds` 与 `update_seconds`，`[progress]` 的 ETA 会把计划中的竞技场时间算进去。

**检查点格式与已知坏模型。** 训练器状态为 **v2** 格式（网络 + BN + Adam + 回放 + 调度视野 + 选项/数据合约）。早期 v1 实验状态在加载时会被拒绝，因为那版只优化策略的写法会把价值分支衰减到约 1e-37。256×16 谱系是 `best_policy_q256_pre_bin` → `best_policy_q256_distill_bin` → `best_policy_q256_rl_bin`，被选中的快照在 `best_policy_q256_rl_val_bin`。更早的、已验收的 12×192 最终检查点是 `best_policy_gomoku_fast_bin`；**不要**从 `best_policy_verify_learning_20260926_*` 继续训练 —— 那批检查点来自修复前的代码，只作为价值分支塌缩的证据保留。

---

## 数据集

两个分区供 256×16 流水线使用。人类棋谱只监督**策略**；蒸馏自对弈记录监督两个头。

| | 人类：RenjuNet（`data/gomoku_v2`） | 蒸馏自对弈（`data/selfplay_256x16_p400`） |
|---|---|---|
| 来源 | `data/renjunet.xml`，159,000 条记录 | 从 192×12 的 `best_policy_quality5000_20260926_bin` 做 3,000 局、每手 400 次搜索 |
| 接受的对局 / 局面 | 4,218 局 / 129,588 局面（规则 7/11/28 = Gomoku Pro / Swap2 / Swap；排除经典连珠） | 3,000 局 / 51,755 局面，每局 17.3 个 |
| 淘汰计数及原因 | 其它规则 138,416；交换开局 6,020；不足 10 手 10,317；终局/结果冲突 18；终局后仍有落子 6；重复对局 5 | 已审计：张量校验通过、自由规则、0 个终局策略样本 |
| 策略标签 | 前 5 手不作为标签；外部样本做 0.1 的合法着法平滑 | 原始 MCTS 访问次数 |
| 价值监督 | 关闭（`value_weight 0`），因为来源规则与本引擎不同 | 开启（`value_weight 1`），胜负为 27,226 / 24,529 |
| 划分（种子 1，10 %） | 116,781 训练 / 12,807 留出 / 619 因与训练集 D4 等价被剔除 → **12,188 独立验证** | 按对局 ID 分组；按行划分会泄漏 80.4 % 的对局 |
| 存储 | float32 `[N,4,15,15]` 状态、`[N,225]` 合法策略、`[N]` 结果、uint64 对局 ID、JSON 元数据；557.6 MiB，mmap 读取 | 同格式 |

二进制数组以「小端 `uint64` 元素个数」开头。载入时会逐条复检（平面为 0/1 且有限、无重叠、最后一手归属正确、策略归一化且已占格概率为 0、黑白子数与行棋方一致、|z| ≤ 1、不含已终局局面）。没有对局 ID 的历史数据只能用 `--infer-game-groups 1` 分组，且要求完整的顺序对局 —— 随机按行划分无法修复混规则或截断的数据，会被直接拒绝。

---

## 强度测评

自由五子棋先手优势极大，因此本项目只报告配对开局结果，竞技场同时输出单局 W/L/D、得分、先手胜率和开局对战绩。

为本 README 在 256×16 谱系上新跑的测量（`build-256x16/`，种子 1，本地检查点）。下面的数字取自训练自己在**第 50 批**跑的竞技场日志；那个检查点后来已被第 250 批覆盖，所以它是一条历史记录，命令只用来展示协议：

```sh
# 训练当时实际做的（runs/train_256x16.log 的 [arena] batch 50）：
#   --eval-games 80 --eval-freq 50 --eval-playout 400 --baseline best_policy_q256_distill_bin
# 第 50 批的快照已被第 250 批检查点覆盖，所以引用的那行无法重跑；
# 下面是等价的独立命令形式：
./build-256x16/gomoku_train arena --model <第 50 批的快照> \
  --baseline best_policy_q256_distill_bin --games 80 --playout 400 \
  --opening-moves 4 --seed 1
# [arena] batch 50 41W 37L 2D score:0.5250 first_player_win_rate:0.795
#         games:80 pairs:6W/4L/30D seconds:1058.0
```

旧 12×192 二进制无法加载同一形状（主干是编译期常量），所以 256×16 的基线就是这次运行自己的监督检查点：那个竞技场测的是**RL 带来了什么**，而不是绝对强度，而 80 局的 0.525 还在「暂时没有差别」的噪声范围内。跨形状比较只能通过 ONNX 导出进行。

更早用 12×192 网络记录、协议族相同：

```sh
./build-training/gomoku_train arena --model best_policy_quality5000_20260926_bin \
  --baseline best_policy_fromzero_20260926_bin --games 80 --playout 400 \
  --opening-moves 4 --seed 1
# arena 58W 19L 3D score=0.74375 first_player_win_rate=0.688312 games=80 pairs=21W/1L/18D

./build-training/gomoku_train arena --model best_policy_verify_rl_20260926_bin \
  --baseline <export-of-the-old-batch-170-model> --games 40 --playout 400 \
  --opening-moves 4 --seed 1
# 40W 0L 0D, pairs 20W/0L/0D, first-player win rate 0.500, 42.1 s
```

价值质量用**新**自对弈局面检验，而不是训练数据：换一个未使用过的种子生成 64 局共 855 个局面，剔除与回放缓冲 D4 重叠的 167 个，对剩余 688 个逐局评估。在那个 12×192 模型上按局平均 MSE：0.1359，最佳执色常数（乐观参考）0.2877，最佳全局常数 0.9939，恒输出 0 为 1.0000。对 256×16 模型，`runs/q256_rl/metrics.csv` 里报告的是**混合验证集**的价值 MSE，在 250 批内从 1.091 降到 0.751 —— 那一列是「人类 + 蒸馏」验证划分，不是自对弈探针。检查点选择真正打分的探针只写在日志里：`probe_value` 在第 10 批为 0.806、第 40 批 0.602、第 50 批 0.705、第 55 批 0.522。

如何解读这些数字：

- 单种子、单协议、单对手 —— 不是 Elo，也不代表对所有开局的结论；
- 参数量变大、训练损失下降、测试通过、内核变快，任何一项都不足以证明棋力提高；
- 先手胜率明显偏离 50 % 时，主导比分的是开局协议，而不是模型强弱；
- 256×16 竞技场在第 50 批**还**没显示出 RL 打赢它自己的监督检查点，上面的表格也照实这么说，而不是把 0.525 粉饰一番。

---

## 部署：ONNX、量化、浏览器与终端对弈

训练检查点保留原始卷积 + BN 张量，以便忠实续训；导出器读取这些权威张量并为推理折合 BN：

```sh
./build-256x16/gomoku_train export --model best_policy_q256_rl_val_bin --output model_export_256x16
python3 tools/export_onnx.py --bin-dir model_export_256x16 256x16.onnx
# 256x16.onnx: 76,050,598 bytes (72.5 MiB), log_policy [B,225] + value [B,1]

python3 tools/play_onnx.py   --model 256x16.onnx --playout 400 --device cuda   # 终端对弈
python3 tools/server_onnx.py --model 256x16.onnx --playout 400 --device cuda \
  --host 127.0.0.1 --port 8000                                                # 浏览器对弈
```

ONNX Runtime 不需要 PyTorch，但它的 CUDA provider 链接的是 pip 装在 `site-packages/nvidia` 下的 CUDA/cuDNN 库 —— 那里不在动态库搜索路径上。因此引擎在创建 session 前会调用 `onnxruntime.preload_dlls()`；不做这一步，ONNX Runtime 只会打一条 warning 然后静默退回 CPU。现在 `--device cuda` 在 provider 无法加载时会直接报错，只有 `--device auto` 才允许回退。

### 原生 C++ 引擎 vs ONNX Runtime（256×16，本机，2026-10-09）

`tests/play_benchmark.cpp` 是 `tools/gomoku.py` 的 MCTS 的逐行镜像 —— 同样的候选生成、同样的战术短路、同样的选择公式与打破平局规则 —— 所以两个驱动器搜索的是同一棵等价树，只有后端不同。两者都对同一局 30 手棋、每手 200 次搜索做复盘：

| 测量项 | C++ CUDA | ONNX Runtime CUDA |
|---|---:|---:|
| 前向，batch 1 | 2.138 ms（468 pos/s） | **1.752 ms（571 pos/s）** |
| 前向，batch 8 | **6.535 ms（1,224 pos/s）** | 9.923 ms（806 pos/s） |
| 前向，batch 32 | 22.896 ms（1,398 pos/s） | 23.358 ms（1,370 pos/s） |
| 30 手复盘，每手 200 次搜索 | 10.31 s 总计，343.8 ms/move | 10.45 s 总计，348.2 ms/move |
| 这 30 手中的网络调用次数 | 4,798 | 4,798 |
| 两个引擎选择不同着法的位置数 | 0 | 0 |

在 MCTS 的批大小下，两个后端端到端相差约 1.3 %，`batch 1` 偏向 ONNX Runtime，更大批偏向 C++：每手 200 次搜索时，搜索是一串小而受延迟约束的调用，这正是 A4 的收集窗口比原始吞吐更重要的原因。C++ 原生 CPU 路径（`--cpu`）在这个形状上完全是另一回事：每前向 3,374.8 ms，即标量运算约 2.5 GFLOPS，而单线程 BLAS 能提供 147 GFLOPS —— 可用于测试，不能用于对弈。

### 量化

`tools/quantize_onnx.py` 生成 FP32 导出的部署变体，并在大小旁边报告数值漂移（校准局面取自数据集开头，质量局面取自末尾，两者不重叠）：

| 变体 | 大小 | top-1 一致率 | 平均 KL | 最大 KL | 价值 MAE | ORT CPU, batch 1 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 76.1 MB | — | — | — | — | 约 300–320 ms |
| FP16 | 38.0 MB | 100.00 % | 0.00000 | 0.0000 | 0.0003 | 约 12–15 ms |
| INT8（QDQ，逐通道） | 19.2 MB | 93.23 % | 0.00585 | 0.0228 | 0.0674 | 6 ms |

FP16 是白赚的 —— 着法不变、体积减半、CPU 上快约 20–25×（表里约 300 → 约 12 ms，重跑一次是 321 → 15 ms）。INT8 会在约 7 % 的局面上改变首选着法，所以把它当作纯 CPU 部署的一个选项，而不是直接替换。以上在 CPU 上用 8 个 intra-op 线程测量（`runs/quantized/` 里还有 FP8、INT4 文件，但提交的工具不生成它们）；这些数字不影响 CUDA provider。

---

## 仓库结构

```
src/            C++17 引擎
  game.{h,cpp}        棋盘、规则、Zobrist 局面键、状态编码
  net_config.h        网络形状的唯一来源
  network.{h,cpp}     PureNet：原生折合 BN 的 CPU 推理、参数与检查点读写
  network_cuda.cpp    LibTorch/CUDA 后端：训练步、批量推理、TF32/BF16、融合 Adam
  selfplay.{h,cpp}    批量无锁 MCTS、批量自对弈、竞技场、子树复用
  dataset.{h,cpp}     mmap 数据集、D4 增强、按整局划分并剔除 D4 重叠
  train.{h,cpp}       Trainer：预训练、蒸馏、自对弈 RL、回放、选择、检查点
  gradient.{h,cpp}    参考 CPU 内核（由梯度测试使用）
  main.cpp            CLI：train | selfplay | arena | human | export
tools/          Python 工具链：数据集转换/审计、ONNX 导出/量化、
                C++-vs-ONNX 对弈基准、Python MCTS 镜像、学习对比基准、网页服务
tests/          C++ 测试/基准可执行文件 + Python unittest 套件
data/           数据集前缀（大文件本地生成，不入库）
runs/           每个 tag 的 metrics.csv、训练日志、ONNX 导出（本地生成）
analysis/       评审笔记、旧基准 JSON/日志、analysis/play_bench（C++ vs ONNX 报告）
```

## 测试与 CI

```sh
ctest --test-dir build-training --output-on-failure     # 本机 11/11，71.0 s（12x192 构建）
GOMOK_RUN_TRAINING_TESTS=1 GOMOK_TRAIN_BINARY=./build/gomoku_train \
  python3 -m unittest discover -s tests -p 'test_*.py'
```

覆盖面：棋盘规则与终局、战术搜索、价值符号、真实并发自对弈叶队列、合法策略标签、数据集分组/增强/D4 不变量、CPU↔CUDA 后端数值一致性、优化器与检查点恢复及写盘失败注入、CLI 错误处理、ONNX 导出数值一致性、Python 工具。CI 会用缩小主干构建 CPU 引擎并跑 ASan+UBSan，另有独立的生产形状编译任务。`gomoku_play_benchmark` 是基准目标，不是 `ctest` 测试。

## 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `GOMOK_SEED` | 1（`--seed` 优先） | 随机种子 |
| `GOMOK_MCTS_THREADS` | `min(8, 物理核数)` | 搜索 worker 数，钳制 1–32 |
| `GOMOK_LEAF_BATCH` | 由推理目标推导 | 每局在飞叶子数，钳制 1…playout |
| `GOMOK_TREE_REUSE` | **1** | 跨手子树复用；`0` 恢复每手重建 |
| `GOMOK_BATCH_WAIT_US` | CUDA 100，CPU 0 | 推理收集窗口 |
| `GOMOK_CACHE_ENTRIES` | 32768 | 每批评估缓存容量 |
| `GOMOK_PROFILE` | 关 | 打印自对弈评估/缓存/批统计 |
| `GOMOK_TF32` / `GOMOK_INFERENCE_TF32` | 1 / 1 | 训练 / 推理的 TF32 内核 |
| `GOMOK_FUSED_ADAM` | 1 | 融合 Adam 更新 |
| `GOMOK_DETERMINISTIC` | 0 | 确定性 cuDNN 内核（更慢） |
| `GOMOK_PINNED_INFERENCE` | 0 | pinned 暂存实验（实测无收益） |
| `GOMOK_PERSISTENT_WORKERS` | 0 | 常驻搜索线程（实测无收益） |
| `GOMOK_LOG_FILE` | 关 | 日志同时写入文件 |
| `GOMOK_RUN_TRAINING_TESTS` | 关 | 启用 GPU 训练生命周期测试 |

## 代码导览

| 路径 | 内容 |
|---|---|
| [`src/net_config.h`](src/net_config.h) | 网络形状的唯一来源 |
| [`src/selfplay.cpp`](src/selfplay.cpp) | 批量无锁 MCTS、自对弈主循环、竞技场、子树复用 |
| [`src/network_cuda.cpp`](src/network_cuda.cpp) | TF32/BF16、BN 折合、融合优化器、检查点读写 |
| [`src/train.cpp`](src/train.cpp) | 预训练、蒸馏、自对弈 RL、回放、选择、续训合约 |
| [`src/main.cpp`](src/main.cpp) | `gomoku_train` 命令行入口 |
| [`tools/train_256x16_quality.sh`](tools/train_256x16_quality.sh) | 参考的 256×16 训练 |
| [`tools/train_quality_5000.sh`](tools/train_quality_5000.sh) | 更早的 12×192 5,000 局训练 |
| [`tests/play_benchmark.cpp`](tests/play_benchmark.cpp) | Python 搜索的 C++ 镜像，用于后端计时 |
| [`tools/gomoku.py`](tools/gomoku.py) | C++ 驱动器所镜像的 Python 棋盘/状态编码/MCTS |
| [`tools/bench_play_onnx.py`](tools/bench_play_onnx.py) | 该对比中的 ONNX Runtime 一侧 |
| [`tools/quantize_onnx.py`](tools/quantize_onnx.py) | FP16/INT8 变体与漂移报告 |
| [`tools/benchmark_learning.py`](tools/benchmark_learning.py) | FP32 与 BF16 的隔离监督对比驱动器 |
| `analysis/play_bench/report.md` | C++-vs-ONNX 的书面报告及原始日志（保存在本地：`analysis/` 未纳入版本控制） |
| [`tests/`](tests) | C++ 测试/基准目标与 Python unittest 套件 |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | CPU ASan/UBSan 任务与生产形状编译任务 |

## 口径、来源与限制

- **7 天基准的来源**：约 7 天这个参考数字是 `junxiaosong/AlphaZero_Gomoku`（PyTorch，15×15，同等自对弈 + 训练量）在**本机**的实测记录，保存在本仓库之外，因此无法由仓库产物重新推导。
- **约 14 小时的来源**：256×16 那次训练没有保留任何一份完整的干净 stdout 日志，因此该数字是带上下界的重建。**实测**：续训那一段（锁 `2026-09-29 01:29` → 最终检查点 `12:35`）在 11.1 h 内跑完 200 批和 4 次竞技场，外推到 250 批约 13.9 h。**运行自身的投影**：`[progress]` 行的 `elapsed` + **剩余** `ETA` 相加给出全程约 16.3 h（第 49 批 `2:53:16 + 13:23:38`，第 56 批 `3:35:53 + 12:40:07`），稳态为每批 211 s。标题采用较低的、基于实测的一端。监督阶段按其实测吞吐再加约 17 分钟，3,000 局蒸馏批次则是另一次性的数据生成步骤。流水线被中断过一次并从第 50 批检查点续上；25 批的检查点间隔限定了最多损失的工作量。重跑该脚本能复现协议；具体墙钟时间受机器与散热影响。
- **7 天 vs 14 小时比较的是同等训练量，而不是同一份代码。** 原项目跑 FP32、一次一局、每次前向一个叶子；7 天这个数字是在同一台笔记本上、按 256×16 训练所用训练量对它做的外部测量。它只是一个数据点，不是一条基准曲线。
- 表中的数字来自本仓库可复现的命令，或本次在本机为 README 新做的测量；除特别说明外均为单次运行、单种子结果。
- 微基准不含自对弈与全量验证；端到端数字包含二者。
- 基于旧网络、旧数据，或当时 TF32 并未真正关闭的早期测量，不会被搬进本 README；本地笔记与可复现命令冲突时，以命令为准。在 12×192 形状上测的行都标注了形状。
- 训练按设计只支持 CUDA；CPU 构建提供推理、搜索、竞技场与测试。
- RenjuNet 的开局与长连规则不同于本引擎的自由规则，因此人类棋谱分区只监督**策略**；价值学习全部来自自对弈，或从已训练引擎蒸馏，或在 RL 过程中生成。

## 参考与致谢

- 本项目的参考实现：[junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)。
- AlphaZero 方法：Silver 等，*A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*（[arXiv:1712.01815](https://arxiv.org/abs/1712.01815)）。
- 训练数据：RenjuNet 棋谱（`data/renjunet.xml`）。
- 数值精度参考：[PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)。

本项目采用 [MIT 许可证](LICENSE) © 2026 LangLang03。
