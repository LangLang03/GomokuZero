# GomokuZero — C++17 / CUDA 实现的 15×15 AlphaZero 五子棋

[![CI](https://github.com/LangLang03/GomokuZero/actions/workflows/ci.yml/badge.svg)](https://github.com/LangLang03/GomokuZero/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

从零实现的 **15×15 自由规则五子棋** AlphaZero 引擎（五连或更长即胜，无连珠禁手、无交换开局）。棋盘、PUCT/MCTS、批量自对弈、数据管线、训练器、检查点、竞技场与 CLI 全部是 C++17；神经网络训练走 LibTorch/CUDA，推理走原生「BN 折合」CPU 路径。Python 工具链负责人类棋谱处理、ONNX 导出以及浏览器/终端对弈服务。

起点是著名的 Python 项目 [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)。在下文这台笔记本上，一次同等规模的 15×15 训练在那里约需 **7 天**，在这里约 **14 小时** —— 实测口径与机制说明见[结果速览](#结果速览)和[为什么快：优化清单](#为什么快优化清单)。

| | |
|---|---|
| 规则 | 15×15 自由规则五子棋，五连及以上即胜，黑先 |
| 网络 | 残差主干 **12 块 × 192 通道**，8,100,525 参数，主干感受野 51×51 |
| 输入 / 输出 | `[B,4,15,15]` 四个平面 → `log_policy [B,225]` + `value [B,1]`（tanh） |
| 搜索 | 批量 PUCT/MCTS，默认每手 400 次搜索，虚拟损失、子树复用、战术精确值 |
| 训练 | 人类棋谱监督预训练 → 自对弈强化学习，Adam + 余弦调度，可选 BF16 |
| 技术栈 | C++17、CMake、LibTorch/CUDA 训练、原生 CPU 推理、ONNX Runtime 部署 |
| 数据 | RenjuNet XML → 4,218 局 / 129,588 局面，按整局划分并剔除 D4 泄漏 |

---

## 目录

- [结果速览](#结果速览)
- [基准设备](#基准设备)
- [快速开始](#快速开始)
- [为什么快：优化清单](#为什么快优化清单)
- [网络与搜索](#网络与搜索)
- [训练流程](#训练流程)
- [数据集](#数据集)
- [强度评测](#强度评测)
- [部署：ONNX、浏览器与终端对弈](#部署onnx浏览器与终端对弈)
- [仓库结构](#仓库结构)
- [测试与 CI](#测试与-ci)
- [环境变量](#环境变量)
- [代码导览](#代码导览)
- [口径、来源与限制](#口径来源与限制)

---

## 结果速览

基准流水线：`tools/train_quality_5000.sh` —— 从随机权重做 6 轮监督预训练，然后 **250 个自对弈批次 × 20 局 = 5000 局**，每批 24 次优化更新（共 6000 次），BF16，每手 256 次搜索，回放容量 20000，每批 35% 人类棋谱样本，其间跑 5 次 80 局竞技场。

| | 参考实现 junxiaosong/AlphaZero_Gomoku（PyTorch） | 本仓库（C++17 + LibTorch/CUDA） |
|---|---:|---:|
| 同机、同等 15×15 训练量 | **约 3 天** | **约 2.5 小时** |
| 加速比 | — | **约 29×** |

这 2.5 小时的实测构成（本机 `build-training/`）：

| 阶段 | 实测 | 依据 |
|---|---:|---|
| 监督预训练 6 轮（116,781 局面 × 6，5,478 次更新，全量 12,188 局面验证） | 约 10 分钟 | `tools/train_quality_5000.sh` 第一阶段，本地运行（按产物时间戳 18:19 → 18:29） |
| 自对弈 RL：250 批 / 5000 局 / 6000 次更新 | 约 2 小时 21 分 | 同一次运行的第二阶段（18:29:00 → 20:49:49） |
| 整条流水线 | 约 2 小时 31 分 | 上述两阶段之和；口径说明见[口径、来源与限制](#口径来源与限制) |

同机、同 192×12 网络的支撑实测：

| 测量项 | 结果 |
|---|---|
| 后端微基准，batch 128，预热 3 步后测 8 步 | TF32：推理 41.9 ms（3,053 样本/秒），训练 152.0 ms（842 样本/秒） |
| 同上，BF16 训练 | 76.6 ms/batch（**1,670 样本/秒**）—— 相对 TF32 **1.98×**，相对提交 `35f5872` **2.14×** |
| 同上，严格 FP32（关闭 TF32） | 推理 172.8 ms（741 样本/秒），训练 354.6 ms（361 样本/秒）—— TF32 值 **4.1× / 2.3×** |
| 监督阶段 3 轮，同种子/同数据/同更新次数 | FP32 460.4 s → BF16 260.5 s（**1.77×**）；最终验证策略交叉熵 2.0105，top-1 44.81% |
| 生产模型自对弈吞吐，每手 256 次搜索，开启子树复用 | 8 局 6.13 s = **0.77 s/局**、56 ms/手，推理批均值 15.6 / 峰值 32，缓存命中 13.4% |
| 竞技场 80 局，每手 400 次搜索，4 手配对开局 | `quality5000`（第 250 批）对 `fromzero`（第 150 批）**58 胜 19 负 3 和**，得分 0.744，开局对 21 胜 1 负 18 和 |
| 价值头在 64 局全新自对弈上（已剔 D4 重叠） | 按局平均 MSE **0.1359**，执色常数基线（乐观参考）0.2877 |
| C++ 测试套件（生产构建） | `ctest` **11/11，35.3 s** |
| 数据集载入 + 校验 + 按整局划分 | 129,588 局面 **0.394 s**，峰值 RSS 570 MiB |

上表每个数字都能追到一条命令或一个产物；它们不是 Elo，也不代表对所有对手、所有开局的结论。更早那次对第 170 批模型 40 胜 0 负的竞技场及其协议见[强度评测](#强度评测)。

---

## 基准设备

本 README 的所有耗时都产自这一台笔记本，基线如下。请把它们当作「单机证据」，而不是与硬件无关的承诺。

| 组件 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8,151 MiB，计算能力 12.0（sm_120），驱动 610.57.04，CUDA UMD 13.3 |
| CPU / 内存 | Intel Core i7-14650HX（24 逻辑核），15 GiB |
| 系统 / 内核 | Arch Linux，内核 7.1.9-arch1-2，glibc 2.44 |
| 编译器 / 构建 | GCC 16.2.1，CMake 4.4.2，`-DCMAKE_BUILD_TYPE=Release`，C++17 |
| C++ 训练后端 | LibTorch **2.12.0+cu130**（CUDA 13.0，cuDNN 9.24），由 CMake 通过 `/usr/bin/python3.14` 发现 |
| Python 工具 | `.venv/` 内 Python 3.14：torch 2.14.0+cu130、numpy 2.5.3、onnx 1.23.0 |
| 部署 | onnxruntime-gpu 1.28.0（CUDA 13 / cuDNN 9），CUDA 库来自 pip 的 `nvidia-*` wheel |

**不需要 nvcc**：CUDA 后端就是 LibTorch，CMake 只需要一个能 `import torch` 的 Python 解释器和 `libtorch_cuda.so`。机器上装了多个环境时用 `-DPython3_EXECUTABLE=...` 指定（本机同时存在系统 torch 2.12 与 venv torch 2.14）。

---

## 快速开始

### 1. 构建与测试

```sh
# 依赖：CMake >= 3.18、C++17 编译器、带 CUDA 的 torch
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'   # 例如 2.12.0+cu130 13.0

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGOMOK_ENABLE_CUDA=ON -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

也支持无 CUDA 构建，用于推理、搜索和测试；**没有 LibTorch/CUDA 后端时训练会被拒绝**，而不是悄悄降级：

```sh
cmake -S . -B build-cpu -DGOMOK_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu -j2 && ctest --test-dir build-cpu --output-on-failure
```

### 2. 对弈、观察自对弈、跑对战

```sh
./build/gomoku_train human --model best_policy_quality5000_20260926_bin --playout 400   # 输入 r,c（0-14）

GOMOK_PROFILE=1 ./build/gomoku_train selfplay \
  --model best_policy_quality5000_20260926_bin --games 8 --batch 8 --playout 256 --seed 1

./build/gomoku_train arena --model best_policy_quality5000_20260926_bin \
  --baseline best_policy_fromzero_20260926_bin --games 80 --playout 400 \
  --opening-moves 4 --seed 1
```

检查点（`best_policy_*_bin/`）是本地产物，不入版本控制：先自己训一个，或用 `./build/gomoku_train export --output my_model_bin` 导出随机权重。

### 3. 准备数据并训练

```sh
python3 tools/parse_renjunet.py data/renjunet.xml --bin-prefix data/gomoku_v2
python3 tools/audit_dataset.py data/gomoku_v2 --state-overlap   # 审计 JSON 输出到 stdout

# 监督预训练（绝对目标 6 轮；--games 0 表示只做监督）
./build/gomoku_train train --mix data/gomoku_v2 --pretrain 6 --games 0 \
  --batch-size 128 --precision bf16 --seed 1 --val-samples 0 --tag pretrain

# 从验证最优检查点开始自对弈 RL（绝对目标：第 250 批）
./build/gomoku_train train --init best_policy_pretrain_val_bin --mix data/gomoku_v2 \
  --pretrain 0 --games 250 --batch-games 20 --batch-size 128 --precision bf16 \
  --playout 256 --epochs 24 --buffer 20000 --mix-ratio 0.35 --seed 1 \
  --val-samples 0 --check-freq 25 --tag rl
```

带竞技场、环境变量固定和 5000 局预算的参考流水线是 [`tools/train_quality_5000.sh`](tools/train_quality_5000.sh)，它会拒绝覆盖已有检查点。注意 `--games` 和 `--pretrain` 都是**绝对目标**而非增量：从第 100 批续训并写 `--games 250`，只会再生成 150 批。

### 4. 导出与部署

```sh
./build/gomoku_train export --model best_policy_quality5000_20260926_bin --output model_export
python3 tools/export_onnx.py --bin-dir model_export model.onnx

python3 -m pip install 'onnxruntime-gpu[cuda,cudnn]'
python3 tools/server_onnx.py --model model.onnx --playout 400 --device cuda \
  --host 127.0.0.1 --port 8000        # 浏览器打开 http://127.0.0.1:8000
```

---

## 为什么快：优化清单

参考实现是单进程 PyTorch 训练器：一次只跑一局、每次前向只评估一个叶子、FP32、每手都重建搜索树。下表的「基线」都指这种朴素做法，每一行都附证据。

### A. 自对弈并行与搜索调度

| # | 优化 | 基线做法 | 本实现 | 证据 / 收益 |
|---|---|---|---|---|
| A1 | **批量自对弈**：一次调用推进 `--batch-games` 局（默认 64） | 一轮一局 | 64 棵树同时在飞，互相供给推理批次 | `src/selfplay.cpp` `BatchedSelfPlay::run_batch`；生产 profile 推理批均值 7–15、峰值 32 |
| A2 | **无锁 MCTS worker 池**（条件变量按轮启动） | 单线程搜索 | `min(8, 物理核)` 个 worker，钳制 1–32 | `SearchWorkerPool`；`GOMOK_MCTS_THREADS`、`--threads` |
| A3 | **CPU 亲和性：把最快的核留给 GPU 协调线程** | 不绑核，搜索线程与推理线程互相抢核 | 读取物理核拓扑与最大频率，跳过最快的 `inference_threads` 个核，其余轮转绑定 | `src/selfplay.cpp` |
| A4 | **批量叶子评估 + 收集窗口**：CUDA 下自旋最多 `GOMOK_BATCH_WAIT_US`（100 µs）后提交一次 GPU 调用 | 一个叶子一次前向，GPU 在两次调用之间空转 | 目标推理批 32；所有对局的叶子合并成一批 | 批大小扫描 16…512 → **32 最优**（2846 eval/s，128 只有 2588）；每手 400 次搜索时 **7.30 s/局 → 6.26 s/局** |
| A5 | **定点虚拟损失**（整数原子计数） | 无虚拟损失，多线程挤向同一子节点 | visits 用整数、价值和用 Q20 定点，默认 8；叶批 ≤1 时自动关闭 | 扫描：重试 −32%，相对旧默认 3 **每局快 6%** |
| A6 | **碰撞重试不消耗搜索预算** | 重试算一次搜索 | 命中 `NODE_EXPANDING` 返回 `Retry`，不递减已发放的搜索次数 | `src/selfplay.cpp` 重试循环，预算用 CAS 钳制 |
| A7 | **跨手子树复用**（`GOMOK_TREE_REUSE=1`，默认开） | 每手 `reset_tree()`，400 次搜索从零开始 | 提根：所选子节点及其后代搬进备用竞技场，已完成的访问计入下一手预算 | CPU 对照：8 局 **9.40 s → 6.44 s**；固定棋路对照：搜索循环 0.040 s → 0.012 s。**生产 CUDA 路径尚未实测** |
| A8 | **跨手局面评估缓存**：Zobrist 键 + FIFO 环，32,768 条，一个批次内 64 局共享 | 每个叶子都重新评估 | 生产自对弈命中率 0.27–0.41；同一批内的重复叶子也会去重 | `GOMOK_CACHE_ENTRIES`；profile 输出 `cache_hits`/`cache_hit_rate` |
| A9 | **战术与终局短路** | 每个叶子都问网络 | 一步必胜 → 精确 +1 且不推理；唯一必防点把展开收敛为单子节点；终局与已解节点永不进网络 | `src/selfplay.cpp`；注释明确：精确战术优先于未训练的价值头 |
| A10 | **预分配节点竞技场与 scratch**（节点 32 字节对齐，搜索期零分配） | 每个节点一个 Python 对象 | 每局每批一个竞技场，worker scratch 跨手复用 | `alignas(32) GNode`、`SearchWorkerScratch`，缓冲在 `run_batch()` 外一次性分配 |
| A11 | **竞技场批量对战** | 一次一局 | 最多 32 局同步推进，每轮每个模型只发一次批量调用 | `play_match`；上文 80 局竞技场在 400 次搜索下约 2 分钟跑完 |

### B. 精度与 GPU 内核

| # | 优化 | 基线做法 | 本实现 | 证据 / 收益 |
|---|---|---|---|---|
| B1 | **TF32 同时作用于 cuDNN 与 cuBLAS**，RAII 作用域、默认开 | 纯 FP32 内核 | `GOMOK_TF32` / `GOMOK_INFERENCE_TF32`（默认 1） | 严格 FP32 → TF32：推理 172.8 → 41.9 ms（**4.1×**），训练 354.6 → 152.0 ms（**2.3×**） |
| B2 | **BF16 训练 autocast**，参数/损失/Adam 动量保持 FP32，头部最后两个非线性强制 FP32 | 只有 FP32 | `--precision bf16` | 训练 152.0 → 76.6 ms（相对 TF32 **1.98×**，相对 HEAD 2.14×）；端到端监督阶段 460.4 s → 260.5 s（**1.77×**），同种子 3 轮实验策略质量无损失 |
| B3 | **融合优化器与融合梯度运算** | 逐参数 Python 循环 | `at::_fused_adam_`、`_foreach_norm` 求全局范数、`_foreach_mul_` 裁剪、5 个标量合并为一次回读 | `GOMOK_FUSED_ADAM=1`；单独 A/B 只有 0.5–1%，留着是为了减少 kernel 启动 |
| B4 | **BN 折合进卷积**，每次权重换代只折一次 | 每次前向都跑 BN kernel | 推理图里只剩 conv + bias + ReLU | `src/network_cuda.cpp` `fold_bn`；自对弈的成千上万次前向完全不跑 BN |
| B5 | **`torch::InferenceMode` + `eval()` + channels-last** | 带 autograd 图的 FP32 NCHW | 无图、无版本计数、对 Tensor Core 友好的内存布局 | `src/network_cuda.cpp` |
| B6 | **故意关闭 cuDNN autotune** | `cudnn.benchmark = True` | MCTS 批大小不断变化，autotune 是净亏；需要确定性时用 `GOMOK_DETERMINISTIC=1` | `src/network_cuda.cpp` 注释 |
| B7 | **冗余卷积 bias 不参与优化** | 所有参数都更新 | 冻结卷积 bias（BN 使其冗余，且其舍入噪声会被 Adam 放大），同时缩小 Adam 状态 | `requires_grad(false)` |
| B8 | **无监督时价值分支根本不进计算图** | `policy + 0 * value` 仍会为价值分支生成零梯度，被 Adam 的 L2 慢慢毁掉 | 关闭价值监督时价值项直接不加入图，价值参数使用独立的 Adam 步数 | 修复了一次实测到的价值头塌缩：旧写法下价值头稠密层权重衰减到约 1e-37 |
| B9 | **原生 CPU 推理路径**（折合 BN、8 路展开卷积、线程本地 scratch、按样本并行） | — | 没有 CUDA 也能跑 MCTS、CLI 和测试；CPU 训练被明确拒绝而非降级 | `src/network.cpp`、`src/network_cuda_stub.cpp` |

### C. 数据、IO 与检查点

| # | 优化 | 基线做法 | 本实现 | 证据 / 收益 |
|---|---|---|---|---|
| C1 | **数据集 mmap + 只存索引的划分** | 全量读进内存当张量 | `mmap(PROT_READ, MAP_PRIVATE)` + 逐条校验；训练/验证只是行索引，不复制数据 | 129,588 局面校验并划分 **0.394 s**，峰值 RSS 570 MiB |
| C2 | **在线 D4 增强**（8 种对称，状态与策略同一变换） | 预展开或无增强 | 每次优化更新重新抽样并重新选对称 | 8× 样本效率、1× 存储；`--augment 1` |
| C3 | **每次更新都换新 minibatch** | 同一 minibatch 反复更新 | 每步重新抽回放样本与外部样本，日志记录 `replay_draws`/`external_draws` | 避免只拟合少量记录（也让 `--epochs` 等于真实更新次数） |
| C4 | **流式、带校验的数据工具** | 纯内存 numpy 管线 | `npz_to_bin.py` 逐成员 mmap；`parse_renjunet.py` 逐局校验 + D4 去重；临时文件 + rename 写出 | 159,000 条 XML → 4,218 局，并留有完整淘汰计数 |
| C5 | **按整局划分 + 剔除 D4 泄漏** | 按局面随机划分 | 用 `splitmix64(id ^ seed)` 对局 ID 排序划分，再剔除与训练集 D4 等价的验证局面；C++ 与 Python 同一算法并有跨语言黄金测试 | 随机划分会泄漏约 88.9% 的对局；整局划分后剩 619/12,807 = 4.83% D4 重叠并剔除 → **12,188 个独立验证局面** |
| C6 | **检查点原子换代 + 恢复合约** | 只存/读权重 | `.staging` → rename，带 `.previous` 回滚；网络 + BN + Adam + 回放 + 批号 + 调度视野 + 种子 + 选项/数据指纹，全部校验后才使用 | 已完成的训练不会被悄悄从头重跑；`--games`/`--pretrain` 在续训后仍是绝对目标 |
| C7 | **诚实的 ETA 与分离计时** | 每轮墙钟时间 | 自对弈与更新分开计时，按近期批次平滑，定期竞技场单独计入 | `[train]`、`[progress]` 日志 |

### D. 工程

| # | 优化 | 理由 |
|---|---|---|
| D1 | 网络形状只在 `src/net_config.h` 定义一次，C++ 与 Python 共用 | 过去形状散落在四处，网络、导出器和测试可能互相不一致 |
| D2 | `gomoku_core` 只编译一次的静态库，所有可执行文件与测试链接它 | 避免每个目标都重编昂贵的 LibTorch 翻译单元 |
| D3 | 默认开 LTO，但 LibTorch 那个 TU 单独用 `-O1 -fno-lto` | 既保留发布优化，又约束编译时间与内存 |
| D4 | CI 用缩小主干跑 ASan + UBSan，另有一个生产形状的编译任务 | 并发与内存缺陷能被抓到，又不必让每个任务都付 12×192 的代价 |
| D5 | 严格 CLI 解析（未知/重复/缺值直接失败），模型加载失败绝不静默换成随机权重 | 杜绝「训错了实验却毫无察觉」这类失败 |

### 没赚到的尝试（保留但默认关闭）

| 尝试 | 结果 |
|---|---|
| `GOMOK_PINNED_INFERENCE=1`：pinned 暂存 + 非阻塞拷贝 | RTX 5060 上无收益；推理 API 每批仍要同步，拷贝无法重叠。默认 0 |
| `GOMOK_PERSISTENT_WORKERS=1`：跨手保留搜索线程与 scratch | RTX 5060 上无收益。默认 0 |
| 融合 Adam | 单独 A/B 只有 0.5–1%，在噪声范围内；仅为减少 kernel 启动而保留 |
| 子树复用在生产 CUDA 路径 | 目前只在小型 CPU 模型上量过（整局短 31%、每手评估 −2.9%）；CUDA 对照尚未做 |
| cuDNN autotune | 故意关闭（见 B6） |
| 原生 CPU 训练 | 设计上拒绝：`train --cpu` 直接报错退出 |

**约 29× 的归因。** 可以单独实测的部分是：精度（TF32/BF16 内核 2–4×，监督阶段端到端 1.77×）、批量 GPU 评估（一次 32 个叶子而不是 1 个；批大小扫描说明 batch=1 时 GPU 是延迟受限而非吞吐受限）、子树复用与共享评估缓存（每手网络调用更少）、战术短路（已定胜负的局面不推理），以及把 Python/GIL 开销移出热路径（C++ 单进程）。只有精度倍数和批大小扫描是隔离实测；其余是机制解释加上 2.5 小时的端到端结果。

---

## 网络与搜索

### 状态编码

四个平面，全部以「当前行棋方」视角编码（与 `Board::encode_state` 及 Python 工具一致）：己方棋子、对方棋子、最后一手、「是否黑方走」（常量平面）。网络因此看不到绝对颜色。

### 网络结构

```
state [4,15,15]
  stem      conv3x3 4 -> 192, BN, ReLU
  trunk     12 x [ conv3x3 192->192, BN, ReLU, conv3x3 192->192, BN, add, ReLU ]   （感受野 51x51）
  policy    conv1x1 192->2, BN, ReLU, flatten(450), fc 450->225, log_softmax
  value     conv1x1 192->1, BN, ReLU, flatten(225), fc 225->64, ReLU, fc 64->1, tanh
```

| 部件 | 参数量 |
|---|---:|
| stem 3×3 4→192 | 7,104 |
| 主干 12 块（占 98.5%） | 7,967,232 |
| 策略头（1×1 卷积 + 450→225 全连接） | 101,861 |
| 价值头（1×1 卷积 + 225→64 + 64→1） | 14,722 |
| **推理导出合计（BN 已折合）** | **8,090,919** |
| 训练检查点可学习参数（含 BN γ/β） | 8,100,525 |

用 `-DGOMOK_NET_CHANNELS=N -DGOMOK_NET_RES_BLOCKS=N` 可以重建其它形状，不同形状的检查点互不兼容。策略头是「1×1 卷积 + 450→225 稠密层」而不是旧的 900→225：旧层占全部参数的 62%，且没有跨棋盘位置的权重共享。

### 搜索

- PUCT/MCTS：`c_puct 5`、`--playout 400`（参考流水线训练时 256）、候选上限 60。
- 叶子按自身行棋方视角评估，回传时交替符号；终局按精确值处理。
- 一步必胜的叶子回传精确值且不推理；唯一必防点能存活到剪枝之后，并把展开收敛为单子节点。
- 虚拟损失只在同一棵树有多个叶子在飞时生效（串行的竞技场搜索会关闭它）。
- 根节点 Dirichlet 噪声（`ε 0.25`、`α 0.3`）在剪枝前对**所有**合法着法抽取；自对弈的策略标签用归一化的原始访问次数，温度只影响实际采样。
- 子树复用把已完成的访问与价值带到下一手；根噪声重抽，已访问的子节点保留，并补上当前的前 K 个候选。
- 竞技场用相同开局交换执色，并同时报告单局 W/L/D 与开局对结果 —— 自由五子棋先手优势明显，只看总分会被误导。

---

## 训练流程

一个 `train` 命令包含两个阶段：

1. **监督预训练**：默认只用人类棋谱的策略交叉熵。RenjuNet 的来源规则与本引擎的开局/长连规则不同，其胜负结果不能直接当作自由五子棋的价值标签，因此数据集元数据把 `value_weight` 设为 0，此时价值分支真正不进入计算图。
2. **自对弈强化学习**：批量自对弈产局，回放环与外部人类棋谱按 `--mix-ratio` 混合，每批执行 `--epochs` 次全新优化更新。

| 方面 | 实现 |
|---|---|
| 优化器 | Adam（β 0.9/0.999，ε 1e-8），`--weight-decay 1e-4` 做 L2，全局范数裁剪 2.0 |
| 学习率 | 预热 `base·(0.1+0.9·(step+1)/warmup)`，随后余弦衰减到 10% 下限；base 2e-3，RL 预热 2 批 / 监督预热 20 步 |
| 损失 | 策略交叉熵 + `value_loss_weight` × 加权价值 MSE（逐样本置信度按批内权重和归一化） |
| 批组成 | `--batch-size`（默认 512，参考流水线用 128），其中 `--mix-ratio` 比例是外部样本，每个样本现场抽一个 D4 变换 |
| 回放 | FIFO 环（参考流水线 `--buffer 20000`），落盘到 `trainer.replay` 以支持续训 |
| 精度 | `--precision bf16`（autocast）或 `fp32`；两者默认都允许 TF32 内核 |
| 检查点 | `best_policy_<tag>_bin`（最新）与 `best_policy_<tag>_val_bin`（监督验证最优）；`best` 前缀是历史命名，不代表最强 |
| 指标 | `runs/<tag>/metrics.csv`：固定训练/验证探针损失、价值误差、top-1、熵、价值权重 |
| 续训 | 校验批号、种子、选项/环境合约、数据集路径/大小/mtime、回放内容与 Adam 状态；保留已保存的学习率视野，延长 `--games` 后停在 LR 下限 |
| 早停 | 监督阶段 `--early-stop-patience`；patience 会随检查点恢复；不会自动回滚到验证最优检查点 |

日志不粉饰：`[train]` 分别打印 `selfplay_seconds` 与 `update_seconds`，`[progress]` 的 ETA 会把计划中的竞技场时间单独算进去。

**检查点格式与已知坏模型。** 训练器状态为 **v2** 格式（网络 + BN + Adam + 回放 + 调度视野 + 选项/数据合约）。早期 v1 实验状态在加载时会被拒绝，因为那版只优化策略的写法会把价值分支衰减到约 1e-37。已验收的最终检查点是 `best_policy_gomoku_fast_bin`：对已验收 RL 模型做零更新恢复再保存，往返之后 `.bin` 张量、回放缓冲与网络元数据逐字节一致。**不要**从 `best_policy_verify_learning_20260926_*` 继续训练 —— 那批检查点来自修复前的代码，只作为价值分支塌缩的证据保留。

---

## 数据集

`data/renjunet.xml`（RenjuNet，159,000 条记录）→ `data/gomoku_v2`：

| | |
|---|---|
| 接受的对局 / 局面 | 4,218 局 / 129,588 局面（来源规则 7/11/28 = Gomoku Pro / Swap2 / Swap；排除经典连珠） |
| 淘汰计数 | 其它规则 138,416；交换开局 6,020；不足 10 手 10,317；终局/结果冲突 18；终局后仍有落子 6；重复对局 5 |
| 策略标签 | 前 5 手不作为标签；外部样本做 0.1 的合法着法平滑 |
| 价值监督 | 关闭（`value_weight 0`），因为来源规则与本引擎不同 |
| 划分（种子 1，10%） | 116,781 训练 / 12,807 留出 / 619 因与训练集 D4 等价被剔除 / **12,188 独立验证** |
| 存储 | float32 `[N,4,15,15]` 状态、`[N,225]` 合法策略、`[N]` 结果（[−1,1]）、uint64 对局 ID、JSON 元数据；合计 557.6 MiB，mmap 读取 |

二进制数组以「小端 `uint64` 元素个数」开头。载入时会逐条复检（平面为 0/1 且有限、无重叠、最后一手归属正确、策略归一化且已占格概率为 0、黑白子数与行棋方一致、|z| ≤ 1、不含已终局局面）。没有对局 ID 的历史数据只能用 `--infer-game-groups 1` 分组，且要求完整的顺序对局 —— 随机按行划分无法修复混规则或截断的数据，会被直接拒绝。

---

## 强度评测

自由五子棋先手优势极大，因此本项目只报告配对开局结果，竞技场同时输出单局 W/L/D、得分、先手胜率和开局对战绩。

为本 README 新跑的测量（`build-training/`，种子 1，本地检查点）：

```sh
./build-training/gomoku_train arena --model best_policy_quality5000_20260926_bin \
  --baseline best_policy_fromzero_20260926_bin --games 80 --playout 400 \
  --opening-moves 4 --seed 1
# arena 58W 19L 3D score=0.74375 first_player_win_rate=0.688312 games=80 pairs=21W/1L/18D
```

更早用已验收 RL 模型对战「第 170 批旧模型的冻结推理导出」记录的结果，协议族相同：

```sh
./build-training/gomoku_train arena --model best_policy_verify_rl_20260926_bin \
  --baseline <第 170 批旧模型的导出目录> --games 40 --playout 400 \
  --opening-moves 4 --seed 1
# 40W 0L 0D，开局对 20W/0L/0D，先手胜率 0.500，耗时 42.1 s
```

价值质量用**新**自对弈局面检验，而不是训练数据：换一个未使用过的种子生成 64 局共 855 个局面，剔除与回放缓冲 D4 重叠的 167 个，对剩余 688 个逐局评估。按局平均 MSE：本模型 **0.1359**，最佳执色常数（乐观参考）0.2877，最佳全局常数 0.9939，恒输出 0 为 1.0000。

如何解读这些数字：

- 单种子、单协议、单对手 —— 不是 Elo，也不代表对所有开局的结论；
- 参数量变大、训练损失下降、测试通过、内核变快，任何一项都不足以证明棋力提高；
- 先手胜率明显偏离 50% 时，主导比分的是开局协议，而不是模型强弱。

---

## 部署：ONNX、浏览器与终端对弈

训练检查点保留原始卷积 + BN 张量，以便忠实续训；导出器读取这些权威张量并为推理折合 BN：

```sh
./build/gomoku_train export --model best_policy_quality5000_20260926_bin --output model_export
python3 tools/export_onnx.py --bin-dir model_export model.onnx     # 30.9 MiB，输出 log_policy [B,225] + value [B,1]

python3 tools/play_onnx.py   --model model.onnx --playout 400 --device cuda   # 终端对弈
python3 tools/server_onnx.py --model model.onnx --playout 400 --device cuda \
  --host 127.0.0.1 --port 8000                                                # 浏览器对弈
```

ONNX Runtime 不需要 PyTorch，但它的 CUDA provider 依赖 pip 把 CUDA/cuDNN 库装到的 `site-packages/nvidia`，那里不在动态库搜索路径上。因此引擎在创建 session 前会调用 `onnxruntime.preload_dlls()`；不做这一步，ONNX Runtime 只会打一条 warning 然后**静默退回 CPU**。现在 `--device cuda` 在 provider 加载失败时会直接报错，只有 `--device auto` 才允许回退。本机经网页服务实测：每手 400 次搜索，AI 落子约 0.5–0.8 s。

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
  train.{h,cpp}       Trainer：预训练、自对弈 RL、回放、指标、检查点与续训
  gradient.{h,cpp}    参考 CPU 梯度内核（由梯度测试使用）
  main.cpp            CLI：train | selfplay | arena | human | export
tools/          Python 工具链：数据转换与审计、ONNX 导出、学习对比基准、ONNX 对弈与网页服务
tests/          C++ 测试/基准可执行文件 + Python unittest 套件
data/           数据集前缀（大文件本地生成，不入库）
runs/           每个 tag 的 metrics.csv（本地生成，不入库）
```

## 测试与 CI

```sh
ctest --test-dir build --output-on-failure          # 本机 11/11，35 s
GOMOK_RUN_TRAINING_TESTS=1 GOMOK_TRAIN_BINARY=./build/gomoku_train \
  python3 -m unittest discover -s tests -p 'test_*.py'
```

覆盖面：棋盘规则与终局、战术搜索、价值符号、真实并发自对弈叶队列、合法策略标签、数据集分组/增强/D4 不变量、CPU↔CUDA 后端数值一致性、优化器与检查点恢复及写盘失败注入、CLI 错误处理、ONNX 导出数值一致性、Python 工具。CI 会用缩小主干构建 CPU 引擎并跑 ASan+UBSan，另有独立的生产形状编译任务。

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
| [`src/train.cpp`](src/train.cpp) | 预训练、自对弈 RL、回放、指标、续训合约 |
| [`src/main.cpp`](src/main.cpp) | `gomoku_train` 命令行入口 |
| [`tools/train_quality_5000.sh`](tools/train_quality_5000.sh) | 参考的 5000 局训练流水线 |
| [`tools/benchmark_learning.py`](tools/benchmark_learning.py) | FP32 与 BF16 的隔离监督对比驱动器 |
| [`tests/`](tests) | C++ 测试/基准目标与 Python unittest 套件 |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | CPU ASan/UBSan 任务与生产形状编译任务 |

## 口径、来源与限制

- **3 天基准的来源**：该数字是参考实现 `junxiaosong/AlphaZero_Gomoku`（PyTorch，15×15，同等自对弈 + 训练量）在**本机**的实测记录，保存在本仓库之外，因此无法由仓库产物重新推导。
- **2.5 小时的来源**：5000 局那次训练没有保留 stdout 日志。时长由产物时间戳与检查点状态推出：预训练 `18:19 → 18:28:59`，自对弈 `18:29:00 → 20:49:49`；检查点内的 `trainer.meta` 确认 6 轮监督（5,478 次更新）与第 250/250 批（6,000 次 RL 更新，回放 9,975/20,000）。重跑该脚本能复现协议，具体墙钟时间受机器与散热影响。
- 表中的数字来自本仓库可复现的命令，或本次在本机为 README 新做的测量；除特别说明外均为单次运行、单种子结果。
- 微基准不含自对弈与全量验证；端到端监督阶段数字包含二者。
- 基于旧网络、旧数据，或当时 TF32 并未真正关闭的早期测量，不会被搬进本 README；本地笔记与可复现命令冲突时，以命令为准。
- 训练按设计只支持 CUDA；CPU 构建提供推理、搜索、竞技场与测试。
- RenjuNet 的开局与长连规则不同于本引擎的自由规则，因此人类棋谱只监督**策略**，价值学习全部来自自对弈。

## 参考与致谢

- 本项目的参考实现：[junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)。
- AlphaZero 方法：Silver 等，*A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*（[arXiv:1712.01815](https://arxiv.org/abs/1712.01815)）。
- 训练数据：RenjuNet 棋谱（`data/renjunet.xml`）。
- 数值精度参考：[PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)。

本项目采用 [MIT 许可证](LICENSE) © 2026 LangLang03。
