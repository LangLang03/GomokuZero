# 问题分析 3：Sanitizer 与并发实测

工具驱动，非静态推断。原始日志在 `analysis/tsan/*.log`
（被 `.gitignore` 忽略，本文件保留结论与复现命令）。

环境：GCC 16.2.1。Sanitizer 由 `-DGOMOK_SANITIZER=<mode>` 开启。
TSan 与 CUDA 运行时不兼容，选中 `thread` 时 CMake 自动关闭 LibTorch 后端。

```sh
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DGOMOK_ENABLE_CUDA=OFF -DGOMOK_SANITIZER=address,undefined -DBUILD_TESTING=ON
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DGOMOK_ENABLE_CUDA=OFF -DGOMOK_SANITIZER=thread -DBUILD_TESTING=ON
```

### 为什么需要一个缩小的网络

TSan 对每次内存访问插桩。当前主干是 25 个卷积、8.1M 参数，而 MCTS 在没有
CUDA 时是**单线程前向**，因此一次自对弈在 TSan 下需要数十分钟到数小时。
并发逻辑（MCTS 选择、ready/completion 队列、节点竞技场、评估缓存）
**与网络形状无关**，所以扫描时把主干缩小：

```sh
-DGOMOK_SANITIZER_CHANNELS=64 -DGOMOK_SANITIZER_RES_BLOCKS=4
```

这两个开关只影响 sanitizer 构建，默认仍是 192 / 12。

---

## 1. 结论速览

| 扫描 | 配置 | 结果 |
|---|---|---|
| **ASan + UBSan** | `ctest`（含自对弈全流程） | **零报告** |
| **ASan + UBSan** | 多线程自对弈 16 局 / 8 MCTS 线程 | **零报告** |
| **TSan** | 自对弈 8 局 / 8 MCTS 线程 | **零报告** |
| **TSan** | 自对弈 16 局 / **16** MCTS 线程 | **零报告** |
| **TSan** | 训练全流程（自对弈 + 反向 + Adam） | **零报告** |
| **TSan** | 全部 `ctest`（含 parity 与 regression） | **零报告** |

**这推翻了一个早期推测。** 最初判断"竞态导致性能下降"，
但 TSan 在 16 线程压力下没有报出任何数据竞争。

正确结论：

- **不存在数据竞争**（内存模型层面正确）。
- 所谓"竞态导致的性能下降"实际是**逻辑层面的线程碰撞**：
  多个 MCTS 线程选中同一个未展开子节点，碰撞后走 `Retry` 路径并 `yield()`，
  白耗 playout 预算。这是**算法/性能问题，不是正确性问题**，
  因此 TSan 按定义不会报它。

这一类问题必须用 profile 数字而非 sanitizer 证明。

---

## 2. 扫描中发现并修复的真实 bug

### 2.1 `fast_fc` 对 value 分支用了硬 ReLU（旧 3 层网络）

`gomoku_smoke` 的 "fast inference consistency" 在 ASan 构建下失败：

```
max_policy_error = 4.768e-07    <-- 纯浮点噪声量级
value_error      = 4.649e-04    <-- 超容差（2e-4）1000 倍
```

| 路径 | 代码 | 激活 |
|---|---|---|
| fast | `fast_fc(val_fc1_, ws.va, ws.vh, true)` | 硬 **ReLU** |
| slow | `fc_forward(...); leaky_relu_forward(ws.vh);` | **LeakyReLU** |

commit `f8f3179`（"Fix dead value head by using LeakyReLU"）把 value 分支改成
LeakyReLU 以防止神经元死亡，但 AVX fast 路径没跟上，**在默认路径上把这个修复
抵消了**。修复后 `ctest` 全绿。该代码在架构重写时已被替换。

### 2.2 折合幂等性（架构重写期间发现）

加载时曾把 `running_var` 设为 `1`，而折合要除以 `sqrt(var + eps)`，
于是每次 保存→加载 都把权重缩放一次 `1/sqrt(1+eps)`。
12 层残差累积后 log 空间偏差达 1.5e-03。

改为 `running_var = 1 - eps`，使折合系数恰好为 1，往返现在**逐位精确**
（`gomoku_backend_parity` 实测 `max log error 0.000e+00`）。

### 2.3 Arena 评估了错误的局面（由外部审查发现，见 `04_...md`）

`ArenaSearch::run()` 编码的是 `game.board`（始终是根局面），而应当编码
`task.board`（叶局面）。这不是竞态，sanitizer 看不见——它需要**语义测试**。
现已由 `gomoku_regression` 覆盖。

**这说明 sanitizer 零报告不等于代码正确**：它只保证内存模型与内存安全，
不保证算法语义。

---

## 3. 线程碰撞：profile 证据（性能问题的正确度量方式）

TSan 零报告说明这不是正确性问题，但仍需量化浪费。
`GOMOK_PROFILE=1` 输出每批评估统计（12 层 × 192 通道，CUDA）：

```
batch 目标 128（原常量，来自旧网络）
  evals=501216 calls=4468  avg_batch=112.2 max_batch=128 cache_hits=69757 (13.9%)
  7.30 s/game

batch 目标 32（重新实测的最优点）
  evals=500884 calls=16979 avg_batch=29.5  max_batch=32  cache_hits=60364 (12.1%)
  6.26 s/game
```

### 已实施的修复

| 改动 | 依据 |
|---|---|
| 虚拟损失权重 1 → **3**，并修正符号 | 原实现给子节点 `value_sum` 加**正值**（`+ virtual_visits * VALUE_SCALE`），使被占用的子节点**更有吸引力**，方向反了；8 线程反复挤入同一分支 |
| 虚拟损失从 Q 计算中解耦 | 不再在读取 Q 时叠加虚拟值，改为原子地加减 `value_sum` 并在 `backup_path` 中补偿，避免读-改-写竞争 |
| CUDA batch 目标 128 → 32 | 对当前主干扫描 16…512，32 达 2846 eval/s，128 仅 2588 |

**重要**：这些修复基于**代码语义分析 + profile 数字**，不是 sanitizer 报告。
`virtual_visits` 加减不匹配、符号反向、最优 batch 这类问题 TSan 看不见。

cache 命中率从旧架构的 10.3% 提升到 12–14%：虚拟损失修正后更少 playout
被浪费在重试上，更多落在已评估节点。

### 仍未处理

- **MCTS 子树复用**：每步 `reset_tree` 丢弃整棵树，400 playout 从零重建。
  这是当前最大的性能项（外部审查 P1-02 / P1-03）。需要处理竞技场
  `child_begin` 重索引，属独立结构改动。
- **推理搬运优化**：无 pinned memory / 异步拷贝（外部审查 P1-08）。
  报告本身也要求先由 profile 证明搬运占比显著。
- **每批自对弈使用同一 `GOMOK_SEED` 推导的 rng 序列**，跨批多样性受限。

---

## 4. 方法论备注

本次扫描中有两次"疑似竞态"最终查明是我自己的操作错误：

1. `pkill -f gomoku_train` 匹配到了执行命令的 shell 自身（命令行含该字符串），
   把自己杀掉，表现为 `exit=1` 且无输出。
2. TSan 自对弈命令漏了 `--model`，`net.load("")` 失败返回 1，
   同样表现为"启动后立即退出"。

两次都不是程序缺陷。**报告"竞态"前必须先排除调用错误**——
这与外部审查报告第 5 节"未经 sanitizer 实测不写成已确认竞态"的原则一致。
