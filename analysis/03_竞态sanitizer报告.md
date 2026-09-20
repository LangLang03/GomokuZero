# 问题分析 3：Sanitizer 与并发实测

工具驱动，非静态推断。原始日志在 `analysis/tsan/*.log`（被 `.gitignore` 忽略，
本文件保留结论与复现命令）。

环境：GCC 16.2.1，TSan/ASan/UBSan 由 `-DGOMOK_SANITIZER=<mode>` 开启。
TSan 与 CUDA 运行时不兼容，选中 thread 时 CMake 自动关闭 LibTorch 后端
（`CMakeLists.txt`）。

```sh
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DGOMOK_ENABLE_CUDA=OFF -DGOMOK_SANITIZER=address,undefined -DBUILD_TESTING=ON
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DGOMOK_ENABLE_CUDA=OFF -DGOMOK_SANITIZER=thread -DBUILD_TESTING=ON
```

---

## 1. 结论速览

| 扫描 | 配置 | 结果 |
|---|---|---|
| **ASan + UBSan** | `ctest`（含自对弈全流程） | **零报告** |
| **ASan + UBSan** | 多线程自对弈 16 局 / 8 MCTS 线程 | **零报告** |
| **TSan** | 自对弈 8 局 / 8 MCTS 线程 | **零报告** |
| **TSan** | 自对弈 16 局 / **16** MCTS 线程 | **零报告** |
| **TSan** | 训练全流程（自对弈 + 反向 + Adam） | **零报告** |

**这推翻了我此前的一个推测。** 我原先判断"竞态导致性能下降"，
但 TSan 在 16 线程压力下没有报出任何数据竞争。

正确的结论是：

- **不存在数据竞争**（内存模型层面是正确的）。
- 所谓"竞态导致的性能下降"实际上是**逻辑层面的线程碰撞**：
  多个 MCTS 线程选中同一个未展开子节点，碰撞后走 `Retry` 路径
  并 `yield()`，白耗 playout 预算。这是**算法/性能问题，不是正确性问题**，
  因此 TSan 按定义不会报它。

这一类问题必须用 profile 数字而非 sanitizer 来证明（见第 3 节）。

---

## 2. 扫描中发现并修复的真实 bug（非竞态）

**`fast_fc` 对 value 分支用了硬 ReLU，而 slow 路径用 LeakyReLU。**

`gomoku_smoke` 的 "fast inference consistency" 在 ASan 构建下失败。
实测误差：

```
max_policy_error = 4.768e-07    <-- 纯浮点噪声量级
value_error      = 4.649e-04    <-- 超容差（2e-4）1000 倍
```

定位到两处激活函数不一致：

| 路径 | 代码 | 激活 |
|---|---|---|
| fast | `network.cpp:804` `fast_fc(val_fc1_, ws.va, ws.vh, true)` | 硬 **ReLU**（`std::max(0.0f, …)`） |
| slow | `network.cpp:809` `fc_forward(...); leaky_relu_forward(ws.vh);` | **LeakyReLU**（负值 ×0.01） |

最小复现（权重设 −1、输入 +0.1）：

```
slow fc_forward 原始输出: -0.4000 -0.4000
slow + leaky_relu_forward:  -0.0040 -0.0040   <-- 负值保留 0.01 倍
fast_fc(relu=true):          0.0000  0.0000   <-- 负值被清零
```

这不是纯数值噪声：commit `f8f3179`（"Fix dead value head by using
LeakyReLU in value branch"）刻意把 value 分支改成 LeakyReLU 来防止
神经元死亡，但 AVX fast 路径没跟上，**在 fast 路径上把这个修复抵消了**。
而 fast 路径正是默认路径（`fast_inference_enabled_ = true`）。

修复：
- `src/gradient.h` 导出共享常量 `LEAKY_RELU_ALPHA = 0.01f`
- `fast_fc` 的 `relu` 参数改为 `leaky`，按斜率实现 LeakyReLU
- `fast_conv1x1_relu` / `fast_conv3x3_relu` 保持不变（两路径都是显式 ReLU，
  且 `va` 作为 ReLU 输出非负，后再接 LeakyReLU 等价）

修复后 `ctest` 全绿。

---

## 3. 线程碰撞：profile 证据（性能问题的正确度量方式）

TSan 零报告说明这不是正确性问题，但仍需量化浪费。
`GOMOK_PROFILE=1` 输出每批的评估统计：

```
[selfplay-profile] evals=23658 calls=258 avg_batch=91.7  max_batch=312 cache_hits=2430  cache_size=23658
[selfplay-profile] evals=19202 calls=119 avg_batch=161.4 max_batch=312 cache_hits=1886  cache_size=19202
```

- `cache_hits / evals` = 2430/23658 ≈ **10.3%**：评估缓存命中率偏低。
- 缓存是**每批新建**的（`run_batch` 内局部变量），不跨手复用。
  400 playout × 80 手 = 32000 次评估里，大量节点在后续手中被重新评估。
- 每手 `reset_tree`（`selfplay.cpp`）**丢弃整棵树**，MCTS 不复用子树。

### 已实施的修复

| 改动 | 依据 | 位置 |
|---|---|---|
| 虚拟损失权重 1 → **3**，并修正符号 | 原实现给子节点 `value_sum` 加 **正值**（`+ virtual_visits * VALUE_SCALE`），使被占用的子节点 **更有吸引力**，方向反了；8 线程反复挤入同一分支 | `VirtualLossWeights`，`selfplay.cpp` |
| 虚拟损失从 Q 计算中解耦 | 原子地加减 `value_sum`，在 `backup_path` 中补偿，避免读-改-写竞争 | `select_leaf` / `backup_path` |

**重要**：修复项是基于**代码语义分析 + profile 数字**，不是 sanitizer 报告。
`virtual_visits` 加/减不匹配、符号反向这类问题 TSan 看不见。
后续会用修改前后的 `cache_hits/evals` 与 `s/game` 对比来验证收益。

### 仍未处理

- MCTS 子树复用（跨手保留选中分支）——需要处理 `child_begin` 重索引
- 每批自对弈使用同一 `GOMOK_SEED` 推导的 rng 序列，跨批多样性受限

---

## 4. 附带的死代码发现

`analysis/` 与本次扫描顺带确认（详见 `02_算法错误分析.md` 第 7 节）：

- `src/inference.cpp` 的 `load_bin` 用 `uint64_t count` 读取 `uint32_t` 头的
  模型文件，一旦参与编译会尝试分配约 3 TB。当前 `CMakeLists.txt:41-49`
  不含该文件，属**未编译**的死代码，建议删除。
- `tests/bench_forward.cpp`、`tests/bench_torch.cpp`、`tests/bench_train.cpp`、
  `tests/dump_forward.cpp` 同样不在构建目标内。
