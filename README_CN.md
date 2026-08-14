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
cpp/
├── src/            # 引擎：game、network、gradient、selfplay、train、logger
├── tools/          # 模型导出：pt -> .bin、pt -> onnx、npz -> bin/pt
├── tests/          # ctest 单元测试 + 基准工具
├── ui/             # Python 对弈界面：Web server + 终端对弈
└── build/          # cmake 产物、训练权重、检查点
```

## 编译

```sh
cd cpp
cmake -B build -DGOMOK_ENABLE_CUDA=OFF   # 纯 C++，无外部依赖
cmake --build build -j
ctest --test-dir build --output-on-failure
```

默认在检测到 Python PyTorch（`python3 -c "import torch"` 可执行）时启用 CUDA 后端；
用 `-DGOMOK_ENABLE_CUDA=OFF` 关闭。

## 使用

```sh
# 训练（79 批 = 每批 64 局 × 400 playout，CUDA）
GOMOK_MCTS_THREADS=8 ./build/gomoku_train train \
    --init cpp_model_v2 --games 79 --playout 400 --batch-games 64 \
    --check-freq 10 --threads 8

# 同上，纯 CPU（--int8 启用量化推理）
GOMOK_MCTS_THREADS=8 ./build/gomoku_train train \
    --init cpp_model_v2 --games 79 --playout 400 --batch-games 64 \
    --check-freq 10 --threads 8 --int8 --cpu

# 自对弈吞吐基准
./build/gomoku_train selfplay --model cpp_model_v2 --playout 400 \
    --games 64 --batch 64 --threads 8

# 终端对战
./build/gomoku_train human --model cpp_model_v2 --playout 400

# Web UI（浏览器，15×15，PyTorch 模型）
cd ui && python3 server.py --port 8000

# 终端对弈（8×8，纯 numpy，无需 torch）
cd ui && python3 human_play.py
```

### 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `GOMOK_MCTS_THREADS` | min(8, 核数) | MCTS 树 worker 线程数 |
| `GOMOK_BATCH_WAIT_US` | 100 | 攒推理 batch 的最大等待时间（µs） |
| `GOMOK_LEAF_BATCH` | min(playout,128) | 每棵树提交前收集的叶子数 |
| `GOMOK_CACHE_ENTRIES` | 32768 | 评估缓存容量 |
| `GOMOK_PROFILE` | 关 | 打印每批的 eval/调用/缓存统计 |
| `GOMOK_SEED` | 时间 | 随机种子 |

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

## 性能（8 核笔记本 + RTX 5060 Laptop 实测）

| 负载 | CUDA | CPU fp32 | CPU int8 |
|---|---|---|---|
| 自对弈 64 局 × 400 playout | ~6.6 s | ~35 s | ~27 s |
| 训练 B=512 × 5 epochs | 0.08 s | 3.3 s | — |
| 单发推理 batch 1 | 0.12 ms | 0.79 ms | 0.79 ms |
| 单发推理 batch 512 | 0.008 ms/局 | 0.17 ms/局 | 0.34 ms/局 |

## 数据

- `cpp/mix_data*` — 20 万条混合训练样本（自对弈 + 棋谱），训练时用 `--mix` 加载。
- `data/` — renju-net 风格棋谱数据（`.npz`）。

## 许可证

MIT。`cpp/ui/` 下的 Python 对弈界面源自
[Junxiao Song 的 AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)
（MIT）。
