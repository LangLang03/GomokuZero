# 五子棋 AlphaZero（C++）

15×15 自由五子棋：双方交替落子，连成五子或更多即获胜，没有连珠禁手或交换开局。C++17 引擎提供棋盘规则、PUCT/MCTS、批量自对弈和原生 CPU 推理；训练使用可选的 LibTorch/CUDA 后端。

当前工作重点是修正数据、搜索、训练和恢复语义，并提供可复现的测量。**通过测试、训练损失下降或参数量增加，都不能证明棋力已经提高。** 新模型的泛化和棋力需要独立验证集与配对开局对战验证；本 README 不沿用旧网络的速度和棋力结论。

## 编译与测试

需要 CMake、C++17 编译器和线程库。CUDA 训练还需要可用的 NVIDIA GPU，以及当前 Python 环境中带 CUDA 的 PyTorch；CMake 从该环境读取 LibTorch 路径与 C++ ABI。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGOMOK_ENABLE_CUDA=ON -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

没有 CUDA 时可编译原生 CPU 版本用于对弈、搜索和测试，**原生 CPU 后端不支持训练**：

```sh
cmake -S . -B build-cpu -DGOMOK_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu -j2
```

输入四个平面依次表示己方棋子、对方棋子、上一手位置、黑方是否执棋。网络经过 3×3 卷积 stem 和残差主干，再分出策略与价值两个头；全连接头会混合整个棋盘的信息，不能仅凭单个卷积特征的感受野判断模型无法学习某种战术。

默认网络为 12 个残差块、192 通道。`-DGOMOK_NET_CHANNELS=N -DGOMOK_NET_RES_BLOCKS=N` 可构建其他规模；不同网络形状的检查点不能互换。较小模型可能更快，但需单独测量其泛化和棋力。

快速检查并发与内存可使用小网络构建；这不代替完整网络的数值与性能测试：

```sh
cmake -S . -B build-tsan -DGOMOK_ENABLE_CUDA=OFF \
  -DGOMOK_NET_CHANNELS=8 -DGOMOK_NET_RES_BLOCKS=1 \
  -DGOMOK_SANITIZER=thread -DBUILD_TESTING=ON
cmake --build build-tsan -j2
ctest --test-dir build-tsan --output-on-failure
# 内存/未定义行为检查：另一个构建目录使用 -DGOMOK_SANITIZER=address,undefined
```

测试覆盖棋盘规则、搜索价值符号与战术、真实并发自对弈、合法策略目标、数据分组和增强、后端数值、优化器恢复及 Python 工具。CUDA 专用验证需要实际启用 CUDA；CPU 构建通过不代表 CUDA 路径已被验证。

## 准备与检查数据

推荐重新转换原始棋谱，保存完整对局 ID 和规则元数据：

```sh
python3 tools/parse_renjunet.py data/renjunet.xml --bin-prefix data/gomoku_v2
python3 tools/audit_dataset.py data/gomoku_v2 --state-overlap \
  --output analysis/gomoku_v2_dataset_audit.json
# 已有符合约定的 NPZ 也可转换：
# python3 tools/npz_to_bin.py dataset.npz data/custom
```

Python 数据工具需要 NumPy。训练读取以下文件：

| 文件 | 内容 |
|---|---|
| `<prefix>_states.bin` | 当前执子方视角的 `[N,4,15,15]` float32 状态 |
| `<prefix>_probs.bin` | `[N,225]` 合法策略分布 |
| `<prefix>_winners.bin` | `[N]` 当前执子方视角结果，范围 `[-1,1]` |
| `<prefix>_game_ids.bin` | `[N]` uint64 对局 ID |
| `<prefix>_meta.json` | 来源规则、目标规则、价值监督权重等元数据 |

二进制数组以 uint64 元素数量开头，使用小端存储。加载器通过 mmap 读取，检查形状、落子合法性、状态通道、策略归一化、终局和同局价值符号。

默认转换器选取来源规则 `7/11/28`，排除经典连珠、显式交换开局、非法棋谱和完整对局的 D4 对称重复；前五手不作为策略训练目标。来源开局/长连规则仍与本引擎不完全一致，**这些棋谱默认只监督策略，价值权重为 0**。不要把人类记录的胜负直接当成本引擎规则下可靠的价值标签。未知元数据的旧数据同样默认关闭价值监督。

验证集按完整对局划分，再移除与训练集状态 D4 等价的验证样本。缺少对局 ID 时，优先重新生成数据；`--infer-game-groups 1` 仅适用于保持完整对局顺序的旧记录。乱序、截断、规则混杂的数据不能靠随机逐行切分获得可靠验证。`--val-fraction 0` 关闭独立验证，也无法据此判断泛化。

## 训练与恢复

查看完整选项：`./build/gomoku_train --help`。未知、重复或缺少值的选项会报错；指定模型加载失败也会报错。

先进行监督预训练：

```sh
./build/gomoku_train train --mix data/gomoku_v2 \
  --pretrain 5 --games 0 --early-stop-patience 3 --seed 1 --tag pretrain
```

每个监督 epoch 将训练分区随机打乱并完整遍历一次，包含最后不足一个 batch 的样本；采用在线 D4 增强。每个 epoch 评估固定的训练/验证探针并保存检查点。早停后保留停止状态，恢复时不会自动绕过已耗尽的 patience。

从验证表现最好的预训练检查点开始自对弈训练：

```sh
./build/gomoku_train train --init best_policy_pretrain_val_bin \
  --mix data/gomoku_v2 --pretrain 0 --games 100 \
  --early-stop-patience 3 --seed 1 --tag rl
```

继续该训练到第 150 批：

```sh
./build/gomoku_train train --init best_policy_rl_bin \
  --mix data/gomoku_v2 --pretrain 0 --games 150 \
  --early-stop-patience 3 --seed 1 --tag rl
```

`train --games` 是**自对弈批次的绝对终点**，不是追加批数或对局数。全新运行 `--games 100 --batch-games 64` 生成 6400 局；从第 100 批恢复到 150 批只生成后 50 批。`--pretrain` 同样是监督 epoch 的绝对终点；`--epochs` 则表示每批自对弈后的优化次数，每次重新采样 minibatch。

恢复会加载网络、BN、Adam 状态、回放缓冲区、批次、监督进度和学习率调度，并检查训练配置及数据路径/大小/修改时间是否一致。保留原训练选项、种子和运行环境；并发调度不保证逐位相同。已有自对弈调度扩展到更远终点时，原调度结束后保持学习率下限；纯预训练检查点首次进入自对弈时建立新的自对弈调度。旧检查点若缺少 trainer 状态，会明确提示回放为空，不能视为完整恢复。 首次从仅监督训练进入自对弈时，可以设置搜索预算、并发局数、回放和混合比例；开始自对弈后再恢复时会检查这些参数。当前 trainer 状态格式为 v2，本次早期实验的 v1 状态会被拒绝，避免继续使用价值分支已塌缩的模型。

| 输出 | 含义 |
|---|---|
| `best_policy_<tag>_bin/` | **最近一次**完整训练检查点；名字中的 `best` 不代表棋力最强 |
| `best_policy_<tag>_val_bin/` | 监督阶段验证目标改善时保存的检查点；不是竞技场棋力冠军 |
| `runs/<tag>/metrics.csv` | 固定训练/验证探针的策略损失、价值误差、top-1、熵 |

早停不会自动把当前网络回退到验证最佳版本；要从它继续，请显式传入对应 `_val_bin`。只有自对弈训练或关闭验证的运行不保证生成 `_val_bin`。

常用训练参数：

| 参数 | 默认值 | 语义 |
|---|---:|---|
| `--batch-size` / `--epochs` / `--buffer` | 512 / 5 / 10000 | 优化 batch、每批自对弈的更新次数、回放容量 |
| `--lr` / `--lr-final-fraction` | 0.002 / 0.1 | 基础学习率与余弦衰减下限比例 |
| `--lr-warmup` / `--pretrain-warmup` | 2 / 20 | 自对弈预热批数 / 监督预热步数 |
| `--weight-decay` / `--grad-clip` | 0.0001 / 2 | Adam L2 正则 / 梯度范数上限 |
| `--mix-ratio` / `--mix-value-weight` | 0.5 / 元数据 | 外部样本比例 / 外部价值样本相对权重；0 禁用其价值监督 |
| `--value-loss-weight` / `--label-smooth` | 1 / 0.1 | 整体价值损失系数 / 外部策略合法动作平滑 |
| `--augment` | 1 | 在线八种旋转/反射增强 |
| `--precision` | `fp32` | `bf16` 需显式启用，并单独检查数值与吞吐 |
| `--val-fraction` / `--val-samples` | 0.1 / 2048 | 按对局留出比例 / 验证探针上限，0 样本上限表示全部 |
| `--val-freq` / `--kl-freq` | 5 / 5 | 自对弈阶段验证 / 观察性 KL 的批次间隔；0 关闭对应探针 |
| `--check-freq` | 50 | 自对弈检查点间隔；结束时也保存 |

每完成一批自对弈会输出 elapsed 和 ETA。ETA按近期批次耗时平滑估算，并把尚未进行的定期竞技场对局单独计入；启动初期样本较少，估算会随着批次完成逐步稳定。纯监督预训练则在进度日志中显示估算剩余时间。

价值样本权重按批内权重总和归一化；它表示样本间的相对置信度，不替代整体 `--value-loss-weight`。

`--precision fp32` 表示训练张量使用 FP32；默认仍允许 CUDA 卷积/矩阵乘法使用 TF32。需要禁用 TF32 时设置 `GOMOK_TF32=0`。`--precision bf16` 显式启用训练混合精度，不能据此预先保证更快或学习效果不变。

## 搜索、对战与评估

```sh
# 批量自对弈，只生成对局并统计吞吐，不更新模型
GOMOK_PROFILE=1 ./build/gomoku_train selfplay \
  --model best_policy_rl_bin --games 8 --batch 8 --playout 400 --seed 1

# 同一开局交换双方执色，比较两个固定模型
./build/gomoku_train arena --model best_policy_rl_bin \
  --baseline best_policy_pretrain_val_bin --games 40 --playout 400 \
  --opening-moves 4 --seed 1

./build/gomoku_train human --model best_policy_rl_bin --playout 400
```

`selfplay --games` 和 `arena --games` 计数的是**对局数**；arena 要求正偶数。终端输入 `r,c`，行列均为 `0..14`。这些命令支持 `--cpu`，但完整默认网络的原生 CPU 推理可能较慢。

MCTS 使用叶子当前执子方价值、交替符号回传及精确终局判断；一步必胜直接使用精确价值，必防动作不受候选上限排除。并发同树搜索才使用虚拟损失，碰撞重试不消耗有效模拟预算。根噪声在候选裁剪前应用。自对弈策略目标是归一化原始访问次数，温度仅控制实际落子；单次模拟只有根先验时以该先验回退。候选截断和有限搜索预算仍是近似，不能保证找到所有远期战术。

自对弈落子后会压缩所选分支并提升为新根，保留已完成的访问与价值；下步只补足剩余的模拟次数。新根会在全部合法着法上重新计算先验和 Dirichlet 噪声，保留已访问的子节点，再加入本次排名靠前的候选。树复用默认开启；设置 `GOMOK_TREE_REUSE=0` 可关闭，以便做同配置对照。新保存的训练检查点会记录这一设置，旧检查点恢复时会提示采用的新设置。

`GOMOK_PERSISTENT_WORKERS=1` 可让 MCTS 搜索线程在一次自对弈 batch 的各步之间常驻，复用线程及临时缓冲区；RTX 5060 实测未见收益，因此默认关闭。CUDA 推理可通过 `GOMOK_PINNED_INFERENCE=1` 试用复用的锁页主机缓冲区和非阻塞拷贝；推理接口每批都需立即返回结果，因此该路径仍会在读取结果前同步，RTX 5060 实测也未见收益，默认关闭。

| 搜索参数 | 默认值 |
|---|---:|
| `--playout` / `--batch-games`（训练） | 400 / 64 |
| `--c-puct` / `--max-candidates` | 5 / 60 |
| `--temp-high` / `--temp-low` / `--temp-switch` | 1 / 0.5 / 30 |
| `--dirichlet-eps` / `--dirichlet-alpha` | 0.25 / 0.3 |
| `--virtual-loss` / `--unseen-smooth` | 8 / 0 |
| `--inference-batch` | 0，自动目标 32 |
| `--opening-moves` | 4，允许 0..8；0 使用空棋盘 |
| `--seed` | `GOMOK_SEED`，未设置时为 1 |

`--unseen-smooth` 是可选的合法动作均匀平滑。arena/人机对战关闭根噪声并取最大访问动作。训练期间可用 `--eval-games N --eval-freq N --eval-playout N --baseline DIR` 周期对战，默认关闭；baseline 必须保持固定且不能是该运行正在写入的检查点。

arena 同时报告单局 W/L/D 与开局对的 `pair_wins/pair_losses/pair_draws`：同开局两局候选方总分大于 1 算对胜，小于 1 算对负，等于 1 算对平。检验应以开局对为单位；小样本仍不足以建立稳定棋力结论。

自由五子棋存在先手偏置。配对开局、先手胜率和足够的对战样本有助于识别它，但少量对局的 50% 或 100% 比分都不能单独证明模型相同或更强。训练损失低、验证损失恶化及泛化差距持续扩大是过拟合线索；训练和验证均无改善也可能来自数据、学习率或预算不足，不能仅由模型规模判断欠拟合。

## 导出、性能与文件布局

```sh
./build/gomoku_train export --model best_policy_rl_bin --output model_export
python3 tools/export_onnx.py --bin-dir model_export model.onnx
# ONNX 导出需要 PyTorch、onnx 和 onnxruntime
```

在带 NVIDIA GPU 的主机上启动 ONNX 网页服务：

```sh
python3 tools/export_onnx.py --bin-dir best_policy_fromzero_20260926_bin build-training/fromzero_20260926.onnx
python3 -c 'import onnxruntime as ort; print(ort.get_available_providers()); assert "CUDAExecutionProvider" in ort.get_available_providers()'
python3 tools/server_onnx.py --model build-training/fromzero_20260926.onnx --playout 400 --device cuda --host 127.0.0.1 --port 8000
```

服务会打印实际选择的 ONNX Runtime provider，浏览器访问 `http://127.0.0.1:8000`。显式指定 `--device cuda` 时，如果 CUDA provider 或设备不可用，服务会报错而不会退回 CPU；`--device auto` 才会在 CUDA 初始化失败时使用 CPU。

训练检查点保存未折合的网络/BN、优化器和 trainer 状态；推理导出保存折合 BN 的卷积权重。推理导出能用来对弈或作为权重起点，无法忠实恢复原优化器和训练过程。ONNX 输入为 `[B,4,15,15]`，输出为 log-policy `[B,225]` 和 value `[B,1]`。

`./build/gomoku_diag_capacity <模型目录> [状态数量]` 报告实际加载网络的形状与随机合法棋盘输出统计。这些统计用于检查输出异常，不能替代真实棋谱验证或证明过拟合/欠拟合。

性能需在同一模型、网络形状、设备、精度、batch 和搜索预算下测量。`gomoku_backend_benchmark [batch] [steps] [模型目录]` 测量预热后的推理/训练吞吐；自对弈需同时记录耗时、有效网络评估数、缓存命中、手数和对局数。微基准提速不能直接等同于端到端提速或学习更快。

一次真实数据对比已完成：同一 192×12 网络、seed、batch 128、3 个完整监督 epoch，并评估全部 12188 个按对局留出且剔除训练侧 D4 重复的验证局面。结果和完整命令见[训练重构与验收报告](analysis/07_训练重构与验收.md)。

| 指标 | FP32（允许 TF32） | BF16 |
|---|---:|---:|
| 监督阶段总耗时，含验证和保存 | 460.418 s | 260.498 s |
| 最终验证策略交叉熵 | 2.010746 | 2.010494 |
| 最终验证 top-1 | 44.7407% | 44.8146% |

此实验的 BF16 监督阶段约快 **1.77×**，同一种子的短期策略质量初步相当；这不证明长期收敛、棋力或完整自对弈训练同幅度提升。

**该实验版本同时暴露出价值分支的 L2 衰减塌缩，实验检查点 `best_policy_verify_learning_20260926_*` 不应作为最终模型或继续自对弈训练的起点。** 最终修复已验证：无价值监督时跳过该分支的梯度、正则与动量更新；新模型的价值输出层从零开始，启用价值监督后使用独立的 Adam 计步。修复后的模型重新预训练并完成160局自对弈，以400次搜索、20组交换执色开局对原第170批模型取得40胜0负。新种子64局剔除D4回放重叠后，按局平均价值MSE为0.1359，优于用评估标签拟合的执色常数参考0.2877。这些是具体协议下的实测，不代表通用Elo或所有对手的结果。

| 环境变量 | 用途 |
|---|---|
| `GOMOK_PROFILE` | 输出自对弈评估、缓存、重试和推理 batch 统计 |
| `GOMOK_MCTS_THREADS` | 覆盖搜索 worker 数，最终限制为 1..32 |
| `GOMOK_LEAF_BATCH` | 覆盖每局最大在途叶子数，限制在 1..playout |
| `GOMOK_TREE_REUSE` | 自对弈子树复用，默认开启；设为 0 关闭 |
| `GOMOK_PERSISTENT_WORKERS` | 自对弈搜索线程与缓冲区跨落子复用，默认 0；设为 1 开启 |
| `GOMOK_BATCH_WAIT_US` | 推理收集等待时间；默认 CUDA 100µs、CPU 0µs |
| `GOMOK_CACHE_ENTRIES` | 单批自对弈的推理缓存容量，默认 32768 |
| `GOMOK_SEED` | 随机种子；CLI `--seed` 优先 |
| `GOMOK_TF32` | 训练中的 TF32 卷积/矩阵乘法开关，默认 1 |
| `GOMOK_INFERENCE_TF32` | 推理中的 TF32 开关，默认 1；设为 0 禁用 |
| `GOMOK_PINNED_INFERENCE` | 试验性 CUDA 推理锁页缓冲区与非阻塞数据拷贝，默认 0 |
| `GOMOK_FUSED_ADAM` | 融合 Adam 更新开关，默认 1 |
| `GOMOK_BENCH_BF16` | 仅供后端基准程序显式选择 BF16，设为 1 启用 |

源码在 `src/`，测试在 `tests/`，转换和导出工具在 `tools/`，审计报告在 `analysis/`。数据和训练产物在本地生成；大型二进制数据、检查点和 `runs/` 不应纳入源码版本控制。

## 本轮验证模型

完整模型：`best_policy_gomoku_fast_bin`，包含网络、Adam、BN、2857条回放及训练进度。它经真实恢复/重新保存验证，与通过对战的模型张量及回放逐字节一致；模型指纹在 `analysis/final_model_20260926.json`。

```sh
# 使用本轮构建和模型对弈
./build-training/gomoku_train human --model best_policy_gomoku_fast_bin --playout 400

# 从第10批继续到第20批；保持已保存的学习率调度
./build-training/gomoku_train train --init best_policy_gomoku_fast_bin \
  --mix data/gomoku_v2 --pretrain 0 --games 20 --tag gomoku_fast \
  --batch-size 128 --precision bf16 --seed 1 --val-samples 0 \
  --playout 64 --batch-games 16 --epochs 20

# 独立的新棋局价值诊断（不更新模型）
./build-training/gomoku_value_evaluation best_policy_gomoku_fast_bin 64 64 20260927
```

已验证的ONNX在 `build-experiment-artifact/gomoku_verified_20260926.onnx`。完整实验与限制见 [验收报告](analysis/07_训练重构与验收.md)。继续训练后应重新评估并重新导出ONNX。
