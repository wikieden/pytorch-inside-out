# PyTorch 从里到外

**在线阅读：<https://wikieden.github.io/pytorch-inside-out/>** · [讲义目录](https://wikieden.github.io/pytorch-inside-out/lessons/index.html) · [全部动画](https://wikieden.github.io/pytorch-inside-out/animations/index.html) · [复习卡片](https://wikieden.github.io/pytorch-inside-out/review.html) · [全部讲义书](https://wikieden.github.io/ai-books/)

> 36 章 · 36 个动画（GIF + 可交互页面）· 135 张复习卡片 · 36 个代码模块和 notebook，数字都在本机 torch 2.15.0 上跑过 · 覆盖 study-plan 阶段 A–F 的全部讲义章节

## 这本书讲什么

从"一行 `y = x @ w` 在 PyTorch 里经过了什么"讲起：Tensor 的内存布局、分派器、自动求导、GPU 的异步执行、`torch.compile` 的三道工序；然后是典型用法（Module、DataLoader、训练循环、存档）和常用算子；再到多机多卡（集合通信、DDP、FSDP2、DTensor 与张量并行、流水线 / 上下文 / 专家并行、分布式存档与排错）、训练性能调优（MFU、profiler、数据管道、混合精度、显存、CUDA Graph），最后是推理优化与融合算子开发（roofline、KV cache、读 Inductor 代码、Triton、注册自定义算子、C++ 扩展、量化与导出、benchmark 与正确性）。

- **适合谁**：会写 Python、用过一点 PyTorch 的人。不需要事先懂 GPU 或分布式；要用到的数学只有求导的链式法则、指数和对数、矩阵乘法，都在用到的那一章现场讲。
- **读完能做什么**：读懂 AISPE 第 13–14 章和 Ultra-Scale Playbook；做 CS336 a1 / a2（训练循环、DDP、FSDP、Triton FlashAttention）；拿到一个慢的训练脚本知道先量什么、改什么；写一个注册进 PyTorch、能被 `torch.compile` 用的融合算子。
- **怎么读**：按章顺序读。每章先看讲解和动画，再自己做练习，最后用复习卡片隔几天考自己。总地图 00-overview.md 适合先扫一遍知道全貌。

## 讲解的特点

- **一次一小步**：每个概念按"白话类比 → 公式（每个符号说清楚怎么读）→ 小数字亲手算一遍"讲。
- **每章一个动画**：关键原理做成可交互页面，▶ 播放、→ 单步、↺ 从头开始，开关和滑块切换不同情形；GIF 版直接嵌在正文里。
- **数字都是实测的**：讲义里的输出、计时、文件大小都在本机（Mac，torch 2.15.0，CPU / MPS，分布式用 2 个 Gloo 进程）上跑过；需要 NVIDIA GPU 的部分标 `[GPU]`。
- **对照原文**：讲到 AISPE、Ultra-Scale Playbook、多机报告的地方标了本地文本的行号；在线官方文档标〔在线〕。
- **保留真实的学习过程**：练习与作答记录里是学习时的原话、对错和点评（目前除第 1 章外都待作答）。
- **术语不设门槛**：缩写第一次出现就给英文全称并逐词解释，全书缩写见 [缩写与术语表](https://wikieden.github.io/pytorch-inside-out/lessons/appendix-glossary.html)。
- **每章都有代码对照**：讲义末尾"代码对照"一节对应 `solutions/chapter-code/chNN_*.py`，配测试核对和讲义一致、配 notebook 一格一格跑；只算讲过的例子，不提前给练习答案。
- **每章收尾有口诀**：全部口诀汇总在本页末尾。

## 示例动画

<table>
<tr>
<td width="33%"><img src="https://wikieden.github.io/pytorch-inside-out/animations/01-op-journey.gif" alt="PyTorch 做了什么：一行 `y = x @ w` 的旅程"><br><b>第 1 章</b> · PyTorch = GPU 上的 numpy + 自动记账；一行代码走四站<br><a href="https://wikieden.github.io/pytorch-inside-out/lessons/01-op-journey.html">讲义</a> · <a href="https://wikieden.github.io/pytorch-inside-out/animations/01-op-journey.html">▶ 交互</a></td>
<td width="33%"><img src="https://wikieden.github.io/pytorch-inside-out/animations/02-tensor-stride.gif" alt="Tensor 内部：一块内存 + 一张"怎么读"的说明书"><br><b>第 2 章</b> · Tensor = 一排数 + 说明书（shape / stride / offset）；view 不复制<br><a href="https://wikieden.github.io/pytorch-inside-out/lessons/02-tensor-stride.html">讲义</a> · <a href="https://wikieden.github.io/pytorch-inside-out/animations/02-tensor-stride.html">▶ 交互</a></td>
<td width="33%"><img src="https://wikieden.github.io/pytorch-inside-out/animations/06-compile-fusion.gif" alt="eager 与 graph：`torch.compile` 里面的三道工序"><br><b>第 6 章</b> · compile 三道工序：Dynamo 抄图 → AOTAutograd → Inductor 融合；断图与重编译<br><a href="https://wikieden.github.io/pytorch-inside-out/lessons/06-compile-stack.html">讲义</a> · <a href="https://wikieden.github.io/pytorch-inside-out/animations/06-compile-fusion.html">▶ 交互</a></td>
</tr>
<tr>
<td width="33%"><img src="https://wikieden.github.io/pytorch-inside-out/animations/17-ring-allreduce.gif" alt="集合通信：五个动词和它们的代价"><br><b>第 17 章</b> · 五个集合通信动词；ring all-reduce 每卡发约 2K<br><a href="https://wikieden.github.io/pytorch-inside-out/lessons/17-collectives.html">讲义</a> · <a href="https://wikieden.github.io/pytorch-inside-out/animations/17-ring-allreduce.html">▶ 交互</a></td>
<td width="33%"><img src="https://wikieden.github.io/pytorch-inside-out/animations/19-fsdp-lifecycle.gif" alt="FSDP2：每卡只存一片，用到时再拼"><br><b>第 19 章</b> · ZeRO 1/2/3 显存账；FSDP2 层前拼、用完丢<br><a href="https://wikieden.github.io/pytorch-inside-out/lessons/19-fsdp2.html">讲义</a> · <a href="https://wikieden.github.io/pytorch-inside-out/animations/19-fsdp-lifecycle.html">▶ 交互</a></td>
<td width="33%"><img src="https://wikieden.github.io/pytorch-inside-out/animations/29-roofline.gif" alt="为什么融合能变快：算力受限与访存受限"><br><b>第 29 章</b> · roofline；访存受限只看字节；融合实测带宽不变时间减半<br><a href="https://wikieden.github.io/pytorch-inside-out/lessons/29-why-fusion.html">讲义</a> · <a href="https://wikieden.github.io/pytorch-inside-out/animations/29-roofline.html">▶ 交互</a></td>
</tr>
</table>

## 目录

| 章 | 主题 | 一句话精要 | 动画 | 卡片 | 状态 |
|---|---|---|---|---|---|
| **A · 核心机制** | | | | | |
| [01](https://wikieden.github.io/pytorch-inside-out/lessons/01-op-journey.html) | PyTorch 做了什么：一行 `y = x @ w` 的旅程 | PyTorch = GPU 上的 numpy + 自动记账；一行代码走四站 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/01-op-journey.html) | P01–P02 | 已讲，练习待作答 |
| [02](https://wikieden.github.io/pytorch-inside-out/lessons/02-tensor-stride.html) | Tensor 内部：一块内存 + 一张"怎么读"的说明书 | Tensor = 一排数 + 说明书（shape / stride / offset）；view 不复制 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/02-tensor-stride.html) | P03–P05 | 已写，练习待作答 |
| [03](https://wikieden.github.io/pytorch-inside-out/lessons/03-dispatcher.html) | 分派器：按标签一层层处理 | 标签梯子：Autocast → Autograd → 后端；Meta 只算形状 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/03-dispatcher.html) | P06–P08 | 已写，练习待作答 |
| [04](https://wikieden.github.io/pytorch-inside-out/lessons/04-autograd.html) | Autograd：账本怎么记、怎么倒着算 | 前向记节点、反向倒着乘；`.grad` 只累加；自定义 Function | [▶](https://wikieden.github.io/pytorch-inside-out/animations/04-autograd.html) | P09–P12 | 已写，练习待作答 |
| [05](https://wikieden.github.io/pytorch-inside-out/lessons/05-cuda-semantics.html) | GPU 语义：下单就走、何时会等、显存怎么借 | GPU 异步：挂单就走；同步点、stream、缓存分配器 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/05-async-queue.html) | P13–P16 | 已写，练习待作答 |
| [06](https://wikieden.github.io/pytorch-inside-out/lessons/06-compile-stack.html) | eager 与 graph：`torch.compile` 里面的三道工序 | compile 三道工序：Dynamo 抄图 → AOTAutograd → Inductor 融合；断图与重编译 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/06-compile-fusion.html) | P17–P20 | 已写，练习待作答 |
| **B · 典型用法** | | | | | |
| [07](https://wikieden.github.io/pytorch-inside-out/lessons/07-nn-module.html) | `nn.Module`：一棵自动登记参数的积木树 | Module 自动登记参数；state_dict 是清单；train / eval 只切层的行为 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/07-module-tree.html) | P21–P23 | 已写，练习待作答 |
| [08](https://wikieden.github.io/pytorch-inside-out/lessons/08-dataloader.html) | Dataset 与 DataLoader：给 GPU 备菜的流水线 | Dataset 回答第 i 个是什么，DataLoader 管分批、拼盘、并行备菜 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/08-dataloader-workers.html) | P24–P26 | 已写，练习待作答 |
| [09](https://wikieden.github.io/pytorch-inside-out/lessons/09-training-loop.html) | 训练循环：五行主干 + 三件护具 | 五行主干 + autocast + 梯度裁剪 + warmup / cosine | [▶](https://wikieden.github.io/pytorch-inside-out/animations/09-training-step.html) | P27–P30 | 已写，练习待作答 |
| [10](https://wikieden.github.io/pytorch-inside-out/lessons/10-save-load.html) | 保存与加载：存清单、不存对象，断点要存全 | 存 state_dict 不存对象；weights_only；续训存全套（AdamW 三倍大） | [▶](https://wikieden.github.io/pytorch-inside-out/animations/10-checkpoint.html) | P31–P33 | 已写，练习待作答 |
| **C · 常用算子** | | | | | |
| [11](https://wikieden.github.io/pytorch-inside-out/lessons/11-shape-broadcast.html) | 形状与广播：从右往左对齐，1 可以拉伸 | 广播从右对齐；keepdim 防陷阱；拆头先 view 再 transpose | [▶](https://wikieden.github.io/pytorch-inside-out/animations/11-broadcast.html) | P34–P36 | 已写，练习待作答 |
| [12](https://wikieden.github.io/pytorch-inside-out/lessons/12-index-scatter.html) | 索引与散射：按掩码挑、按下标取、按下标写回 | mask 填 −∞；where 形状不变；gather 按号取，scatter_add 按号加回 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/12-gather-scatter.html) | P37–P40 | 已写，练习待作答 |
| [13](https://wikieden.github.io/pytorch-inside-out/lessons/13-reduce-matmul.html) | 规约、矩阵乘与数值稳定 | dim 是被压扁的维；einsum；logsumexp 防溢出；bf16 累加卡在 256 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/13-numerics.html) | P41–P44 | 已写，练习待作答 |
| [14](https://wikieden.github.io/pytorch-inside-out/lessons/14-functional-attention.html) | `nn.functional`：归一化、激活与注意力 | LayerNorm / RMSNorm、SwiGLU；SDPA 自动挑 Flash；FlexAttention | [▶](https://wikieden.github.io/pytorch-inside-out/animations/14-attention-memory.html) | P45–P49 | 已写，练习待作答 |
| [15](https://wikieden.github.io/pytorch-inside-out/lessons/15-loss-optimizer.html) | 损失与优化器：cross_entropy 的细节和 AdamW 的一步 | cross_entropy 吃 logits；ignore_index；AdamW 一步怎么算 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/15-sgd-vs-adamw.html) | P50–P53 | 已写，练习待作答 |
| **D · 分布式训练** | | | | | |
| [16](https://wikieden.github.io/pytorch-inside-out/lessons/16-process-model.html) | 进程模型：一张卡一个进程，靠编号分工 | 每卡一进程；rank / local_rank / world_size；torchrun | [▶](https://wikieden.github.io/pytorch-inside-out/animations/16-ranks.html) | P54–P57 | 已写，练习待作答 |
| [17](https://wikieden.github.io/pytorch-inside-out/lessons/17-collectives.html) | 集合通信：五个动词和它们的代价 | 五个集合通信动词；ring all-reduce 每卡发约 2K | [▶](https://wikieden.github.io/pytorch-inside-out/animations/17-ring-allreduce.html) | P58–P60 | 已写，练习待作答 |
| [18](https://wikieden.github.io/pytorch-inside-out/lessons/18-ddp.html) | DDP：每卡一份模型，反向时边算边对答案 | DDP：广播参数 + 平均梯度；分桶重叠；no_sync | [▶](https://wikieden.github.io/pytorch-inside-out/animations/18-ddp-buckets.html) | P61–P63 | 已写，练习待作答 |
| [19](https://wikieden.github.io/pytorch-inside-out/lessons/19-fsdp2.html) | FSDP2：每卡只存一片，用到时再拼 | ZeRO 1/2/3 显存账；FSDP2 层前拼、用完丢 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/19-fsdp-lifecycle.html) | P64–P67 | 已写，练习待作答 |
| [20](https://wikieden.github.io/pytorch-inside-out/lessons/20-dtensor-tp.html) | DeviceMesh、DTensor 与张量并行 | DeviceMesh、DTensor 三种摆放；TP 先列切后行切 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/20-tensor-parallel.html) | P68–P71 | 已写，练习待作答 |
| [21](https://wikieden.github.io/pytorch-inside-out/lessons/21-pp-cp-ep.html) | 流水线、上下文、专家并行，以及怎么组合 | 流水线气泡 (p − 1)/m、1F1B；CP、EP；五维组合 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/21-pipeline-bubble.html) | P72–P75 | 已写，练习待作答 |
| [22](https://wikieden.github.io/pytorch-inside-out/lessons/22-dcp-debug.html) | 分布式存档与排错：每卡存一片，卡死先查顺序 | DCP 每卡存一片；hang 多半是通信顺序不一致 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/22-hang.html) | P76–P79 | 已写，练习待作答 |
| **E · 训练性能调优** | | | | | |
| [23](https://wikieden.github.io/pytorch-inside-out/lessons/23-perf-method.html) | 调优方法论：先算账，再测量，只改瓶颈 | 6N、MFU；四类瓶颈；改一处测一次 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/23-mfu.html) | P80–P82 | 已写，练习待作答 |
| [24](https://wikieden.github.io/pytorch-inside-out/lessons/24-profiler.html) | profiler：给训练拍一段"录像" | profiler：Self vs Total；fused AdamW 实测快 30% | [▶](https://wikieden.github.io/pytorch-inside-out/animations/24-trace.html) | P83–P86 | 已写，练习待作答 |
| [25](https://wikieden.github.io/pytorch-inside-out/lessons/25-input-sync.html) | 让 GPU 别等：数据管道与同步点 | 等数据计时；四段数据路；揪出同步点；stream 预取 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/25-input-pipeline.html) | P87–P90 | 已写，练习待作答 |
| [26](https://wikieden.github.io/pytorch-inside-out/lessons/26-mixed-precision.html) | 混合精度：bf16、fp16、TF32、FP8 怎么选 | 指数管范围、尾数管精度；GradScaler；TF32；FP8 缩放 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/26-float-formats.html) | P91–P94 | 已写，练习待作答 |
| [27](https://wikieden.github.io/pytorch-inside-out/lessons/27-memory.html) | 显存：谁占了、怎么省、怎么看 | 显存四块；激活重计算 9.1 → 1.1 MB；memory snapshot | [▶](https://wikieden.github.io/pytorch-inside-out/animations/27-memory-curve.html) | P95–P98 | 已写，练习待作答 |
| [28](https://wikieden.github.io/pytorch-inside-out/lessons/28-compile-graphs.html) | 训练里用 compile 与 CUDA Graphs：消灭碎活和启动开销 | 40 个小算子 compile 快 8 倍；区域编译；CUDA Graph | [▶](https://wikieden.github.io/pytorch-inside-out/animations/28-cuda-graph.html) | P99–P102 | 已写，练习待作答 |
| **F · 推理优化与融合开发** | | | | | |
| [29](https://wikieden.github.io/pytorch-inside-out/lessons/29-why-fusion.html) | 为什么融合能变快：算力受限与访存受限 | roofline；访存受限只看字节；融合实测带宽不变时间减半 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/29-roofline.html) | P103–P106 | 已写，练习待作答 |
| [30](https://wikieden.github.io/pytorch-inside-out/lessons/30-inference-mode.html) | 推理模式与 KV cache：生成一个 token 要做什么 | inference_mode；KV cache 原理与大小；decode 上限 = 带宽 ÷ 权重 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/30-kv-cache.html) | P107–P111 | 已写，练习待作答 |
| [31](https://wikieden.github.io/pytorch-inside-out/lessons/31-read-inductor.html) | 读 Inductor 生成的代码：它已经帮你融了什么 | 读 Inductor 生成代码：fused kernel 名、extern 调库 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/31-inductor-fusion.html) | P112–P115 | 已写，练习待作答 |
| [32](https://wikieden.github.io/pytorch-inside-out/lessons/32-triton-model.html) | Triton 编程模型：一个 program 处理一块数据 | Triton：一个 program 管一块；offs、mask、load / store | [▶](https://wikieden.github.io/pytorch-inside-out/animations/32-triton-grid.html) | P116–P119 | 已写，练习待作答 |
| [33](https://wikieden.github.io/pytorch-inside-out/lessons/33-custom-op.html) | 注册自定义算子：让你的 kernel 成为 PyTorch 的"一等公民" | custom_op + register_fake + register_autograd + opcheck | [▶](https://wikieden.github.io/pytorch-inside-out/animations/33-custom-op.html) | P120–P123 | 已写，练习待作答 |
| [34](https://wikieden.github.io/pytorch-inside-out/lessons/34-cpp-extension.html) | C++ / CUDA 扩展：TORCH_LIBRARY 把 C++ 函数登记进分派表 | TORCH_LIBRARY 加行、IMPL 填列；-O3；手写不自动赢 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/34-torch-library.html) | P124–P127 | 已写，练习待作答 |
| [35](https://wikieden.github.io/pytorch-inside-out/lessons/35-quant-export.html) | 量化与导出：模型变小、脱离 Python 部署 | 对称量化；weight-only；torch.export + AOTInductor | [▶](https://wikieden.github.io/pytorch-inside-out/animations/35-quantize.html) | P128–P131 | 已写，练习待作答 |
| [36](https://wikieden.github.io/pytorch-inside-out/lessons/36-bench-correctness.html) | benchmark 与正确性：测准、对齐、再上线 | 测准时间、有效带宽、assert_close 容差、测试矩阵 | [▶](https://wikieden.github.io/pytorch-inside-out/animations/36-bench.html) | P132–P135 | 已写，练习待作答 |

## 联系与合作

欢迎来信：<wikieden@gmail.com>。可以合作的方向：

- 讲义勘误与建议
- 课程 / 企业内训
- 内容合作与转载授权
- 赞助本系列持续更新

赞助请发邮件联系。

---

本仓库由私有学习仓库里的脚本 `tools/publish-transformer/build.py` 自动生成，内容摘自讲义目录，不要直接修改这里的文件。
