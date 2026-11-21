"""讲义：第 18 章 DDP：每卡一份模型，反向时边算边对答案。

运行：uv run python ch18_ddp.py     # 本机起 2 个 Gloo 进程
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from dist_util import run


def ddp_demo(rank, world):
    torch.manual_seed(rank)                       # 故意让两个 rank 初始化不同
    m = nn.Linear(2, 1)
    d = DDP(m)                                    # 包装时 broadcast rank 0 的参数
    w_after = m.weight.detach().clone().tolist()
    xb = torch.tensor([[1.0, 0.0]]) if rank == 0 else torch.tensor([[0.0, 1.0]])
    d(xb).sum().backward()                        # 本地梯度 [1, 0] / [0, 1]，all_reduce 后取平均
    return dict(weight=w_after, grad=m.weight.grad.tolist())


def bucket_timeline(n_buckets=4, backward_ms=40, comm_ms=32, overlap=True):
    """分桶重叠的总用时（不算启动开销）：返回 ms。"""
    seg, per = backward_ms / n_buckets, comm_ms / n_buckets
    t = 0.0
    for b in range(n_buckets):
        start = (b + 1) * seg if overlap else backward_ms
        t = max(t, start) + per
    return max(backward_ms, t)


def main():
    print("\n第二步 DDP 三件事：")
    for r, res in enumerate(run(ddp_demo)):
        print(f"  rank {r}: {res}")
    print("\n第三步 4 段反向各 10 ms、每桶通信 8 ms：算完再发", bucket_timeline(overlap=False), "ms，分桶重叠", bucket_timeline(), "ms")


if __name__ == "__main__":
    main()
