"""讲义：第 17 章 集合通信：五个动词和它们的代价。

运行：uv run python ch17_collectives.py     # 本机起 2 个 Gloo 进程
"""

import torch
import torch.distributed as dist

from dist_util import run


def five_verbs(rank, world):
    x = torch.tensor([1.0, 2.0]) * (rank + 1)
    y = x.clone()
    dist.all_reduce(y)
    g = [torch.zeros(2) for _ in range(world)]
    dist.all_gather(g, x)
    out = torch.zeros(1)
    dist.reduce_scatter_tensor(out, torch.tensor([10.0, 20.0]) * (rank + 1))
    b = torch.tensor([float(rank)])
    dist.broadcast(b, src=1)
    return dict(all_reduce=y.tolist(), all_gather=[t.tolist() for t in g], reduce_scatter=out.tolist(),
                broadcast=b.tolist())


def ring_allreduce_sent(K: float, N: int) -> float:
    """ring all-reduce 每卡发送的元素数：2(N−1)K/N。"""
    return 2 * (N - 1) * K / N


def main():
    print("\n第一步 2 个进程的五个动词：")
    for r, res in enumerate(run(five_verbs)):
        print(f"  rank {r}: {res}")
    print("\n第三步 每卡发送量（以 K 为单位）：4 卡", ring_allreduce_sent(1, 4), "，64 卡", round(ring_allreduce_sent(1, 64), 3))


if __name__ == "__main__":
    main()
