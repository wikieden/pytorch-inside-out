"""讲义：第 16 章 进程模型：一张卡一个进程，靠编号分工。

运行：uv run python ch16_process_model.py     # 本机起 2 个 Gloo 进程
"""

import os

import torch.distributed as dist
from torch.utils.data import DistributedSampler

from dist_util import run


def global_rank(node_rank: int, nproc_per_node: int, local_rank: int) -> int:
    return node_rank * nproc_per_node + local_rank


def sampler_indices(n: int, world: int, rank: int, k: int = 3) -> list:
    """DistributedSampler（不打乱）分给 rank 的前 k 个下标。"""
    s = DistributedSampler(list(range(n)), num_replicas=world, rank=rank, shuffle=False)
    return list(s)[:k]


def whoami(rank, world):
    return dict(rank=dist.get_rank(), world=dist.get_world_size(), local=os.environ["LOCAL_RANK"],
                master=f'{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"] != ""}')


def main():
    print("\n第二步 机器 1 上 local_rank 2（每台 4 卡）→ rank", global_rank(1, 4, 2))
    print("\n第三步 两个进程各自报到：", run(whoami))
    print("\n第四步 8 卡时 rank 0 / 1 拿到的前 3 个下标：", sampler_indices(100, 8, 0), sampler_indices(100, 8, 1))


if __name__ == "__main__":
    main()
