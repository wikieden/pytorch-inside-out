"""讲义：第 19 章 FSDP2：每卡只存一片，用到时再拼。

运行：uv run python ch19_fsdp2.py     # 本机起 2 个 Gloo 进程
"""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from dist_util import run


def zero_memory_gb(psi: float, n: int) -> dict:
    """混合精度 AdamW 每卡显存（GB）：DDP 16Ψ，ZeRO-1/2/3（Playbook 第 840–873 行公式）。"""
    g = 1e9
    return dict(DDP=16 * psi / g, ZeRO1=(4 * psi + 12 * psi / n) / g, ZeRO2=(2 * psi + 14 * psi / n) / g,
                ZeRO3=16 * psi / n / g)


def shard_demo(rank, world):
    mesh = init_device_mesh("cpu", (world,))
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 2))
    for layer in m:
        fully_shard(layer, mesh=mesh)
    fully_shard(m, mesh=mesh)
    p = m[0].weight
    m(torch.ones(1, 4)).sum().backward()
    return dict(type=type(p).__name__, global_shape=tuple(p.shape), local_shape=tuple(p.to_local().shape),
                placements=str(p.placements), grad_local=tuple(m[0].weight.grad.to_local().shape))


def main():
    print("\n第一、二步 7B、8 卡每卡显存（GB）：", {k: round(v, 2) for k, v in zero_memory_gb(7e9, 8).items()})
    print("\n第三步 fully_shard 之后：")
    for r, res in enumerate(run(shard_demo)):
        print(f"  rank {r}: {res}")


if __name__ == "__main__":
    main()
