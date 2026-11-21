"""讲义：第 22 章 分布式存档与排错：每卡存一片，卡死先查顺序。

运行：uv run python ch22_dcp_debug.py     # 本机起 2 个 Gloo 进程，存档写到临时目录
"""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from dist_util import run

CKPT = os.path.join(tempfile.gettempdir(), "pytorch-deep-dive-ch22-ckpt")


def dcp_roundtrip(rank, world):
    mesh = init_device_mesh("cpu", (world,))
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 2))
    for layer in m:
        fully_shard(layer, mesh=mesh)
    fully_shard(m, mesh=mesh)
    opt = torch.optim.AdamW(m.parameters(), lr=0.1)
    m(torch.ones(1, 4)).sum().backward()
    opt.step()
    msd, osd = get_state_dict(m, opt)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=CKPT)       # 所有 rank 都要调用
    w0 = m[0].weight.full_tensor().clone()                           # full_tensor 也是集合通信
    with torch.no_grad():
        m[0].weight.zero_()
    msd, osd = get_state_dict(m, opt)
    state = {"model": msd, "optim": osd}
    dcp.load(state, checkpoint_id=CKPT)
    set_state_dict(m, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    ok = torch.equal(m[0].weight.full_tensor(), w0)
    dist.barrier()
    return dict(restored=ok, files=sorted(os.listdir(CKPT)) if rank == 0 else None)


def call_sequence(rank: int, over_threshold: dict, fixed: bool) -> list:
    """每步调用了哪些集合通信（不真的通信，只列出顺序，用来比对两个 rank 是否一致）。"""
    seq = []
    for step in (1, 2, 3):
        if fixed:
            seq.append(f"flag{step}")
        seq.append(f"grad{step}")
        hit = any(over_threshold[r][step] for r in over_threshold) if fixed else over_threshold[rank][step]
        if hit:
            seq.append(f"stats{step}")
    return seq


def main():
    print("\n第一步 DCP 存一遍、读回来：")
    for r, res in enumerate(run(dcp_roundtrip)):
        print(f"  rank {r}: {res}")
    over = {0: {1: False, 2: True, 3: False}, 1: {1: False, 2: False, 3: False}}
    print("\n第三步 条件依赖本 rank：", call_sequence(0, over, False), "vs", call_sequence(1, over, False))
    print("先统一条件：", call_sequence(0, over, True), "vs", call_sequence(1, over, True))


if __name__ == "__main__":
    main()
