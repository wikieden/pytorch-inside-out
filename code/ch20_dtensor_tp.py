"""讲义：第 20 章 DeviceMesh、DTensor 与张量并行。

运行：uv run python ch20_dtensor_tp.py     # 本机起 2 个 Gloo 进程
"""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from dist_util import run


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(2, 4, bias=False)
        self.w2 = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


def dtensor_demo(rank, world):
    mesh = init_device_mesh("cpu", (world,))
    dt = distribute_tensor(torch.arange(8.0).reshape(4, 2), mesh, [Shard(0)])
    rep = dt.redistribute(mesh, [Replicate()])
    s = (dt * 2).sum()
    return dict(local=dt.to_local().tolist(), replicate_shape=tuple(rep.to_local().shape),
                sum_placement=str(s.placements), sum=s.full_tensor().item())


def tp_demo(rank, world):
    mesh = init_device_mesh("cpu", (world,))
    m = MLP()
    with torch.no_grad():
        m.w1.weight.copy_(torch.tensor([[1.0, 0], [0, 1], [1, 0], [0, 1]]))     # (out=4, in=2)：就是讲义 W1 的转置
        m.w2.weight.copy_(torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1]]))         # (out=2, in=4)
    x = torch.tensor([[1.0, 2.0]])
    full = m(x).tolist()
    parallelize_module(m, mesh, {"w1": ColwiseParallel(), "w2": RowwiseParallel()})
    return dict(full=full, tp=m(x).tolist(), w1=(tuple(m.w1.weight.to_local().shape), str(m.w1.weight.placements)),
                w2=(tuple(m.w2.weight.to_local().shape), str(m.w2.weight.placements)))


def main():
    print("\n第二步 DTensor：")
    for r, res in enumerate(run(dtensor_demo)):
        print(f"  rank {r}: {res}")
    print("\n第三步 MLP 张量并行：")
    for r, res in enumerate(run(tp_demo)):
        print(f"  rank {r}: {res}")


if __name__ == "__main__":
    main()
