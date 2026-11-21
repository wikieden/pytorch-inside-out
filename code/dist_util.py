"""本机起多个进程跑分布式例子（Gloo 后端，CPU）。第 16–22 章共用。

Mac 上 Gloo 默认会找错网卡导致卡住，这里固定用回环地址（第 16 章第三步的坑）。
"""

import os
import platform
import socket

import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, fn, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    if platform.system() == "Darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        out[rank] = fn(rank, world)
    finally:
        dist.destroy_process_group()


def run(fn, world: int = 2) -> list:
    """在 world 个进程里各跑一次 fn(rank, world)，返回按 rank 排好的结果列表。fn 必须是模块顶层函数。"""
    with mp.Manager() as m:
        out = m.dict()
        mp.spawn(_worker, args=(world, _free_port(), fn, out), nprocs=world, join=True)
        return [out[r] for r in range(world)]
