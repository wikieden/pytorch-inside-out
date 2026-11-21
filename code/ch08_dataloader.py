"""讲义：第 8 章 Dataset 与 DataLoader：给 GPU 备菜的流水线。

运行：uv run python ch08_dataloader.py
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def batches(n: int = 10, batch_size: int = 4, drop_last: bool = False):
    ds = TensorDataset(torch.randn(n, 4), torch.arange(n))
    return [(tuple(x.shape), y.tolist()) for x, y in DataLoader(ds, batch_size=batch_size, drop_last=drop_last)]


def pad_demo():
    seqs = [torch.ones(2), torch.ones(5), torch.ones(3)]
    return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)


def gpu_busy(t_data: float, t_step: float, workers: int) -> float:
    """稳定后 GPU 忙碌占比：0 个 worker 串行；k 个 worker 时供给速度 k / t_data。"""
    if workers == 0:
        return t_step / (t_data + t_step)
    per_batch = t_data / workers
    return t_step / max(t_step, per_batch)


def main():
    print("\n第一步 10 个样本、每批 4：", batches())
    print("drop_last=True 时批数：", len(batches(drop_last=True)))
    print("\n第二步 补齐：\n", pad_demo())
    print("\n第三步 读一批 40 ms、训一步 25 ms 时 GPU 忙碌占比：")
    for k in (0, 1, 2, 4):
        print(f"  num_workers={k}: {gpu_busy(40, 25, k):.0%}")


if __name__ == "__main__":
    main()
