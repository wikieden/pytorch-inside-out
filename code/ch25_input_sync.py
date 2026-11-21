"""讲义：第 25 章 让 GPU 别等：数据管道与同步点。

运行：uv run python ch25_input_sync.py
"""

import torch


def data_wait_share(t_data_ms: float, t_compute_ms: float) -> float:
    return t_data_ms / (t_data_ms + t_compute_ms)


def accumulate_on_device(losses):
    """指标在张量上累加，只在最后 .item() 一次（同步一次）。"""
    running = torch.zeros(())
    for loss in losses:
        running += loss.detach()
    return (running / len(losses)).item()


def main():
    print("\n第一步 等数据 18 ms、计算 30 ms → GPU 等待占比", f"{data_wait_share(18, 30):.1%}")
    print("\n第三步 在张量上累加 4 个 loss，最后取一次：", accumulate_on_device([torch.tensor(v) for v in (1.0, 2.0, 3.0, 4.0)]))


if __name__ == "__main__":
    main()
