"""讲义：第 23 章 调优方法论：先算账，再测量，只改瓶颈。

运行：uv run python ch23_perf_method.py
"""


def train_flops_per_token(n_params: float) -> dict:
    return dict(forward=2 * n_params, backward=4 * n_params, total=6 * n_params)


def mfu(n_params: float, tokens_per_s: float, n_gpus: int, peak_flops: float) -> float:
    return 6 * n_params * tokens_per_s / (n_gpus * peak_flops)


def train_time_days(n_params, n_tokens, n_gpus, peak_flops, mfu_value) -> float:
    return 6 * n_params * n_tokens / (n_gpus * peak_flops * mfu_value) / 86400


def main():
    print("\n第一步 7B 每 token：", train_flops_per_token(7e9), "；训 1 万亿 token 共", f"{6 * 7e9 * 1e12:.1e}", "FLOPs")
    print("\n第二步 7B、8×H100、9.6 万 token/s → MFU", f"{mfu(7e9, 96_000, 8, 989e12):.1%}")
    print("同样配置训 1 万亿 token 约", round(train_time_days(7e9, 1e12, 8, 989e12, 0.51), 1), "天")


if __name__ == "__main__":
    main()
