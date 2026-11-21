"""讲义：第 21 章 流水线、上下文、专家并行，以及怎么组合。

运行：uv run python ch21_pp_cp_ep.py
"""


def bubble_ratio(p: int, m: int = 1) -> float:
    """气泡 / 有效时间 = (p − 1) / m（Playbook 第 1693 行）。"""
    return (p - 1) / m


def simulate_afab(p: int, m: int, tf: int = 1, tb: int = 2):
    """逐格模拟 AFAB 调度：返回 (总时长, 每卡有效时长, 卡 0 同时存的激活份数峰值)。"""
    f_end = [[0] * m for _ in range(p)]
    t = 0
    for s in range(p):                      # 前向：第 s 段第 i 个小批，要等上一段算完
        t = 0
        for i in range(m):
            start = max(t, f_end[s - 1][i] if s else 0)
            f_end[s][i] = start + tf
            t = f_end[s][i]
    b_end = [[0] * m for _ in range(p)]
    for s in reversed(range(p)):
        t = f_end[s][m - 1]
        for i in range(m):
            dep = f_end[s][i] if s == p - 1 else b_end[s + 1][i]
            start = max(t, dep)
            b_end[s][i] = start + tb
            t = b_end[s][i]
    total = max(max(r) for r in b_end)
    return total, m * (tf + tb), m


def main():
    print("\n第一步 只送一个 batch：p = 4 →", bubble_ratio(4), "倍")
    print("\n第二步 p = 4、m = 8 →", bubble_ratio(4, 8), "；m = 32 →", round(bubble_ratio(4, 32), 3))
    total, useful, peak = simulate_afab(4, 8)
    print("逐格模拟 AFAB p=4、m=8：总长", total, "有效", useful, "气泡/有效", round((total - useful) / useful, 3), "卡 0 激活峰值", peak)


if __name__ == "__main__":
    main()
