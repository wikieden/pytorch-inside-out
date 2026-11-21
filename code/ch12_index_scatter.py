"""讲义：第 12 章 索引与散射：按掩码挑、按下标取、按下标写回。

运行：uv run python ch12_index_scatter.py
"""

import torch


def masked_softmax():
    s = torch.tensor([[2.0, 1.0, 0.0], [0.5, 3.0, 1.0]])
    keep = torch.tensor([[True, False, False], [True, True, False]])
    return s.masked_fill(~keep, float("-inf")).softmax(-1)


def leaky_where():
    x = torch.tensor([-2.0, 1.0, 3.0])
    return torch.where(x > 0, x, 0.1 * x), x[x > 0]


def gather_target_logp():
    logp = torch.log(torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]))
    tgt = torch.tensor([0, 2])
    return logp.gather(1, tgt.unsqueeze(1)).squeeze(1)


def scatter_add_demo():
    out = torch.zeros(3)
    idx = torch.tensor([0, 2, 0, 1])
    src = torch.tensor([1.0, 2.0, 3.0, 4.0])
    return out.scatter_add(0, idx, src), torch.zeros(3).index_add(0, idx, src)


def topk_demo():
    return torch.tensor([0.1, 0.5, 0.2, 0.9]).topk(2)


def main():
    print("\n第一步 mask + softmax：\n", masked_softmax())
    print("\n第二步 where / 布尔索引：", leaky_where())
    print("\n第三步 gather 目标类 log 概率：", gather_target_logp())
    print("\n第四步 scatter_add / index_add：", scatter_add_demo())
    print("\n第五步 topk：", topk_demo())


if __name__ == "__main__":
    main()
