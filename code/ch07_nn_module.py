"""讲义：第 7 章 nn.Module：一棵自动登记参数的积木树。

运行：uv run python ch07_nn_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.fc1(x))))


class PlainList(nn.Module):
    """反例：普通 list 里的层不会被登记。"""

    def __init__(self):
        super().__init__()
        self.blocks = [nn.Linear(16, 16) for _ in range(4)]
        self.head = nn.Linear(16, 2)


def count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def dropout_once(seed: int = 0):
    torch.manual_seed(seed)
    d = nn.Dropout(0.5)
    d.train()
    return d(torch.ones(8)).tolist()


def train_eval_repeat():
    """train 时两次输出不同、eval 时相同。"""
    torch.manual_seed(0)
    m = MLP()
    x = torch.ones(1, 4)
    m.train()
    a, b = m(x), m(x)
    m.eval()
    c, d = m(x), m(x)
    return torch.equal(a, b), torch.equal(c, d)


def main():
    m = MLP()
    print("\n第一步 named_parameters：", [(n, tuple(p.shape)) for n, p in m.named_parameters()])
    print("参数总数：", count(m))
    print("\n第二步 模块树：", [n for n, _ in m.named_modules()])
    print("state_dict 键：", list(m.state_dict().keys()))
    print("\n第三步 Dropout 训练时：", dropout_once())
    print("train 两次相同？eval 两次相同？", train_eval_repeat())
    print("\n第四步 hook")
    h = m.fc1.register_forward_hook(lambda mod, inp, out: print("  fc1 输出形状", tuple(out.shape)))
    m(torch.ones(1, 4))
    h.remove()


if __name__ == "__main__":
    main()
