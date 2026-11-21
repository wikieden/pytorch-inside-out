"""讲义：第 35 章 量化与导出：模型变小、脱离 Python 部署。

运行：uv run python ch35_quant_export.py
"""

import io
import os
import tempfile

import torch
import torch.nn as nn


def symmetric_quant(w, bits: int):
    """对称量化：s = max|w| / (2^(b−1) − 1)，q = round(w / s)，还原 q × s。"""
    w = torch.tensor(w)
    s = w.abs().max() / (2 ** (bits - 1) - 1)
    q = torch.round(w / s)
    return s.item(), q.int().tolist(), (q * s).tolist()


def state_mb(m: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return buf.tell() / 2**20


def int8_weight_only():
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024)).eval()
    x = torch.randn(4, 1024)
    ref = m(x)
    before = state_mb(m)
    quantize_(m, Int8WeightOnlyConfig())
    y = m(x)
    return round(before, 2), round(state_mb(m), 2), ((y - ref).norm() / ref.norm()).item(), type(m[0].weight).__name__


def export_and_aoti():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4)).eval()
    x = torch.randn(2, 16)
    batch = torch.export.Dim("batch", min=1, max=64)
    ep = torch.export.export(m, (x,), dynamic_shapes={"input": {0: batch}})
    ops = [n.target.__name__ for n in ep.graph.nodes if n.op == "call_function"]
    path = torch._inductor.aoti_compile_and_package(ep, package_path=os.path.join(tempfile.gettempdir(), "ptdd_ch35.pt2"))
    runner = torch._inductor.aoti_load_package(path)
    return ops, tuple(runner(torch.randn(5, 16)).shape), torch.allclose(runner(x), m(x), atol=1e-5)


def main():
    print("\n第一步 [-0.6, 0.2, 1.0]：int8", symmetric_quant([-0.6, 0.2, 1.0], 8), "\n                    int4", symmetric_quant([-0.6, 0.2, 1.0], 4))
    print("\n第二步 int8 weight-only（前 MB, 后 MB, 相对误差, 权重类型）：", int8_weight_only())
    print("\n第三、四步 export 的算子 / AOTInductor 跑 batch 5 / 与原模型一致：", export_and_aoti())


if __name__ == "__main__":
    main()
