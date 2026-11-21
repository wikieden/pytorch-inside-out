"""讲义：第 34 章 C++ / CUDA 扩展：TORCH_LIBRARY 把 C++ 函数登记进分派表。

运行：uv run python ch34_cpp_extension.py     # 第一次要编译几秒；需要 ninja（已在依赖里）和 C++ 编译器
"""

import os
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
at::Tensor add_relu_cpu(const at::Tensor& x, const at::Tensor& y) {
  TORCH_CHECK(x.sizes() == y.sizes(), "shape mismatch");
  auto xc = x.contiguous(), yc = y.contiguous();
  auto out = torch::empty_like(xc);
  const float* a = xc.data_ptr<float>(); const float* b = yc.data_ptr<float>(); float* o = out.data_ptr<float>();
  at::parallel_for(0, xc.numel(), 1 << 16, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; ++i) { float v = a[i] + b[i]; o[i] = v > 0 ? v : 0; }
  });
  return out;
}
TORCH_LIBRARY(NS, m) { m.def("add_relu(Tensor x, Tensor y) -> Tensor"); }
TORCH_LIBRARY_IMPL(NS, CPU, m) { m.impl("add_relu", &add_relu_cpu); }
"""


def build(ns: str, flags: list):
    """即时编译并登记 torch.ops.<ns>.add_relu；返回编译秒数。"""
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]   # ninja 在 .venv/bin
    t = time.time()
    load_inline(name=ns, cpp_sources=SRC.replace("NS", ns), is_python_module=False, extra_cflags=flags)
    torch.library.register_fake(f"{ns}::add_relu")(lambda x, y: torch.empty_like(x))
    return time.time() - t


def per_call_ms(fn, reps: int = 20) -> float:
    fn()
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1e3


def main():
    print("\n第二步 编译（-O3）用时", round(build("ptdd_ext_o3", ["-O3"]), 1), "s")
    op = torch.ops.ptdd_ext_o3.add_relu
    print("结果：", op(torch.tensor([-2.0, 1.0, 3.0]), torch.tensor([1.0, -3.0, 1.0])).tolist())
    print("opcheck：", torch.library.opcheck(op.default, (torch.randn(5), torch.randn(5))))
    build("ptdd_ext_o0", ["-O0"])
    a, b = torch.randn(1 << 24), torch.randn(1 << 24)
    cf = torch.compile(lambda: torch.relu(a + b))
    print("\n第四步 每次 ms：eager", round(per_call_ms(lambda: torch.relu(a + b)), 2),
          "-O0", round(per_call_ms(lambda: torch.ops.ptdd_ext_o0.add_relu(a, b)), 2),
          "-O3", round(per_call_ms(lambda: op(a, b)), 2), "compile", round(per_call_ms(cf), 2))


if __name__ == "__main__":
    main()
