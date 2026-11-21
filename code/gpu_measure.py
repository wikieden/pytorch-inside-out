"""讲义里标 [GPU] 的部分：在 NVIDIA GPU 上实测（2026-11-15 在 hisense165，RTX 5090 Laptop 上跑过）。

运行：uv run python gpu_measure.py            # 全部跑一遍，结果写到 gpu_results.json
      uv run python gpu_measure.py ch28 ch32  # 只跑某几章
没有 CUDA 时直接退出；Mac 上的版本见各章 chNN_*.py。
"""

import json
import os
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
MEASURES = {}


def measure(fn):
    MEASURES[fn.__name__] = fn
    return fn


def cuda_ms(fn, reps: int = 20, warmup: int = 3) -> float:
    """用 CUDA Event 计时，返回每次毫秒数（第 5、36 章的正确做法）。"""
    for _ in range(warmup):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def r(x, n=2):
    return round(float(x), n)


# ---------------------------------------------------------------- 硬件基本数
@measure
def device():
    """本机峰值：bf16 矩阵乘 TFLOPS 和显存带宽，算出拐点（第 29 章 roofline）。"""
    p = torch.cuda.get_device_properties(0)
    bw = device_bw()
    n = 8192
    a = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
    tf = 2 * n**3 / cuda_ms(lambda: a @ a, reps=10) / 1e9
    return dict(name=p.name, sm=f"sm_{p.major}{p.minor}", mem_gib=r(p.total_memory / 2**30, 1), sms=p.multi_processor_count,
                torch=torch.__version__, cuda=torch.version.cuda, bandwidth_tbs=r(bw), bf16_tflops=r(tf, 0),
                ridge_flop_per_byte=r(tf * 1e12 / (bw * 1e12), 0))


# ---------------------------------------------------------------- 第 5 章
@measure
def ch05():
    """异步：launch 立刻返回，sync 才等到算完；缓存分配器：del 后 allocated 降、reserved 不降。"""
    a = torch.randn(8192, 8192, device=DEV)
    a @ a
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    b = a @ a
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    del b
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    x = torch.empty(256 * 2**20, device=DEV)  # 1 GiB
    alloc1, res1 = torch.cuda.memory_allocated() - base, torch.cuda.memory_reserved()
    del x
    alloc2, res2 = torch.cuda.memory_allocated() - base, torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    res3 = torch.cuda.memory_reserved()
    g = 2**30
    return dict(launch_ms=r((t1 - t0) * 1e3, 3), done_ms=r((t2 - t0) * 1e3), alloc_after_new_gib=r(alloc1 / g),
                alloc_after_del_gib=r(alloc2 / g), reserved_after_del_gib=r(res2 / g), reserved_after_empty_cache_gib=r(res3 / g),
                reserved_before_del_gib=r(res1 / g))


# ---------------------------------------------------------------- 第 14 章
@measure
def ch14():
    """SDPA 各后端在 GPU 上的速度；FlexAttention 编译后和 SDPA 比。"""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    B, H, T, D = 2, 16, 4096, 64
    q, k, v = (torch.randn(B, H, T, D, device=DEV, dtype=torch.bfloat16) for _ in range(3))
    out = {}
    for be in [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]:
        try:
            with sdpa_kernel(be):
                out[be.name.lower() + "_ms"] = r(cuda_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)))
        except Exception as e:  # noqa: BLE001
            out[be.name.lower() + "_ms"] = f"不可用 {type(e).__name__}"
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out["default_ms"] = r(cuda_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)))
    bm = create_block_mask(lambda b, h, qi, ki: qi >= ki, None, None, T, T, device=DEV)
    flex = torch.compile(flex_attention)
    o = flex(q, k, v, block_mask=bm)
    out["flex_compiled_ms"] = r(cuda_ms(lambda: flex(q, k, v, block_mask=bm)))
    out["flex_max_err_vs_math"] = r((o.float() - ref.float()).abs().max(), 4)
    out["shape"] = f"B{B} H{H} T{T} D{D} bf16 causal"
    out["math_attn_matrix_gib"] = r(B * H * T * T * 2 / 2**30)
    return out


# ---------------------------------------------------------------- 第 15 / 24 章
@measure
def ch15():
    """AdamW 三种实现：for 循环 / foreach / fused（第 15 章表格，第 24 章 profiler 例子）。"""
    torch.manual_seed(0)
    m = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(48)]).to(DEV)
    for p in m.parameters():
        p.grad = torch.randn_like(p)
    out = dict(n_params_m=r(sum(p.numel() for p in m.parameters()) / 1e6, 1), n_tensors=len(list(m.parameters())))
    small = [torch.randn(64, 64, device=DEV, requires_grad=True) for _ in range(2000)]  # 同样的事，换成 2000 个小张量
    for p in small:
        p.grad = torch.randn_like(p)
    out.update(small_n_tensors=len(small), small_n_params_m=r(2000 * 64 * 64 / 1e6, 1))
    for prefix, params in [("big_", list(m.parameters())), ("small_", small)]:
        for name, kw in [("for_loop", dict(foreach=False)), ("foreach", dict(foreach=True)), ("fused", dict(fused=True))]:
            opt = torch.optim.AdamW(params, lr=1e-3, **kw)
            out[prefix + name + "_ms"] = r(cuda_ms(opt.step, reps=30), 3)
    out["big_fused_floor_ms"] = r(out["n_params_m"] * 1e6 * 7 * 4 / (device_bw() * 1e12) * 1e3, 3)  # 读 p g m v、写 p m v
    return out


# ---------------------------------------------------------------- 第 25 章
@measure
def ch25():
    """sync debug mode 抓隐藏同步点；指标每步 .item() vs 留在 GPU 上累计。"""
    x = torch.randn(1024, 1024, device=DEV)
    torch.cuda.set_sync_debug_mode("warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = (x > 0).sum().item()
        _ = x[x > 0]
        _ = x.nonzero()
    torch.cuda.set_sync_debug_mode(0)
    caught = [str(i.message).split("\n")[0][:60] for i in w]
    xs, ws = torch.randn(64, 64, device=DEV), torch.randn(64, 64, device=DEV)  # 小活：一步只要几微秒，同步开销才显出来

    def step_item():
        tot = 0.0
        for _ in range(200):
            tot += (xs @ ws).relu().mean().item()

    def step_device():
        tot = torch.zeros((), device=DEV)
        for _ in range(200):
            tot += (xs @ ws).relu().mean()
        return tot.item()

    return dict(sync_warnings=len(caught), examples=caught[:3], item_every_step_ms=r(cuda_ms(step_item, reps=10)),
                accumulate_on_device_ms=r(cuda_ms(step_device, reps=10)))


# ---------------------------------------------------------------- 第 26 章
@measure
def ch26():
    """各精度矩阵乘吞吐（fp32 / TF32 / fp16 / bf16 / FP8），以及 autocast bf16 训练一步的提速。"""
    n = 8192
    out = {}
    for label, dt, prec in [("fp32", torch.float32, "highest"), ("tf32", torch.float32, "high"),
                            ("fp16", torch.float16, "highest"), ("bf16", torch.bfloat16, "highest")]:
        torch.set_float32_matmul_precision(prec)
        a = torch.randn(n, n, device=DEV, dtype=dt)
        out[label + "_tflops"] = r(2 * n**3 / cuda_ms(lambda: a @ a, reps=10) / 1e9, 0)
    torch.set_float32_matmul_precision("highest")
    a8 = torch.randn(n, n, device=DEV).to(torch.float8_e4m3fn)
    b8 = torch.randn(n, n, device=DEV).to(torch.float8_e4m3fn).t()
    one = torch.tensor(1.0, device=DEV)
    out["fp8_tflops"] = r(2 * n**3 / cuda_ms(lambda: torch._scaled_mm(a8, b8, scale_a=one, scale_b=one,
                                                                       out_dtype=torch.bfloat16), reps=10) / 1e9, 0)
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4096, 16384), nn.GELU(), nn.Linear(16384, 4096)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), fused=True)
    x = torch.randn(4096, 4096, device=DEV)

    def step(amp):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = m(x).float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    out["mlp_step_fp32_ms"] = r(cuda_ms(lambda: step(False), reps=10))
    out["mlp_step_bf16_autocast_ms"] = r(cuda_ms(lambda: step(True), reps=10))
    torch.set_float32_matmul_precision("high")
    out["mlp_step_tf32_ms"] = r(cuda_ms(lambda: step(False), reps=10))
    torch.set_float32_matmul_precision("highest")
    return out


# ---------------------------------------------------------------- 第 27 章
@measure
def ch27():
    """激活重算省多少显存、多花多少时间；峰值显存用 max_memory_allocated 量。"""
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(0)
    blocks = nn.ModuleList([nn.Sequential(nn.Linear(2048, 8192), nn.GELU(), nn.Linear(8192, 2048)) for _ in range(8)]).to(DEV)
    x = torch.randn(8192, 2048, device=DEV, requires_grad=True)

    def run(ckpt):
        h = x
        for b in blocks:
            h = checkpoint(b, h, use_reentrant=False) if ckpt else b(h)
        h.sum().backward()

    out = {}
    for ckpt in (False, True):
        run(ckpt)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        run(ckpt)
        torch.cuda.synchronize()
        key = "ckpt" if ckpt else "plain"
        out[key + "_peak_extra_gib"] = r((torch.cuda.max_memory_allocated() - base) / 2**30)
        out[key + "_ms"] = r(cuda_ms(lambda: run(ckpt), reps=5))
    out["act_per_block_mib_formula"] = r(8192 * (8192 * 2 + 2048) * 4 / 2**20, 0)  # 两份 8192 宽中间值 + 输入
    return out


# ---------------------------------------------------------------- 第 28 章
@measure
def ch28():
    """很多小算子：eager / compile / compile + CUDA Graph（reduce-overhead）/ 手写 CUDA Graph，每次调用微秒数。"""
    import ch28_compile_graphs as M

    x = torch.randn(64, device=DEV)

    def us(fn, n=1000):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return r((time.perf_counter() - t) / n * 1e6, 1)

    torch._dynamo.reset()
    c_default = torch.compile(M.many_small_ops)
    c_ro = torch.compile(M.many_small_ops, mode="reduce-overhead")
    out = dict(eager_us=us(lambda: M.many_small_ops(x)), compile_us=us(lambda: c_default(x)),
               reduce_overhead_us=us(lambda: c_ro(x)))
    static_x = x.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            M.many_small_ops(static_x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_y = M.many_small_ops(static_x)
    out["manual_graph_us"] = us(g.replay)
    torch.testing.assert_close(static_y, M.many_small_ops(x))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        M.many_small_ops(x)
        torch.cuda.synchronize()
    out["eager_kernels"] = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        c_default(x)
        torch.cuda.synchronize()
    out["compiled_kernels"] = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    # 第二种情况：16 层小 Linear（矩阵乘融合不了，compile 只能少几个 kernel），CUDA Graph 才显出优势
    mlp = nn.Sequential(*[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(16)]).to(DEV).eval()
    xm = torch.randn(8, 256, device=DEV)
    torch._dynamo.reset()
    with torch.no_grad():
        m_c, m_ro = torch.compile(mlp), torch.compile(mlp, mode="reduce-overhead")
        out.update(mlp_eager_us=us(lambda: mlp(xm)), mlp_compile_us=us(lambda: m_c(xm)), mlp_reduce_overhead_us=us(lambda: m_ro(xm)))
        sx = xm.clone()
        s2 = torch.cuda.Stream()
        s2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s2):
            for _ in range(3):
                mlp(sx)
        torch.cuda.current_stream().wait_stream(s2)
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            mlp(sx)
        out["mlp_manual_graph_us"] = us(g2.replay)
    return out


# ---------------------------------------------------------------- 第 29 章
@measure
def ch29():
    """融合：gelu(x+b)*s 前后的时间和有效带宽；LayerNorm (8192, 4096) bf16 的实测 vs 理论下限。"""
    import ch29_why_fusion as M

    n = 1 << 26
    x, b, s = (torch.randn(n, device=DEV) for _ in range(3))
    cf = torch.compile(M.f)
    te, tc = cuda_ms(lambda: M.f(x, b, s)), cuda_ms(lambda: cf(x, b, s))
    eager_bytes, fused_bytes = 8 * n * 4, 4 * n * 4  # eager：3 个 kernel 共读写 8 份；融合后读 3 写 1
    out = dict(n=n, eager_ms=r(te, 3), compiled_ms=r(tc, 3), speedup=r(te / tc),
               eager_eff_bw_tbs=r(eager_bytes / te / 1e9), fused_eff_bw_tbs=r(fused_bytes / tc / 1e9))
    ln = nn.LayerNorm(4096, device=DEV, dtype=torch.bfloat16)
    h = torch.randn(8192, 4096, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        t_ln = cuda_ms(lambda: ln(h), reps=50)
    bw = device_bw()
    out.update(layernorm_us=r(t_ln * 1e3, 1), layernorm_floor_us=r(2 * h.numel() * 2 / (bw * 1e12) * 1e6, 1),
               measured_bw_tbs=r(bw))
    return out


_BW = None


def device_bw():
    """实测显存带宽（TB/s）：o = x + y，读 2 份写 1 份，共 3 × 256 MiB；取 5 次里最好的（笔记本卡频率会浮动）。"""
    global _BW
    if _BW is None:
        x, y = torch.randn(64 * 2**20, device=DEV), torch.randn(64 * 2**20, device=DEV)
        o = torch.empty_like(x)
        best = min(cuda_ms(lambda: torch.add(x, y, out=o)) for _ in range(5))
        _BW = 3 * x.numel() * 4 / best / 1e9
    return _BW


# ---------------------------------------------------------------- 第 30 章
@measure
def ch30():
    """decode 上限 = 带宽 ÷ 权重字节：用 batch 1 的矩阵向量乘读 4 GiB bf16 权重，看实际 tok/s；batch 变大几乎不变慢。"""
    torch.manual_seed(0)
    layers = [torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16) for _ in range(32)]  # 32 × 128 MiB = 4 GiB
    wbytes = sum(w.numel() * 2 for w in layers)
    out = dict(weight_gib=r(wbytes / 2**30))
    for bs in (1, 8, 64, 512):
        h = torch.randn(bs, 8192, device=DEV, dtype=torch.bfloat16)

        def tok():
            y = h
            for w in layers:
                y = y @ w
            return y

        ms = cuda_ms(tok, reps=10)
        out[f"batch{bs}_ms_per_step"] = r(ms, 3)
        if bs == 1:
            out["batch1_eff_bw_tbs"] = r(wbytes / ms / 1e9)
            out["batch1_ceiling_tok_s"] = r(device_bw() * 1e12 / wbytes, 0)
            out["batch1_measured_tok_s"] = r(1e3 / ms, 0)
    out["7b_bf16_ceiling_tok_s_this_gpu"] = r(device_bw() * 1e12 / 14e9, 0)
    with torch.no_grad():
        kv = torch.empty(2, 32, 1, 8, 8192, 128, device=DEV, dtype=torch.bfloat16)  # K/V × 层 × batch × kv 头 × T × d
    out["kv_cache_gib_32L_8kvh_T8192"] = r(kv.numel() * 2 / 2**30)
    return out


# ---------------------------------------------------------------- 第 31 章
@measure
def ch31():
    """GPU 上 Inductor 生成的是 Triton kernel：数 kernel、看 autotune 配置、矩阵乘交给谁。"""
    from torch._inductor.utils import run_and_get_code

    def f(x, w, b):
        return F.gelu(x @ w + b) * 2

    x, w, b = torch.randn(512, 1024, device=DEV), torch.randn(1024, 1024, device=DEV), torch.randn(1024, device=DEV)
    torch._dynamo.reset()
    _, codes = run_and_get_code(torch.compile(f), x, w, b)
    code = "\n".join(codes)
    lines = code.splitlines()
    return dict(triton_kernels=code.count("@triton.jit"), uses_extern_mm=any("extern_kernels.mm" in ln or "extern_kernels.addmm" in ln
                                                                           for ln in lines),
                heuristics=sorted({ln.strip().split("(")[0] for ln in lines if ln.strip().startswith("@triton_heuristics")}),
                kernel_names=sorted({ln.split("=")[0].strip() for ln in lines if "async_compile.triton(" in ln}),
                code_lines=len(lines))


# ---------------------------------------------------------------- 第 32 章
@measure
def ch32():
    """真正的 Triton kernel：add_relu 和按行 softmax，与 PyTorch 对拍并比速度。"""
    import triton
    import triton.language as tl

    @triton.jit
    def add_relu_k(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask)
        tl.store(o_ptr + offs, tl.maximum(v, 0.0), mask=mask)

    @triton.jit
    def softmax_k(x_ptr, o_ptr, n_cols, stride, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        e = tl.exp(x)
        tl.store(o_ptr + row * stride + cols, e / tl.sum(e, axis=0), mask=mask)

    n = 1 << 26
    x, y = torch.randn(n, device=DEV), torch.randn(n, device=DEV)
    o = torch.empty_like(x)

    def tri_add():
        add_relu_k[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)

    tri_add()
    torch.testing.assert_close(o, torch.relu(x + y))
    t_tri, t_eager = cuda_ms(tri_add), cuda_ms(lambda: torch.relu(x + y))
    m = torch.randn(8192, 4000, device=DEV)
    om = torch.empty_like(m)

    def tri_sm():
        softmax_k[(m.shape[0],)](m, om, m.shape[1], m.stride(0), BLOCK=triton.next_power_of_2(m.shape[1]))

    tri_sm()
    torch.testing.assert_close(om, torch.softmax(m, -1))
    t_sm, t_sm_torch = cuda_ms(tri_sm), cuda_ms(lambda: torch.softmax(m, -1))
    bw = lambda nbytes, ms: r(nbytes / ms / 1e9)  # noqa: E731
    return dict(triton_version=triton.__version__, add_relu_triton_ms=r(t_tri, 3), add_relu_eager_ms=r(t_eager, 3),
                add_relu_triton_bw_tbs=bw(3 * n * 4, t_tri), add_relu_eager_bw_tbs=bw(5 * n * 4, t_eager),
                softmax_triton_ms=r(t_sm, 3), softmax_torch_ms=r(t_sm_torch, 3), softmax_block=triton.next_power_of_2(4000),
                softmax_triton_bw_tbs=bw(2 * m.numel() * 4, t_sm))


# ---------------------------------------------------------------- 第 33 章
@measure
def ch33():
    """triton_op：Inductor 能看见里面的 Triton kernel；对比 custom_op 黑盒，compile 后都不断图，opcheck 通过。"""
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    @triton.jit
    def k(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(o_ptr + offs, tl.maximum(tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), 0.0), mask=mask)

    @triton_op("ptdd_gpu::add_relu", mutates_args={})
    def add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        o = torch.empty_like(x)
        n = x.numel()
        wrap_triton(k)[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
        return o

    x, y = torch.randn(10_000, device=DEV), torch.randn(10_000, device=DEV)
    torch.library.opcheck(add_relu, (x, y), test_utils=("test_schema", "test_faketensor"))
    torch._dynamo.reset()
    expl = torch._dynamo.explain(lambda a, b: add_relu(a, b) * 2)(x, y)
    from torch._inductor.utils import run_and_get_code

    _, codes = run_and_get_code(torch.compile(lambda a, b: add_relu(a, b) * 2), x, y)
    code = "\n".join(codes)
    return dict(opcheck="通过", graph_breaks=expl.graph_break_count, user_kernel_in_inductor_code="def k(" in code or "k_0" in code,
                correct=bool(torch.allclose(add_relu(x, y), torch.relu(x + y))))


# ---------------------------------------------------------------- 第 34 章
@measure
def ch34():
    """C++ 扩展的 CUDA 版：load_inline 编一个 add_relu CUDA kernel（需要 nvcc），与 eager 比速度。"""
    from torch.utils.cpp_extension import load_inline

    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["CUDA_HOME"] + "/bin" + os.pathsep + os.environ["PATH"]
    cuda_src = r"""
#include <torch/extension.h>
__global__ void k(const float* a, const float* b, float* o, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) { float v = a[i] + b[i]; o[i] = v > 0.f ? v : 0.f; }
}
at::Tensor add_relu(at::Tensor x, at::Tensor y) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.sizes() == y.sizes());
  auto o = torch::empty_like(x);
  int64_t n = x.numel();
  k<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), o.data_ptr<float>(), n);
  return o;
}
"""
    t = time.time()
    mod = load_inline(name="ptdd_cuda_add_relu", cpp_sources="at::Tensor add_relu(at::Tensor x, at::Tensor y);",
                      cuda_sources=cuda_src, functions=["add_relu"], extra_cuda_cflags=["-O3"], verbose=False)
    build_s = time.time() - t
    n = 1 << 26
    x, y = torch.randn(n, device=DEV), torch.randn(n, device=DEV)
    torch.testing.assert_close(mod.add_relu(x, y), torch.relu(x + y))
    te, tc = cuda_ms(lambda: torch.relu(x + y)), cuda_ms(lambda: mod.add_relu(x, y))
    cf = torch.compile(lambda a, b: torch.relu(a + b))
    tcomp = cuda_ms(lambda: cf(x, y))
    return dict(build_s=r(build_s, 1), eager_ms=r(te, 3), cuda_ext_ms=r(tc, 3), compiled_ms=r(tcomp, 3),
                cuda_ext_bw_tbs=r(3 * n * 4 / tc / 1e9))


# ---------------------------------------------------------------- 第 35 章
@measure
def ch35():
    """torchao int8 权重量化在 GPU 上：batch 1 线性层（访存受限）提速多少；export + AOTInductor 生成 CUDA 包。"""
    import tempfile

    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    torch.manual_seed(0)
    m = nn.Sequential(*[nn.Linear(4096, 4096, bias=False) for _ in range(16)]).to(DEV, torch.bfloat16).eval()
    x = torch.randn(1, 4096, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = m(x)
        cm = torch.compile(m)
        t_bf16 = cuda_ms(lambda: cm(x), reps=50)
        quantize_(m, Int8WeightOnlyConfig())
        torch._dynamo.reset()
        cq = torch.compile(m)
        q = cq(x)
        t_int8 = cuda_ms(lambda: cq(x), reps=50)
    rel = ((q.float() - ref.float()).norm() / ref.float().norm()).item()
    m2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)).to(DEV).eval()
    xe = torch.randn(4, 64, device=DEV)
    ep = torch.export.export(m2, (xe,), dynamic_shapes={"input": {0: torch.export.Dim("batch", min=1, max=64)}})
    path = torch._inductor.aoti_compile_and_package(ep, package_path=os.path.join(tempfile.gettempdir(), "ptdd_gpu_ch35.pt2"))
    runner = torch._inductor.aoti_load_package(path)
    x5 = torch.randn(5, 64, device=DEV)
    return dict(bf16_compiled_ms=r(t_bf16, 3), int8_wo_compiled_ms=r(t_int8, 3), speedup=r(t_bf16 / t_int8),
                rel_err=r(rel, 4), weight_mib_bf16=r(16 * 4096 * 4096 * 2 / 2**20, 0), aoti_package_kb=r(os.path.getsize(path) / 1024, 0),
                aoti_matches=bool(torch.allclose(runner(x5), m2(x5), atol=1e-5)))


# ---------------------------------------------------------------- 第 36 章
@measure
def ch36():
    """测速的坑在 GPU 上：不 sync 的 time.perf_counter 只量到下单；CUDA Event / triton do_bench 才是真时间。"""
    from triton.testing import do_bench

    a = torch.randn(8192, 8192, device=DEV)
    a @ a
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        a @ a
    no_sync = (time.perf_counter() - t) / 10 * 1e3
    torch.cuda.synchronize()
    with_sync = 0.0
    t = time.perf_counter()
    for _ in range(10):
        a @ a
    torch.cuda.synchronize()
    with_sync = (time.perf_counter() - t) / 10 * 1e3
    first = torch.randn(4096, 4096, device=DEV)
    torch.cuda.synchronize()
    t = time.perf_counter()
    torch.linalg.qr(first)  # 第一次调用含库加载 / 初始化
    torch.cuda.synchronize()
    cold = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    torch.linalg.qr(first)
    torch.cuda.synchronize()
    warm = (time.perf_counter() - t) * 1e3
    return dict(no_sync_ms=r(no_sync, 3), with_sync_ms=r(with_sync), cuda_event_ms=r(cuda_ms(lambda: a @ a, reps=10)),
                do_bench_ms=r(do_bench(lambda: a @ a)), qr_first_call_ms=r(cold, 1), qr_second_call_ms=r(warm, 1))


# ---------------------------------------------------------------- 第 17 / 18 章（单卡上能做的部分）
@measure
def ch17():
    """单卡机器上 NCCL 能做什么：world=1 跑通；两个进程挤一张卡会被 NCCL 拒绝（多卡并行要多卡机器）。"""
    import torch.distributed as dist
    import torch.multiprocessing as mp

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29533", RANK="0", WORLD_SIZE="1")
    dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
    t = torch.ones(4, device=DEV)
    dist.all_reduce(t)
    dist.destroy_process_group()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_two_rank_one_gpu, args=(rank, q)) for rank in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(60)
        if p.is_alive():
            p.kill()
    msgs = []
    while not q.empty():
        msgs.append(q.get())
    return dict(nccl_version=".".join(map(str, torch.cuda.nccl.version())), world1_all_reduce=t.tolist(),
                two_ranks_same_gpu=sorted(set(msgs)) or ["超时（卡住）"])


def _two_rank_one_gpu(rank, q):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29534", RANK=str(rank), WORLD_SIZE="2")
    try:
        dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
        t = torch.ones(4, device="cuda:0") * (rank + 1)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        q.put(f"跑通 {t.tolist()}")
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        q.put(f"{type(e).__name__}: {str(e).splitlines()[0][:120]}")


def main(names):
    if not torch.cuda.is_available():
        print("没有 CUDA，跳过（Mac 上的版本见 chNN_*.py）")
        return
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_results.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name in names or MEASURES:
        print(f"\n== {name}: {MEASURES[name].__doc__.splitlines()[0]}", flush=True)
        try:
            res[name] = MEASURES[name]()
        except Exception as e:  # noqa: BLE001
            res[name] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
        for k, v in res[name].items():
            print(f"   {k}: {v}", flush=True)
        torch.cuda.empty_cache()
    res["_meta"] = dict(date=time.strftime("%Y-%m-%d"), host=os.uname().nodename)
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    print("\n写入", path)


if __name__ == "__main__":
    main(sys.argv[1:])
