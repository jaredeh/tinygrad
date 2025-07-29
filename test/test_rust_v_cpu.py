import os
os.environ["CLANG"] = "1"  # Ensure CPU uses CLANG
os.environ["RUST"] = "1"   # Enable RUST backend
import unittest
import time
import numpy as np
np.set_printoptions(linewidth=160)
from tinygrad import Tensor, Device, GlobalCounters, TinyJit
from tinygrad.nn import Conv2d
from tinygrad.helpers import colorize_float, getenv, CI

IN_CHANS = [int(x) for x in getenv("IN_CHANS", "4,16,64").split(",")]

save_ops, save_mem = 0, 0
CNT = getenv("CNT", 8)
def helper_test_speed(f1, *args, device="CPU"):
  global save_ops, save_mem
  ets = []
  ret = None
  cache_defeat = np.zeros((2048,2048))
  Device.DEFAULT = device  # Set device for this run
  for i in range(CNT):
    del ret

    # operation cache defeats
    args = [(x+1).realize() if isinstance(x, Tensor) else (None if x is None else (x+1)) for x in args]
    args = [(x-1).realize() if isinstance(x, Tensor) else (None if x is None else (x-1)) for x in args]

    # Force syncing and convert to numpy
    [x.numpy() for x in args if isinstance(x, Tensor)]

    # Clear 32MB global memory cache
    cache_defeat += 1

    # Synchronize device
    Device[device].synchronize()

    GlobalCounters.global_ops = 0
    GlobalCounters.global_mem = 0
    st = time.perf_counter()
    ret = f1(*args)
    if isinstance(ret, Tensor):
      #ret.realize()  # Ensure computation is executed
      Device[device].synchronize()
    et = (time.perf_counter() - st) * 1000
    if i >= 1: ets.append(et)
    if GlobalCounters.global_ops:
      save_ops, save_mem = GlobalCounters.global_ops, GlobalCounters.global_mem
  return ret.numpy() if isinstance(ret, Tensor) else ret.cpu().numpy(), np.min(ets)

def helper_test_generic_square(name, N, f1, f2, onearg=False):
  np.random.seed(0)  # Ensure reproducibility
  data_a = np.random.rand(N, N) - 0.5
  data_b = np.random.rand(N, N) - 0.5 if not onearg else None

  # Create input tensors for CPU
  Device.DEFAULT = "CPU"
  cpu_a = Tensor(data_a)
  cpu_b = Tensor(data_b) if not onearg else None

  # Create input tensors for RUST
  Device.DEFAULT = "RUST"
  rust_a = Tensor(data_a)
  rust_b = Tensor(data_b) if not onearg else None

  helper_test_generic(f"{name:30s} {N:5d}x{N:5d}", TinyJit(f1), (cpu_a, cpu_b), TinyJit(f2), (rust_a, rust_b))

def helper_test_matvec(name, N, M):
  np.random.seed(0)  # Ensure reproducibility
  data_a = np.random.rand(N) - 0.5  # Vector of shape (N,)
  data_b = np.random.rand(N, M) - 0.5  # Matrix of shape (N, M)

  # Create input tensors for CPU
  Device.DEFAULT = "CPU"
  cpu_a = Tensor(data_a)
  cpu_b = Tensor(data_b)

  # Create input tensors for RUST
  Device.DEFAULT = "RUST"
  rust_a = Tensor(data_a)
  rust_b = Tensor(data_b)

  helper_test_generic(f"{name:30s} {N:5d}x{M:5d}", TinyJit(lambda a, b: a @ b), (cpu_a, cpu_b), TinyJit(lambda a, b: a @ b), (rust_a, rust_b))

prefix = None
def helper_test_generic(name, f1, f1_args, f2, f2_args):
  global prefix
  val_cpu, et_cpu = helper_test_speed(f1, *f1_args, device="CPU")
  val_rust, et_rust = helper_test_speed(f2, *f2_args, device="RUST")

  desc = "faster" if et_cpu > et_rust else "slower"
  flops = save_ops*1e-6
  mem = save_mem*1e-6
  print(("\r" if not CI else "") + f"{name:42s} {et_cpu:7.2f} ms ({flops/et_cpu:9.2f} GFLOPS {mem/et_cpu:7.2f} GB/s) in CLANG, {et_rust:7.2f} ms ({flops/et_rust:9.2f} GFLOPS {mem/et_rust:7.2f} GB/s) in RUST, {colorize_float(et_rust/et_cpu)} {desc} {flops:10.2f} MOPS {mem:8.2f} MB")
  np.testing.assert_allclose(val_cpu, val_rust, atol=1e-3, rtol=1e-3)

def helper_test_conv(bs, in_chans, out_chans, kernel_size, img_size_y, img_size_x):
  np.random.seed(0)  # Ensure reproducibility
  data_dat = np.random.rand(bs, in_chans, img_size_y, img_size_x) - 0.5
  data_weight = np.random.rand(out_chans, in_chans, kernel_size, kernel_size) - 0.5

  # Create input tensors and Conv2d for CPU
  Device.DEFAULT = "CPU"
  cpu_dat = Tensor(data_dat)
  cpu_conv = Conv2d(in_chans, out_chans, kernel_size, bias=None)
  cpu_conv.weight = Tensor(data_weight)

  # Create input tensors and Conv2d for RUST
  Device.DEFAULT = "RUST"
  rust_dat = Tensor(data_dat)
  rust_conv = Conv2d(in_chans, out_chans, kernel_size, bias=None)
  rust_conv.weight = Tensor(data_weight)

  def f_cpu(dat): return cpu_conv(dat).realize()
  def f_rust(dat): return rust_conv(dat).realize()
  helper_test_generic(f"conv bs:{bs:3d} chans:{in_chans:3d} -> {out_chans:3d} k:{kernel_size}", TinyJit(f_cpu), (cpu_dat,), TinyJit(f_rust), (rust_dat,))

@unittest.skipIf(getenv("BIG") == 0, "no big tests")
@unittest.skipIf(getenv("MOCKGPU"), "no MOCKGPUs")
class TestBigSpeed(unittest.TestCase):
  def test_add(self):
    def f(a, b): return a+b
    helper_test_generic_square('add', 8192, f, f)
  def test_exp(self):
    def f(a, b): return a.exp()
    helper_test_generic_square('exp', 8192, f, f, onearg=True)
  def test_gemm_2048(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 2048, f, f)
  def test_gemm_4096(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 4096, f, f)
  def test_large_conv_1x1(self): helper_test_conv(bs=32, in_chans=128, out_chans=128, kernel_size=1, img_size_y=128, img_size_x=128)
  def test_large_conv_3x3(self): helper_test_conv(bs=4, in_chans=128, out_chans=128, kernel_size=3, img_size_y=130, img_size_x=130)
  def test_large_conv_5x5(self): helper_test_conv(bs=4, in_chans=128, out_chans=128, kernel_size=5, img_size_y=132, img_size_x=132)
  def test_matvec_4096_16384(self): helper_test_matvec('matvec_4096_16384', 4096, 16384)
  def test_matvec_16384_4096(self): helper_test_matvec('matvec_16384_4096', 16384, 4096)

@unittest.skipIf(getenv("BIG") == 1, "only big tests")
@unittest.skipIf(getenv("MOCKGPU"), "no MOCKGPUs")
class TestSpeed(unittest.TestCase):
  def test_sub(self):
    def f(a, b): return a-b
    helper_test_generic_square('sub', 4096, f, f)

  def test_pow(self):
    def f(a, b): return a.pow(b)
    helper_test_generic_square('pow', 2048, f, f)

  def test_sum(self):
    def f(a, b): return a.sum()
    helper_test_generic_square('sum', 2048, f, f, onearg=True)
    helper_test_generic_square('sum', 4096, f, f, onearg=True)

  def test_partial_sum(self):
    R = 256
    def f(a, b): return a.reshape(int(4096//R), int(4096*R)).sum(axis=1)
    helper_test_generic_square('partial_sum', 4096, f, f, onearg=True)

  @unittest.skip("not really used in models")
  def test_cumsum(self):
    def f0(a, b): return a.cumsum(axis=0)
    def f1(a, b): return a.cumsum(axis=1)
    helper_test_generic_square('cumsum_0', 256, f0, f0, onearg=True)
    helper_test_generic_square('cumsum_1', 256, f1, f1, onearg=True)

  def test_cat(self):
    helper_test_generic_square('cat_0', 2048, lambda x,y: x.cat(y,dim=0), lambda x,y: x.cat(y,dim=0))
    helper_test_generic_square('cat_1', 2048, lambda x,y: x.cat(y,dim=1), lambda x,y: x.cat(y,dim=1))

  def test_array_packing(self):
    N = 2048
    def f(a, b): return a.reshape(N, N // 32, 32).permute(1,0,2).contiguous()
    helper_test_generic_square('array_packing', N, f, f, onearg=True)

  def test_permute(self):
    for N in [1024, 4096]:
      # this is a 64MB tensor, M1 L1 cache is 128kB
      # to fit easily in L1, rotations should be 128x128 chunks. 128x128 is also the AMX size
      def f(a, b): return a.permute(1,0).contiguous()
      helper_test_generic_square('permute', N, f, f, onearg=True)

  def test_double_permute(self):
    N = 64
    np.random.seed(0)  # Ensure reproducibility
    data_a = np.random.rand(N, N, N, N) - 0.5

    Device.DEFAULT = "CPU"
    cpu_a = Tensor(data_a)
    Device.DEFAULT = "RUST"
    rust_a = Tensor(data_a)

    def f(a): return a.permute(1,0,3,2).contiguous()
    helper_test_generic(f"double_permute {N},{N},{N},{N}", TinyJit(f), (cpu_a,), TinyJit(lambda a: f(a).realize()), (rust_a,))

  def test_neg(self):
    def f(a, b): return -a
    helper_test_generic_square('neg', 4096, f, f, onearg=True)

  def test_exp(self):
    def f(a, b): return a.exp()
    helper_test_generic_square('exp', 2048, f, f, onearg=True)

  def test_sqrt(self):
    def f(a, b): return a.sqrt()
    helper_test_generic_square('sqrt', 2048, f, f, onearg=True)

  def test_relu(self):
    def f(a, b): return a.relu()
    helper_test_generic_square('relu', 4096, f, f, onearg=True)

  def test_max(self):
    def f(a, b): return a.max()
    helper_test_generic_square('max', 4096, f, f, onearg=True)

  def test_mul_sum(self):
    def f(a, b): return (a*b).sum()
    helper_test_generic_square('mul_sum', 4096, f, f)

  def test_add_a(self):
    def f(a, b): return a + b
    helper_test_generic_square('add', 1, f, f)

  def test_add_big(self):
    for N in [1024, 4096]:
      def f(a, b): return a + b
      helper_test_generic_square('add', N, f, f)

  def test_add_constant(self):
    def f(a, b): return a+2.0
    helper_test_generic_square('add_constant', 4096, f, f, onearg=True)

  def test_add_sq(self):
    def f(a, b): return a*a + b*b
    helper_test_generic_square('add_sq', 4096, f, f)

  def test_gemm(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 1024, f, f)

  def test_gemm_medium(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 512, f, f)

  def test_gemm_small(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 256, f, f)

  def test_gemm_unrolled(self):
    N = 512
    def f1(a, b): return a@b.T
    def f2(a, b): return (a.reshape(N, 1, N).expand(N, N, N) * b.reshape(1, N, N).expand(N, N, N)).sum(axis=2)
    helper_test_generic_square('gemm_unrolled', N, f1, f2)

  def test_gemm_unrolled_permute_l(self):
    N = 512
    def f1(a, b): return a.T@b.T
    def f2(a, b): return (a.permute(1,0).reshape(N, 1, N).expand(N, N, N) * b.reshape(1, N, N).expand(N, N, N)).sum(axis=2)
    helper_test_generic_square('gemm_unrolled_permute_l', N, f1, f2)

  def test_gemm_unrolled_permute_r(self):
    N = 512
    def f1(a, b): return a@b
    def f2(a, b): return (a.reshape(N, 1, N).expand(N, N, N) * b.permute(1,0).reshape(1, N, N).expand(N, N, N)).sum(axis=2)
    helper_test_generic_square('gemm_unrolled_permute_r', N, f1, f2)

  def test_gemm_unrolled_permute_lr(self):
    N = 512
    def f1(a, b): return a.T@b
    def f2(a, b): return (a.permute(1,0).reshape(N, 1, N).expand(N, N, N) * b.permute(1,0).reshape(1, N, N).expand(N, N, N)).sum(axis=2)
    helper_test_generic_square('gemm_unrolled_permute_lr', N, f1, f2)

  def test_matvec_1024_1024(self): helper_test_matvec('matvec_1024_1024', 1024, 1024)
  def test_matvec_1024_4096(self): helper_test_matvec('matvec_1024_4096', 1024, 4096)
  def test_matvec_4096_1024(self): helper_test_matvec('matvec_4096_1024', 4096, 1024)
  def test_matvec_4096_4096(self): helper_test_matvec('matvec_4096_4096', 4096, 4096)

  def test_openpilot_conv2d(self):
    bs, in_chans, out_chans = 1,12,32
    np.random.seed(0)
    data_dat = np.random.rand(bs, 64, 128, 12) - 0.5
    data_weight = np.random.rand(out_chans, in_chans, 3, 3) - 0.5

    Device.DEFAULT = "CPU"
    cpu_dat = Tensor(data_dat)
    cpu_conv = Conv2d(in_chans, out_chans, 3, bias=None, padding=1)
    cpu_conv.weight = Tensor(data_weight)

    Device.DEFAULT = "RUST"
    rust_dat = Tensor(data_dat)
    rust_conv = Conv2d(in_chans, out_chans, 3, bias=None, padding=1)
    rust_conv.weight = Tensor(data_weight)

    def f_cpu(dat): return cpu_conv(dat.permute(0,3,1,2)).realize()
    def f_rust(dat): return rust_conv(dat.permute(0,3,1,2)).realize()
    helper_test_generic(f"conv bs:{bs:3d} chans:{in_chans:3d} -> {out_chans:3d} k:3", TinyJit(f_cpu), (cpu_dat,), TinyJit(f_rust), (rust_dat,))

  def test_conv2d(self):
    for bs in [32]:
      for in_chans in IN_CHANS:
        for out_chans in [32]:
          helper_test_conv(bs, in_chans, out_chans, 3, 34, 34)

if __name__ == '__main__':
  unittest.main()
