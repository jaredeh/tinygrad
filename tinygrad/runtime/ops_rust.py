from tinygrad.renderer.rust import RustRenderer
from tinygrad.helpers import from_mv, cpu_profile, capstone_flatdump
from tinygrad.device import Compiler
from tinygrad.runtime.ops_cpu import CPUAllocator, CPUComputeQueue, CPUProgram
from tinygrad.runtime.support.hcq import HCQCompiled, HCQBuffer, HCQSignal, MMIOInterface
import subprocess, ctypes, functools
from tinygrad.runtime.support.elf import jit_loader

class RustJITCompiler(Compiler):
  def __init__(self, cachekey="compile_rustc_jit"): super().__init__(cachekey)

  def compile(self, src: str) -> bytes:
    args = ["--edition=2021", "-Aunused_parens", "-Aunused_mut", "-Aunused-variables", "-C", "opt-level=3", "-C", "target-cpu=native", "-C", "debuginfo=0", "--crate-type=cdylib", "--emit=obj",
            "-C", "panic=abort"]
    obj = subprocess.check_output(['rustc', *args, '-o', '-', '-'], input=src.encode('utf-8'))
    print(f"Compiled {len(obj)} bytes")
    print(f"obj[0:128]: {obj[:128]}")
    with open("test.o", "wb") as f:
      f.write(obj)
    j = jit_loader(obj)
    with open("testj.o", "wb") as f:
      f.write(j)
    return j

  def disassemble(self, lib:bytes): return capstone_flatdump(lib)

class RustAllocator(CPUAllocator):
  def _copyin(self, dest, src:memoryview):
    with cpu_profile('TINY -> RUST', self.dev.device, is_copy=True): ctypes.memmove(dest.va_addr, from_mv(src), len(src))
  def _copyout(self, dest:memoryview, src):
    with cpu_profile('RUST -> TINY', self.dev.device, is_copy=True): ctypes.memmove(from_mv(dest), src.va_addr, len(dest))
  def _map(self, buf:HCQBuffer):
    if buf.view is None or not isinstance(buf.view, MMIOInterface): raise RuntimeError("Cannot map buffer without view to cpu")

class RustDevice(HCQCompiled):
  def __init__(self, device:str=""):
    super().__init__(device, RustAllocator(self), RustRenderer(), RustJITCompiler(), functools.partial(CPUProgram, self), HCQSignal, CPUComputeQueue,
                     supports_graph=False)
