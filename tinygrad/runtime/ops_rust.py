from tinygrad.renderer.rust import RustRenderer
from tinygrad.helpers import from_mv, cpu_profile, capstone_flatdump, OSX, mv_address
from tinygrad.device import Compiler
from tinygrad.runtime.ops_cpu import CPUAllocator, CPUComputeQueue, CPUProgram
from tinygrad.runtime.support.hcq import HCQCompiled, HCQBuffer, HCQSignal, MMIOInterface

import subprocess, ctypes, functools, tempfile, pathlib, os
from tinygrad.runtime.support.elf import jit_loader

class RustJITCompiler(Compiler):
  def __init__(self, cachekey="compile_rustc_jit"): super().__init__(cachekey)

  def compile(self, src: str) -> bytes:
    # NOTE: rustc writes intermedidate files here under the hood even though we're using stdout/stdin here
    # so we need to use a temporary directory to avoid conflicts when running multiple instances of rustc in parallel
    with tempfile.TemporaryDirectory(delete=True) as tmpdir:
      env = os.environ.copy()
      env["TMPDIR"] = tmpdir
      args = ["--edition=2021", "-Aunused_parens", "-Aunused_mut", "-Aunused-variables", "-C", "opt-level=3", "-C", "target-cpu=native", "-C", "debuginfo=0", "-C", "panic=abort"]
      obj = subprocess.check_output(['rustc', *(args + [ "--crate-type=cdylib", "--emit=obj"]), '-o', '-', '-'], input=src.encode('utf-8'), env=env, cwd=tmpdir)
      try:
        j = jit_loader(obj)
      except Exception as e:
        # Probably because we have unresolvable symbols like intrinsics
        # fall back to cdylib build
        obj = subprocess.check_output(['rustc', *(args + [ "--crate-type=cdylib"]), '-o', '-', '-'], input=src.encode('utf-8'), env=env, cwd=tmpdir)
        if os.environ.get("RUSTDEBUG", False): print(f"Error loading jit: {e}")
        j = obj + b"RUSTCDYLIB"
      # print(f"Compiled {len(obj)} bytes")
      # print(f"obj[0:128]: {obj[:128]}")
      # with open("test.o", "wb") as f:
      #   f.write(obj)
      # with open("testj.o", "wb") as f:
      #   f.write(j)
    return j

  # TODO: detect RUSTCDYLIB and disassemble with llvm-objdump -d ?
  def disassemble(self, lib:bytes): return capstone_flatdump(lib)

class RustAllocator(CPUAllocator):
  def _copyin(self, dest, src:memoryview):
    with cpu_profile('TINY -> RUST', self.dev.device, is_copy=True): ctypes.memmove(dest.va_addr, from_mv(src), len(src))
  def _copyout(self, dest:memoryview, src):
    with cpu_profile('RUST -> TINY', self.dev.device, is_copy=True): ctypes.memmove(from_mv(dest), src.va_addr, len(dest))
  def _map(self, buf:HCQBuffer):
    if buf.view is None or not isinstance(buf.view, MMIOInterface): raise RuntimeError("Cannot map buffer without view to cpu")


class RustProgram(CPUProgram):
  def __init__(self, dev, name:str, lib:bytes):
    super().__init__(dev, name, lib)
    # Hack for unresolved llvm intrinsics, overriding standard ctypes.CFUNCTYPE call with CDLL
    # Haven't figured out how to avoid this yet
    if lib.endswith(b"RUSTCDYLIB"):
      trimmedlib = lib[:-12]
      # write to disk so we can load it
      with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
        pathlib.Path(cached_file_path.name).write_bytes(trimmedlib)
        self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]


class RustDevice(HCQCompiled):
  def __init__(self, device:str=""):
    super().__init__(device, RustAllocator(self), RustRenderer(), RustJITCompiler(), functools.partial(RustProgram, self), HCQSignal, CPUComputeQueue,
                     supports_graph=False)
