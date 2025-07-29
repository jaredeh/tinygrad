from tinygrad.renderer.rust import RustRenderer
from tinygrad.helpers import from_mv, cpu_profile
from tinygrad.device import Compiler
from tinygrad.runtime.ops_cpu import CPUAllocator, CPUComputeQueue, CPUProgram
from tinygrad.runtime.support.hcq import HCQCompiled, HCQBuffer, HCQArgsState, HCQSignal, HCQProgram, MMIOInterface
import subprocess, tempfile, pathlib, ctypes, sys, functools

class RustJITCompiler(Compiler):
  def __init__(self, cachekey="compile_rust_jit"): super().__init__(cachekey)

  def compile_file(self, cmd:str, src:str) -> bytes:
    # TODO: remove file write. sadly rustc doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output((cmd+str(output_file.name)).split(), input=(src).encode('utf-8'))
      output = pathlib.Path(output_file.name).read_bytes()
    return output

  def compile(self, src: str) -> bytes:
    output = self.compile_file("rustc --edition=2021 -Aunused_parens -Aunused_mut -Aunused-variables -C opt-level=3  -C target-cpu=native -C debuginfo=0 --crate-type=cdylib - -o ", src)
    # print(f"compile len={len(output)}")
    # with open(f"/tmp/blarp", "wb") as f:
    #   f.write(output)
    return output

class CDLLStyleProgram(HCQProgram):
  def __init__(self, dev, name:str, lib:bytes):
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]
    super().__init__(HCQArgsState, dev, name, kernargs_alloc_size=0)

  def __del__(self):
    if getattr(sys, 'is_finalizing', lambda: True)(): return
    if sys.platform == 'win32': ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.mem), ctypes.c_size_t(0), 0x8000) #0x8000 - MEM_RELEASE

class RustAllocator(CPUAllocator):
  def _copyin(self, dest, src:memoryview):
    with cpu_profile('TINY -> RUST', self.dev.device, is_copy=True): ctypes.memmove(dest.va_addr, from_mv(src), len(src))
  def _copyout(self, dest:memoryview, src):
    with cpu_profile('RUST -> TINY', self.dev.device, is_copy=True): ctypes.memmove(from_mv(dest), src.va_addr, len(dest))
  def _map(self, buf:HCQBuffer):
    if buf.view is None or not isinstance(buf.view, MMIOInterface): raise RuntimeError("Cannot map buffer without view to cpu")

class RustDevice(HCQCompiled):
  def __init__(self, device:str=""):
    super().__init__(device, RustAllocator(self), RustRenderer(), RustJITCompiler(), functools.partial(CDLLStyleProgram, self), HCQSignal, CPUComputeQueue,
                     supports_graph=False)
