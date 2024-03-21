import ctypes, subprocess, pathlib, tempfile
from tinygrad.device import Compiled, MallocAllocator, Compiler
from tinygrad.helpers import cpu_time_execution
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.rust import uops_to_rust, RustLanguage
from tinygrad.ops import LazyOp

import functools

RUST_PROGRAM_HEADER = ''

class RustCompiler(Compiler):
  linearizer_opts = LinearizerOptions("RUST", supports_float4=False, has_local=False)
  def render(self, name:str, uops) -> str: return uops_to_rust(RustLanguage(), name, uops)
  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=False) as output_file:
      # print(src)
      command = ('rustc -O --crate-type=cdylib - -o '+str(output_file.name)).split()
      input = (RUST_PROGRAM_HEADER+src).encode('utf-8')
      subprocess.check_output(command, input=input)
      # print(f"command: {command}")
      # print(f"output: {output_file.name}")
      return pathlib.Path(output_file.name).read_bytes()


class RustProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib = name, lib
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]

  def __call__(self, *bufs, vals=(), wait=False):
    print(f"calling {self.name} with {len(bufs)} buffers")
    i = 0
    for buf in bufs:
      print(f"  buf_{i} {buf}")
      i = i+1
    print(f"  vals {vals}")
    return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)
#  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class RustDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, RustCompiler("compile_rust"), RustProgram)
