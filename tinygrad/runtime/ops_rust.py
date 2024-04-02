from tinygrad.device import Compiled, MallocAllocator, Compiler, CompilerOptions, CDLLStyleProgram
from tinygrad.renderer.rust import uops_to_rust, RustLanguage

RUST_COMPILE_CMD = 'rustc -Aunused_parens -Aunused_mut -C opt-level=3 -C target-cpu=native -C debuginfo=0 --crate-type=cdylib - -o '

class RustCompiler(Compiler):
  compiler_opts = CompilerOptions("RUST", supports_float4=False, has_local=False)
  def render(self, name:str, uops, bufsizes=[]) -> str: return uops_to_rust(RustLanguage(), name, uops, bufsizes)
  def compile(self, src:str) -> bytes: return self.compile_file(RUST_COMPILE_CMD, src)

RustProgram = CDLLStyleProgram

class RustDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, RustCompiler("compile_rust"), RustProgram, has_bufsz=True)
