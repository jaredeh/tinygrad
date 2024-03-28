import ctypes, subprocess, pathlib, tempfile
from tinygrad.device import Compiled, MallocAllocator, Compiler
from tinygrad.helpers import cpu_time_execution, prod, to_function_name
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.rust import uops_to_rust, RustLanguage
from tinygrad.ops import LazyOp, get_lazyop_info
from tinygrad.device import CompiledASTRunner
from tinygrad.lazy import LazyBuffer
from tinygrad.codegen.linearizer import Linearizer
from typing import Tuple
import functools

RUST_PROGRAM_HEADER = ''

class RustCompiler(Compiler):
  linearizer_opts = LinearizerOptions("RUST", supports_float4=False, has_local=False)
  def render(self, name:str, uops, outputs=[], inputs=[]) -> str:
    a = uops_to_rust(RustLanguage(), name, uops, outputs, inputs)
    print(a, file=open(f"/tmp/tinygrad/{name}.rs", "w"))
    return a
  def compile(self, src:str) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False) as output_file:
      command = (f"rustc -Aunused_parens -Aunused_mut -O --crate-type=cdylib - -o {str(output_file.name)}").split()
      input = (RUST_PROGRAM_HEADER+src).encode('utf-8')
      subprocess.check_output(command, input=input)
      return pathlib.Path(output_file.name).read_bytes()

class RustProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib = name, lib
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]
      print(f"RustProgram__init__ called with name={name}, cached_file_path={cached_file_path.name}")
  #def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)
  def __call__(self, *bufs, vals=(), wait=False):
    print(f"RustProgram.__call__ called with fxn={self.fxn} bufs={bufs}, vals={vals}, wait={wait}")
    return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)


class RustDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, RustCompiler("compile_rust"), RustProgram)

  def to_program(self, k:Linearizer, outputs:Tuple[LazyBuffer,...]=[], inputs:Tuple[LazyBuffer,...]=[]) -> CompiledASTRunner:
    assert self.compiler is not None, "compiler is required to run AST"
    k.linearize()
    info = get_lazyop_info(k.ast[0])
    ops, mem = k.uops.flops_mem()
    run_count = prod((k.global_size if k.global_size else []) + (k.local_size if k.local_size else []))
    # NOTE: we use min here to ignore the indexing FLOPS
    ret = CompiledASTRunner(k.name, self.compiler.render(to_function_name(k.name), k.uops, outputs=outputs, inputs=inputs), self.dname, k.global_size, k.local_size,
                            k.uops.vars(), min(info.flops, ops * run_count), min(info.mem_estimate, mem * run_count), outcount=len(k.outbufs))
    return ret

  @functools.lru_cache(None)    # pylint: disable=method-cache-max-size-none
  def get_runner(self, *ast:LazyOp, outputs:Tuple[LazyBuffer,...], inputs:Tuple[LazyBuffer,...]) -> CompiledASTRunner: return self.to_program(self.get_linearizer(*ast), outputs=outputs, inputs=inputs)
