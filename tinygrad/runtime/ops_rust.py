from tinygrad.renderer.rust import RustRenderer
from tinygrad.device import Compiled, Compiler, MallocAllocator, CPUProgram
import subprocess, tempfile, pathlib, ctypes

class RustCompiler(Compiler):
  def compile_file(self, cmd:str, src:str) -> bytes:
    # TODO: remove file write. sadly clang/rustc doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=False) as output_file:
      subprocess.check_output((cmd+str(output_file.name)).split(), input=(src).encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

  def compile(self, src: str) -> bytes:
    output = self.compile_file("rustc -Aunused_parens -Aunused_mut -Aunused-variables -C opt-level=3 -C target-cpu=native -C debuginfo=0 --crate-type=cdylib - -o ", src)
    # print(f"compile len={len(output)}")
    # with open(f"/tmp/blarp", "wb") as f:
    #   f.write(output)
    return output

class CDLLStyleProgram(CPUProgram):
  def __init__(self, name:str, lib:bytes):
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]

class RustDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, RustRenderer(), RustCompiler(), CDLLStyleProgram)

