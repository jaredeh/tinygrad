import subprocess, ctypes, tempfile, pathlib
from tinygrad.device import Compiled, Compiler, MallocAllocator, CPUProgram, MAP_JIT
from tinygrad.renderer.rust import RustRenderer
from tinygrad.helpers import capstone_flatdump, mv_address
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

class CDLLStyleProgram(CPUProgram):
  def __init__(self, name:str, lib:bytes):
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]

class RustProgram(CPUProgram):
  def __init__(self, name:str, lib:bytes):
    OSX=False
    from mmap import mmap, PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE
    # On apple silicon with SPRR enabled (it always is in macos) RWX pages are unrepresentable: https://blog.svenpeter.dev/posts/m1_sprr_gxf/
    # MAP_JIT allows us to easily flip pages from RW- to R-X and vice versa. It is a noop on intel cpus. (man pthread_jit_write_protect_np)
    self.mem = mmap(-1, len(lib), MAP_ANON | MAP_PRIVATE | (MAP_JIT if OSX else 0), PROT_READ | PROT_WRITE | PROT_EXEC)

    if OSX: CPUProgram.rt_lib.pthread_jit_write_protect_np(False)
    self.mem.write(lib)
    if OSX: CPUProgram.rt_lib.pthread_jit_write_protect_np(True)

    # __clear_cache isn't a normal libc function, but a compiler support routine found in libgcc_s for gcc and compiler-rt for clang.
    # libgcc_s comes as shared library but compiler-rt is only a bunch of static library archives which we can't directly load, but fortunately
    # it somehow found its way into libSystem on macos (likely because it used __builtin_clear_cache) and libgcc_s is ~always present on linux
    # Using ["name"] instead of .name because otherwise name is getting mangled: https://docs.python.org/3.12/reference/expressions.html#index-5
    #CPUProgram.rt_lib["__clear_cache"](ctypes.c_void_p(mv_address(self.mem)), ctypes.c_void_p(mv_address(self.mem) + len(lib)))

    self.fxn = ctypes.CFUNCTYPE(None)(mv_address(self.mem))

class RustDevice(Compiled):
  #def __init__(self, device:str): super().__init__(device, MallocAllocator, RustRenderer(), RustJITCompiler(), CDLLStyleProgram)
  def __init__(self, device:str): super().__init__(device, MallocAllocator, RustRenderer(), RustJITCompiler(), CPUProgram)
