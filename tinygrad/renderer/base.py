from typing import DefaultDict, Dict, List, Union, NamedTuple, Set, Optional
from collections import defaultdict, Counter
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.codegen.uops import UOpGraph
from tinygrad.helpers import strip_parens, getenv, prod


class BaseLanguage(NamedTuple):
  kernel_prefix: str = ""
  barrier: str = ""
  indent: str = ""
  variable_prefix: str = ""
  buffer_suffix: str = ""
  end_token: Optional[str] = None
  load_global: bool = False

  def render_if(self, u) -> str: raise NotImplementedError()
  def render_kernel(self, kernel, function_name, bufs, regs) -> str: raise NotImplementedError()

  def kk(self,*s:str): self.kernel.append("\n".join([f"{self.indent*self.depth}{t}" for t in s]))

  def reset(self,  uops:UOpGraph, outputs, inputs):
    self.kernel:List[str] = []
    self.bufs = []
    self.c:DefaultDict[str, int] = defaultdict(int)
    self.r: Dict[UOp, Union[List[str], str]] = {}
    self.bufsizes = [b.size for b in outputs+inputs]
    self.child_count = Counter(v for ru in uops for v in ru.vin)
    self.depth = 1

  def uop_loop(self, u):
      self.kk(self.render_loop(u))
      self.depth += 1

  def uop_define_global(self, u):
    assert len(self.bufs) == u.arg[0], f"missed a global buffer {len(self.bufs)} {u.arg}"
    self.bufs.append((u.arg[1], (u.dtype,u.arg[2])))
    self.r[u] = f"{self.variable_prefix}{u.arg[1]}"
    if self.load_global: self.kk(self.render_define_global(u))

  def uop_define_var(self, u):
    self.bufs.append((u.arg.expr, (u.dtype,False)))
    self.r[u] = f"{self.variable_prefix}{u.arg.expr}"
    if self.load_global: self.kk(self.render_define_var(u))

  def uop_define_local(self, u):
    self.kk(self.render_define_local(u))
    self.r[u] = u.arg[0]

  def uop_if(self, u):
    self.kk(self.render_if(u))
    self.depth += 1

  def uop_barrier(self, u):
     if self.barrier: self.kk(self.barrier)

  def uop_endloop(self, u):
    if not self.end_token: raise NotImplementedError(f"no code for {u.uop}")
    self.depth -= 1
    self.kk(self.end_token)

  def uop_endif(self, u): self.uop_endloop(u)

  def uop_phi(self, u):
    self.kk(self.render_phi(u))
    self.r[u] = self.r[u.vin[0]]

  def uop_bitcast(self, u): self.uop_cast(u,bitcast=True)

  def uop_store(self, u):
    assert u.vin[0].dtype is not None and u.vin[2].dtype is not None
    rendered_store = self.render_store(self.r[u.vin[0]], u.vin[0].dtype, self.r[u.vin[2]], u.vin[2].dtype, strip_parens(self.r[u.vin[1]]), u.vin[1].dtype, u.vin[0].uop is UOps.DEFINE_LOCAL)
    self.kk(f"if ({self.r[u.vin[3]]}) {{ {rendered_store} }}" if len(u.vin) > 3 else rendered_store)

  def ssa(self, u, prefix="t", dtype=None) -> str:
    ret = f"{self.variable_prefix}{prefix}{self.c[prefix]}"
    if getattr(self,"ssa_dtype",False): ret += f"_{dtype if dtype else self.types[u.dtype]}_"
    if u is not None: self.r[u] = ret
    self.c[prefix] += 1
    return ret

  def render_kernel(self, function_name, uops): raise NotImplementedError(f"no code for {type(self)}")

  def uops_to_code(self, function_name: str, uops: UOpGraph, outputs:Optional[List]=[], inputs:Optional[List]=[]) -> str:
      self.reset(uops, outputs, inputs)
      for u in uops:
          if u.uop not in [UOps.IF, UOps.BARRIER, UOps.ENDLOOP, UOps.ENDIF, UOps.STORE]: assert u.dtype is not None, f"None dtype for uop {u.uop}"
          method = getattr(self,f"uop_{u.uop.name.lower()}", None)
          if not method: raise NotImplementedError(f"No code for {u.uop}")
          method(u)
      return self.render_kernel(function_name, uops)
