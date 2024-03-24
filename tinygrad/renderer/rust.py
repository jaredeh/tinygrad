from typing import Dict, List, Optional, NamedTuple, Tuple, Union, DefaultDict, cast, Literal, Callable
import math, re
from collections import defaultdict, Counter
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.helpers import strip_parens, getenv, DEBUG
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType
from tinygrad.codegen.uops import UOpGraph

RUST_TYPE_MAP = {dtypes.float: "f32", dtypes.int: "i32", dtypes.bool: "bool"}


class RustLanguage(NamedTuple):
  kernel_prefix: str = '#[no_mangle]\npub extern "C" '
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = ""
  barrier: str = ""
  code_for_workitem: Dict[Union[Literal["g"], Literal["l"], Literal["i"]], Callable] = {}
  global_max: List[int] = []
  local_max: List[int] = []
  extra_args: List[str] = []
  float4: Optional[str] = None
  uses_vload: bool = False
  uses_ptr_arithmetic: bool = False
  type_map: Dict[DType, str] = RUST_TYPE_MAP
  code_for_op: Dict = {
    UnaryOps.NEG: lambda x,dtype: f"(!{x})" if dtype is dtypes.bool else f"(-{x})", UnaryOps.SQRT: lambda x,dtype: f"{x}.sqrt()",
    UnaryOps.EXP2: lambda x,dtype: f"{x}.exp2()", UnaryOps.LOG2: lambda x,dtype: f"{x}.log2()", UnaryOps.SIN: lambda x,dtype: f"{x}.sin()",
    BinaryOps.ADD: lambda a,b,dtype: f"({a}+{b})", BinaryOps.SUB: lambda a,b,dtype: f"({a}-{b})",
    BinaryOps.MUL: lambda a,b,dtype: f"({a}*{b})" if dtype is not dtypes.bool else f"({a} && {b})",
    BinaryOps.DIV: lambda a,b,dtype: f"({a}/{b})", BinaryOps.MOD: lambda a,b,dtype: f"({a}%{b})",
    BinaryOps.MAX: lambda a,b,dtype: f"{a}.max({b})" if not bool(re.match(r'^-?\d+\.\d+$', a)) else f"{a}_{RUST_TYPE_MAP[dtype]}.max({b})",
    BinaryOps.CMPLT: lambda a,b,dtype: f"({a}<{b})", BinaryOps.CMPEQ: lambda a,b,dtype: f"({a}=={b})", BinaryOps.XOR: lambda a,b,dtype: f"({a}^{b})",
    TernaryOps.WHERE: lambda a,b,c,dtype: f"(if {a} {{{b}}} else {{{c}}})" }

  def detect_bool(self, x:str) -> bool:
    for c in ['<', '>', '!', '&', '|', '==']:
      if c in x: return True
    return False

  def detect_expression(self, x:str) -> bool:
    return any(not char.isalnum() and char != '_' for char in x)

  # returns a str expression of the casted xs with the given type
  def render_cast(self, x:List[str], src_dtype:List, var_dtype:DType, bitcast=False) -> str:
    if bitcast: raise NotImplementedError("bitcast not implemented in rust")
    assert len(x) == var_dtype.count, f"cast is wrong size {len(x)} != {var_dtype.count}"
    if len(x) != 1:
      raise NotImplementedError("vectorized cast not implemented in rust")
    if src_dtype[0] is not None and src_dtype[0] == var_dtype: return x[0] # no cast needed
    import sys
    print(f"casting {x[0]} from {src_dtype[0]} to {var_dtype} {self.detect_bool(x[0])}", file=sys.stderr)
    if var_dtype is dtypes.bool: return f"({x[0]} != { '0.0' if src_dtype[0] is dtypes.float else '0' })"
    return f"({x[0]}{' as usize' if dtypes.is_float(var_dtype) and (self.detect_bool(x[0]) or src_dtype[0] is dtypes.bool) else ''} as {self.render_dtype(var_dtype)})"

  # returns a str expression of the const with the given type
  def render_const(self, x:Union[float,int,bool], var_dtype) -> str:
    if math.isnan(x): val = f"{self.render_dtype(var_dtype)}::NAN"
    elif math.isinf(x): val = f"{self.render_dtype(var_dtype)}::{'NEG_INFINITY' if x < 0 else 'INFINITY'}"
    else: val = f"{float(x)}" if dtypes.is_float(var_dtype) else f"{int(x)}" if dtypes.is_int(var_dtype) else f"{bool(x)}".lower()
    return (self.render_cast([val]*var_dtype.count, [None]*var_dtype.count, var_dtype)
      if var_dtype.count > 1 or var_dtype not in [dtypes.float, dtypes.int, dtypes.bool] else val)

  # returns a str expression of the loaded value with the output type
  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    return f"{buf_name}[{self.render_index(idx)}]"

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool,int]]], uops:UOpGraph, prefix=None) -> str:
    buftypes = [(name,("&mut " if mutable else "&")+self.buffer_prefix+"["+self.render_dtype(dtype)+f"; {size}]"+self.buffer_suffix) for name,(dtype,mutable,size) in bufs]
    prg = ''.join([f"{self.kernel_prefix}fn {function_name}(",] +
    [', '.join([f'{name}: {t}' for name,t in buftypes] + self.extra_args)] +
    [") {\n"] + ['\n'.join(kernel), "\n}"])
    return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  def render_index(self, idx:str) -> str:
    return f"({idx}) as usize" if self.detect_expression(idx) else f"{idx} as usize"

  # returns a str statement that does the store
  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx:str, idx_dtype:DType, local=False) -> str:
    return f"{buf_name}[{self.render_index(idx)}] = {var_name} as {self.render_dtype(buf_dtype)};"

  def render_local(self, name:str, dtype:DType, size:int): return self.smem_align + self.smem_prefix + f"{dtype.name} {name}[{size}];"
  def render_dtype(self, var_dtype:DType) -> str: return self.type_map[var_dtype] if var_dtype in self.type_map else var_dtype.name
  def render_alu(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType) -> str:
    return f"let {buf_name}:{self.render_dtype(buf_dtype)} = {var_name}; //{var_dtype} {buf_dtype}" if not var_dtype is dtypes.bool else f"let {buf_name} = {var_name}; //{var_dtype} {buf_dtype}"

def uops_to_rust(lang:RustLanguage, function_name:str, uops:UOpGraph, outputs, inputs) -> str:
  kernel = []
  bufsizes = [b.size for b in outputs+inputs]
  bufs: List[Tuple[str, Tuple[DType, bool, int]]] = []
  depth = 1
  def kk(s): kernel.append("  "*depth+s)

  c: DefaultDict[str, int] = defaultdict(int)
  r: Dict[UOp, str] = {}
  def ssa(u, prefix="t"):
    nonlocal c, r
    ret = f"{prefix}{c[prefix]}"
    if u is not None: r[u] = ret
    c[prefix] += 1
    return ret

  child_count = Counter(v for ru in uops for v in ru.vin)

  for u in uops:
    uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
    # these four uops don't have output dtypes
    if uop is UOps.IF:
      kk(f"if ({r[vin[0]]}) {{")
      depth += 1
    elif uop is UOps.BARRIER: kk(lang.barrier)
    elif uop in {UOps.ENDLOOP, UOps.ENDIF}:
      depth -= 1
      kk("}")
    elif uop is UOps.STORE:
      assert vin[0].dtype is not None and vin[2].dtype is not None
      if len(vin) > 3: kk(f"if ({r[vin[3]]}) {{")
      kk(lang.render_store(r[vin[0]], vin[0].dtype, r[vin[2]], vin[2].dtype, strip_parens(r[vin[1]]), vin[1].dtype, vin[0].uop is UOps.DEFINE_LOCAL))
      if len(vin) > 3: kk("}")
    else:
      assert dtype is not None, f"None dtype for uop {uop}"
      if uop is UOps.LOOP:
        kk(f"for {(expr := ssa(u,'ridx'))} in {r[vin[0]]}..{r[vin[1]]} {{")
        depth += 1
      elif uop is UOps.ALU:
        if args in {BinaryOps.ADD,BinaryOps.MUL,BinaryOps.XOR}: operands = [strip_parens(r[v]) if v.arg == args else r[v]for v in vin]
        else: operands = [r[v] for v in vin]
        val = lang.code_for_op[args](*operands, dtype)
        assert child_count[u] != 0, f"childless ALU op found {u}"
        if child_count[u] <= 1 and args != BinaryOps.MAX and not getenv("EXPAND_SSA"): r[u] = val
        else: kk(lang.render_alu(ssa(u,'alu'),dtype,val,vin[0].dtype))
      elif uop is UOps.LOAD:
        val = lang.render_load(dtype, r[vin[0]], vin[0].dtype, strip_parens(r[vin[1]]), vin[0].uop is UOps.DEFINE_LOCAL)
        if len(vin) > 3: val = lang.code_for_op[TernaryOps.WHERE](r[vin[2]], val, r[vin[3]], dtype)
        kk(f"let {ssa(u,'val')} = {val};")
      elif uop is UOps.PHI:
        kk(f"{r[vin[0]]} = {r[vin[1]]} as {lang.render_dtype(dtype)};")
        r[u] = r[vin[0]]
      elif uop in {UOps.CAST, UOps.BITCAST}:
        if uop is UOps.BITCAST:
          assert len(vin) == 1
          precast = ssa(None,'precast')
          kk(f"{lang.render_dtype(cast(DType, vin[0].dtype))} {precast} = {r[vin[0]]};")
          val = lang.render_cast([precast], [vin[0].dtype], dtype, bitcast=True)
        else:
          val = lang.render_cast([r[x] for x in vin], [x.dtype for x in vin], dtype, bitcast=False)
        if child_count[u] <= 1: r[u] = val
        else: kk(f"let {ssa(u,'cast')} = {val};")
      elif uop is UOps.DEFINE_LOCAL:
        kk(lang.render_local(args[0], dtype, args[1]))
        r[u] = args[0]
      elif uop is UOps.DEFINE_GLOBAL:
        assert len(bufs) == args[0], f"missed a global buffer {len(bufs)} {args}"
        assert len(bufs) < len(bufsizes), f"more buffers than we have sizes for len(bufs): {len(bufs)} len(bufsizes):{len(bufsizes)}"
        if DEBUG > 0 and args[3] != bufsizes[len(bufs)]: print(f"warning: buffer size mismatch {args[3]} {bufsizes[len(bufs)]}")
        bufs.append((args[1], (dtype,args[2],bufsizes[len(bufs)])))
        r[u] = args[1]
      elif uop is UOps.DEFINE_ACC: kk(f"let mut {ssa(u,'acc')} = {lang.render_const(args, dtype)} as {lang.render_dtype(dtype)};")
      elif uop is UOps.CONST: r[u] = lang.render_const(args, dtype) if args >= 0 else f"({lang.render_const(args, dtype)})"
      elif uop is UOps.GEP:
        assert vin[0].dtype is not None
        from_ssa = vin[0].uop in {UOps.LOAD, UOps.WMMA, UOps.DEFINE_ACC}
        r[u] = (r[vin[0]] if from_ssa else f"{(r[vin[0]])}") + (f"[{args}]" if vin[0].dtype.count > 4 else f".{'xyzw'[args]}")
      else: raise RuntimeError(f"failed to render {uop}")

  return lang.render_kernel(function_name, kernel, bufs, uops)
