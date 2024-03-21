from typing import Dict, List, Optional, NamedTuple, Tuple, Union, DefaultDict, cast, Literal, Callable
import math, functools
from collections import defaultdict, Counter
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.helpers import strip_parens, getenv, prod
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType
from tinygrad.codegen.uops import UOpGraph

class RustLanguage(NamedTuple):
  kernel_prefix: str = '#[no_mangle]\npub extern "C" '
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = "const int"
  barrier: str = ""
  code_for_workitem: Dict[Union[Literal["g"], Literal["l"], Literal["i"]], Callable] = {}
  global_max: List[int] = []
  local_max: List[int] = []
  extra_args: List[str] = []
  float4: Optional[str] = None
  uses_vload: bool = False
  uses_ptr_arithmetic: bool = False
  type_map: Dict[DType, str] = {dtypes.float: "f32", dtypes.int: "i32"}
  # code_for_op: Dict = {
  #   UnaryOps.NEG: lambda x,dtype: f"(!{x})" if dtype is dtypes.bool else f"(-{x})", UnaryOps.SQRT: lambda x,dtype: f"sqrt({x})",
  #   UnaryOps.EXP2: lambda x,dtype: f"exp2({x})", UnaryOps.LOG2: lambda x,dtype: f"log2({x})", UnaryOps.SIN: lambda x,dtype: f"sin({x})",
  #   BinaryOps.ADD: lambda a,b,dtype: f"({a}+{b})", BinaryOps.SUB: lambda a,b,dtype: f"({a}-{b})", BinaryOps.MUL: lambda a,b,dtype: f"({a}*{b})",
  #   BinaryOps.DIV: lambda a,b,dtype: f"({a}/{b})", BinaryOps.MAX: lambda a,b,dtype: f"max({a},{b})", BinaryOps.MOD: lambda a,b,dtype: f"({a}%{b})",
  #   BinaryOps.CMPLT: lambda a,b,dtype: f"({a}<{b})", BinaryOps.CMPEQ: lambda a,b,dtype: f"({a}=={b})", BinaryOps.XOR: lambda a,b,dtype: f"({a}^{b})",
  #   TernaryOps.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})"}

  code_for_op: Dict = {
    UnaryOps.NEG: lambda x,dtype: f"(!{x})" if dtype is dtypes.bool else f"(-{x})", UnaryOps.SQRT: lambda x,dtype: f"{x}.sqrt()",
    UnaryOps.EXP2: lambda x,dtype: f"{x}.exp2()", UnaryOps.LOG2: lambda x,dtype: f"{x}.log2()", UnaryOps.SIN: lambda x,dtype: f"{x}.sin()",
    BinaryOps.ADD: lambda a,b,dtype: f"({a}+{b})", BinaryOps.SUB: lambda a,b,dtype: f"({a}-{b})", BinaryOps.MUL: lambda a,b,dtype: f"({a}*{b})",
    BinaryOps.DIV: lambda a,b,dtype: f"({a}/{b})", BinaryOps.MAX: lambda a,b,dtype: f"{a}.max({b})", BinaryOps.MOD: lambda a,b,dtype: f"({a}%{b})",
    BinaryOps.CMPLT: lambda a,b,dtype: f"({a}<{b})", BinaryOps.CMPEQ: lambda a,b,dtype: f"({a}=={b})", BinaryOps.XOR: lambda a,b,dtype: f"({a}^{b})",
    TernaryOps.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})"}

  # returns a str expression of the casted xs with the given type
  def render_cast(self, x:List[str], var_dtype:DType, bitcast=False) -> str:
    if bitcast: return f"(*(({self.buffer_prefix}{self.render_dtype(var_dtype)}*)&{x[0]}))"
    if len(x) == 1: return f"({self.render_dtype(var_dtype)})({x[0]})"
    assert len(x) == var_dtype.count, f"cast is wrong size {len(x)} != {var_dtype.count}"
    assert self.float4 is not None, "vectorized cast is not supported on this platform"
    return f"{self.float4.replace('float4', self.render_dtype(var_dtype))}({','.join(x)})"

  # returns a str expression of the const with the given type
  def render_const(self, x:Union[float,int,bool], var_dtype) -> str:
    if math.isnan(x): val = "NAN"
    elif math.isinf(x): val = ("-" if x < 0 else "") + "INFINITY"
    elif var_dtype == dtypes.float64: val = f"{float(x)}"
    else: val = f"{float(x)}" if dtypes.is_float(var_dtype) else f"{int(x)}" if dtypes.is_int(var_dtype) else f"{bool(x)}".lower()
    return (self.render_cast([val]*var_dtype.count, var_dtype)
      if var_dtype.count > 1 or var_dtype not in [dtypes.float, dtypes.int, dtypes.bool] else val)

  # returns a str expression of the loaded value with the output type
  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert output_dtype == dtypes.float.vec(4), f"images must be float4, getting {output_dtype}"
      return f"read_imagef({buf_name}, smp, {idx})"
    if self.uses_vload and buf_dtype.scalar() == dtypes.float16 and output_dtype.scalar() != dtypes.float16:
      return f"vload_half{'' if output_dtype.count == 1 else str(output_dtype.count)}(0, {buf_name}+{idx})"
    if output_dtype.count > 1:
      out_val = f"*(({self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix}{buf_dtype.name}{output_dtype.count}*)({buf_name}+{idx}))"  # noqa: E501
    else:
      out_val = f"*({buf_name}+{idx})" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}]"
    return self.render_cast([out_val], output_dtype) if output_dtype != buf_dtype else out_val

  def get_kernel_modifier(self, uops:UOpGraph) -> str: return ""
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool,int]]], uops:UOpGraph, prefix=None) -> str:
    buftypes = [(name,("&mut " if mutable else "&")+self.buffer_prefix+"["+self.render_dtype(dtype)+f"; {size}]"+self.buffer_suffix) for name,(dtype,mutable,size) in bufs]
    prg = ''.join([f"{self.kernel_prefix}fn {self.get_kernel_modifier(uops)}{function_name}(",] +
    [', '.join([f'{name}: {t}' for name,t in buftypes] + self.extra_args)] +
    [") {\n"] + ['\n'.join(kernel), "\n}"])
    return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  # returns a str statement that does the store
  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx:str, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert var_dtype == dtypes.float.vec(4), f"images must be float4, getting {var_dtype}"
      return f"write_imagef({buf_name}, {idx}, {var_name});"
    if self.uses_vload and buf_dtype.scalar() == dtypes.float16 and var_dtype.scalar() != dtypes.float16:
      return f"vstore_half{'' if var_dtype.count == 1 else str(var_dtype.count)}({var_name}, 0, {buf_name}+{idx});"
    if var_dtype.count > 1:
      prefix = self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix
      return f"*(({prefix}{buf_dtype.name}{var_dtype.count}*)({buf_name}+{idx})) = {var_name};"
    return f"*({buf_name}+{idx}) = {var_name};" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}] = {var_name};"

  def render_local(self, name:str, dtype:DType, size:int): return self.smem_align + self.smem_prefix + f"{dtype.name} {name}[{size}];"
  def render_dtype(self, var_dtype:DType) -> str: return self.type_map[var_dtype] if var_dtype in self.type_map else var_dtype.name
  # def render_dtype(self, var_dtype:DType) -> str:
  #   print(var_dtype)
  #   print(var_dtype.name)
  #   if var_dtype in self.type_map:
  #     o = self.type_map[var_dtype]
  #   else:
  #     o = var_dtype.name
  #   return o

def uops_to_rust(lang:RustLanguage, function_name:str, uops:UOpGraph) -> str:
  kernel = []
  bufs: List[Tuple[str, Tuple[DType, bool, int]]] = []
  #pend_close = None
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
    #print(f"uop: {uop} dtype: {dtype} vin: {vin} args: {args}")
    #print(u)
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
      kk(lang.render_store(r[vin[0]], vin[0].dtype, r[vin[2]], vin[2].dtype, strip_parens(r[vin[1]]), vin[0].uop is UOps.DEFINE_LOCAL))
      if len(vin) > 3: kk("}")
    else:
      assert dtype is not None, f"None dtype for uop {uop}"
      if uop is UOps.LOOP:
        kk(f"for {(expr := ssa(u,'ridx'))} in {r[vin[0]]}..{r[vin[1]]} {{")
        depth += 1
      elif uop is UOps.ALU:
        # remove parens if ALU types are the same. TODO: can do more here
        if args in {BinaryOps.ADD,BinaryOps.MUL,BinaryOps.XOR}: operands = [strip_parens(r[v]) if v.arg == args else r[v]for v in vin]
        else: operands = [r[v] for v in vin]
        val = lang.code_for_op[args](*operands, dtype)
        assert child_count[u] != 0, f"childless ALU op found {u}"
        # TODO: fix index rendering issue. fix clang nested max macro issue
        if child_count[u] <= 1 and args != BinaryOps.MAX and not getenv("EXPAND_SSA"): r[u] = val
        else: kk(f"let {ssa(u,'alu')} = {val};")
      elif uop is UOps.SPECIAL:
        kk(f"int {args[1]} = {lang.code_for_workitem[args[1][0]](args[0])}; /* {args[2]} */")
        r[u] = args[1]
      elif uop is UOps.LOAD:
        val = lang.render_load(dtype, r[vin[0]], vin[0].dtype, strip_parens(r[vin[1]]), vin[0].uop is UOps.DEFINE_LOCAL)
        # NOTE: this relies on the load not happening if it's in the unselected branch
        if len(vin) > 3: val = lang.code_for_op[TernaryOps.WHERE](r[vin[2]], val, r[vin[3]], dtype)
        kk(f"let {ssa(u,'val')} = {val};")
      elif uop is UOps.PHI:
        kk(f"{r[vin[0]]} = {r[vin[1]]};")
        r[u] = r[vin[0]]
      elif uop is UOps.CAST:
        if isinstance(args, tuple) and args[1]:  # bitcast
          assert len(vin) == 1
          precast = ssa(None,'precast')
          kk(f"{lang.render_dtype(cast(DType, vin[0].dtype))} {precast} = {r[vin[0]]};")
          val = lang.render_cast([precast], dtype, bitcast=True)
        else:
          val = lang.render_cast([r[x] for x in vin], dtype, bitcast=False)
        if child_count[u] <= 1: r[u] = val
        else: kk(f"{dtype.name} {ssa(u,'cast')} = {val};")
      elif uop is UOps.DEFINE_LOCAL:
        kk(lang.render_local(args[0], dtype, args[1]))
        r[u] = args[0]
      elif uop is UOps.DEFINE_VAR:
        bufs.append((args.expr, (dtype,False)))
        r[u] = args.expr
      elif uop is UOps.DEFINE_GLOBAL:
        assert len(bufs) == args[0], f"missed a global buffer {len(bufs)} {args}"
        bufs.append((args[1], (dtype,args[2],args[3])))
        r[u] = args[1]
      elif uop is UOps.WMMA: kk(f"{dtype.name} {ssa(u, 'wmma')} = {args}({r[vin[0]]}, {r[vin[1]]}, {r[vin[2]]});")
      elif uop is UOps.DEFINE_ACC: kk(f"let mut {ssa(u,'acc')} = {lang.render_const(args, dtype)};")
      elif uop is UOps.CONST: r[u] = lang.render_const(args, dtype) if args >= 0 else f"({lang.render_const(args, dtype)})"
      elif uop is UOps.GEP:
        assert vin[0].dtype is not None
        from_ssa = vin[0].uop in {UOps.LOAD, UOps.WMMA, UOps.DEFINE_ACC}
        r[u] = (r[vin[0]] if from_ssa else f"{(r[vin[0]])}") + (f"[{args}]" if vin[0].dtype.count > 4 else f".{'xyzw'[args]}")
      else: raise RuntimeError(f"failed to render {uop}")

  return lang.render_kernel(function_name, kernel, bufs, uops)
