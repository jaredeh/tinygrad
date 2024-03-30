from typing import Dict, List, Optional, NamedTuple, Tuple, Union, DefaultDict, cast, Literal, Callable
import math, re
from collections import defaultdict, Counter
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.helpers import strip_parens, getenv, DEBUG
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType
from tinygrad.codegen.uops import UOpGraph
from functools import reduce
from tinygrad.renderer.base import BaseLanguage
import operator

import sys

RUST_TYPE_MAP = {dtypes.long:["i64",8], dtypes.ulong:["u64",8], dtypes.float64:["f64",8], dtypes.double:["f64",8],
                 dtypes.int:["i32",4], dtypes.uint32:["u32",4], dtypes.int32:["i32",4], dtypes.float:["f32",4],
                 dtypes.int16:["i16",2], dtypes.uint16:["u16",2], dtypes.short:["i16",2], dtypes.ushort:["u16",2],
                 dtypes.int8:["i8",1], dtypes.uint8:["u8",1], dtypes.char:["i8",1], dtypes.uchar:["u8",1], dtypes.bool:["bool",1]}

def detect_bool(x:str) -> bool:
  if len([1 for c in [' < ', ' && ', ' == '] if c in x]) != 1: return False
  if len([1 for c in [' as '] if c in x]) != 0: return False
  return True

def detect_expression(x:str) -> bool: return any(not char.isalnum() and char != '_' for char in x)
def to_signed(x:str) -> DType: return [y for y in RUST_TYPE_MAP.keys() if RUST_TYPE_MAP[y][1] == RUST_TYPE_MAP[x][1] and not dtypes.is_unsigned(y) and not dtypes.is_float(y)][0]
def to_unsigned(x:str) -> DType: return [y for y in RUST_TYPE_MAP.keys() if RUST_TYPE_MAP[y][1] == RUST_TYPE_MAP[x][1] and dtypes.is_unsigned(y)][0]
#def detect_as_cast(x:str) -> bool: return any([1 for c in [' as '] if c in x])
def detect_as_cast(x:str): return bool(re.search(r'\sas\s[a-zA-Z0-9]+$', x))
def detect_numeric(x:str) -> bool:
  return not any(not char.isdigit() and char != '-' and char != '.' for char in x)
  # a = [not char.isdigit() and char != '-' and char != '.' for char in x]
  # b = not any(a)
  # print(f"detect_numeric x: {x} a: {a} b: {b}", file=sys.stderr)
  # return b
def detect_neg_const(x:str) -> bool: return detect_numeric(x) and x[0] == '-'
def negate_const(x:str) -> str: return f"{-int(x)}"
def has_parens(x:str) -> bool: return x[0] == '(' and x[-1] == ')'
def add_parens(x:str, on_cast:bool=False) -> str: return f"({x})" if detect_expression(x) and not has_parens(x) else f"({x})" if (detect_as_cast(x) and on_cast) else f"{x}"
def render_dtype(var_dtype:DType) -> str: return RUST_TYPE_MAP[var_dtype][0] if var_dtype in RUST_TYPE_MAP else var_dtype.name
def is_float(dtype) -> bool: return False if dtype is None else dtypes.is_float(dtype)
def is_unsigned(dtype) -> bool: return False if dtype is None else dtypes.is_unsigned(dtype)
def arust_cast(x, dst_dtype, src_dtype=None, force_cast=False, idx=False, ops=False):
  val = x
  if detect_numeric(x):
    val = f"{x}_{render_dtype(dst_dtype)}" if is_float(dst_dtype) or is_float(src_dtype) or ops else x
    if idx: return val
  if src_dtype is not None and src_dtype == dst_dtype: return f"{val} as usize" if idx else val
  print(f"  rust_cast A val={val}", file=sys.stderr)
  if dst_dtype is dtypes.bool: return f"({add_parens(val)} != { '0' if src_dtype is None else '0.0' if is_float(src_dtype) else '0' })" if val not in ["false","true"] else val
  print(f"  rust_cast B val={val}", file=sys.stderr)
  if src_dtype is dtypes.bool or detect_bool(val) and src_dtype is not None: val = f"{add_parens(val)}" if not is_float(dst_dtype) else f"{add_parens(val)} as usize"
  print(f"  rust_cast C idx={idx} val={val}", file=sys.stderr)
  if idx: return f"{add_parens(val)} as    usize"
  print(f"  rust_cast D val={val}", file=sys.stderr)
  if detect_neg_const(x) and detect_expression(x) and not detect_as_cast(x) and is_unsigned(src_dtype): val = f"{rust_cast(val,to_signed(src_dtype),force_cast=True)}"
  return f"{val} as {render_dtype(dst_dtype)}" if force_cast else val
def rust_cast(x, dst_dtype, src_dtype=None, force_cast=False, idx=False, ops=False):
  a = arust_cast(x, dst_dtype, src_dtype=src_dtype, force_cast=force_cast, idx=idx, ops=ops)
  print(f"rust_cast x: {x} dst_dtype: {dst_dtype}={is_float(dst_dtype)} src_dtype: {src_dtype}={is_float(src_dtype)} force_cast: {force_cast} idx: {idx} a: {a}", file=sys.stderr)
  return a

def do_add(a,b,dtype):
  print(f"do_add({a},{b},{dtype}) to_signed({dtype})={to_signed(dtype)} detect_neg({b})={detect_neg_const(b)} is_unsigned({dtype})={is_unsigned(dtype)}", file=sys.stderr)
  #return f"({a}+{b})" if dtype is dtypes.bool else f"({add_parens(rust_cast(a,to_signed(dtype),force_cast=True))}+{b})" if detect_neg(b) and is_unsigned(dtype) else f"({a}+{b})"
  return f"({a}+{b})" if dtype is dtypes.bool else f"( {a} - {negate_const(b)} )" if detect_neg_const(b) and is_unsigned(dtype) else f"({a}+{b})"
  #f"({a} || {b})" if dtype is dtypes.bool else f"({add_parens(rust_cast(a,to_signed(dtype)))}+{b})" if detect_neg(b) and is_unsigned(dtype) else f"({a}+{b})"


class RustLanguage(BaseLanguage):
  kernel_prefix: str = '#[no_mangle]\npub extern "C" '
  indent="  "
  end_token: str = "}"
  type_map: Dict[DType, str] = RUST_TYPE_MAP
  code_for_op: Dict = {
    UnaryOps.NEG: lambda x,dtype: f"(!{x})" if dtype is dtypes.bool else f"(-{x})", UnaryOps.SQRT: lambda x,dtype: f"{add_parens(rust_cast(x,dtype))}.sqrt()",
    UnaryOps.EXP2: lambda x,dtype: f"{add_parens(rust_cast(x,dtype))}.exp2()", UnaryOps.LOG2: lambda x,dtype: f"{add_parens(rust_cast(x,dtype))}.log2()",
    UnaryOps.SIN: lambda x,dtype: f"{add_parens(rust_cast(x,dtype))}.sin()",
    BinaryOps.ADD: lambda a,b,dtype: do_add(a,b,dtype),
    BinaryOps.SUB: lambda a,b,dtype: f"({a}-{b})",
    BinaryOps.MUL: lambda a,b,dtype: f"({rust_cast(a,dtype,ops=True)}*{rust_cast(b,dtype,ops=True)})" if dtype is not dtypes.bool else f"({a} && {b})",
    BinaryOps.DIV: lambda a,b,dtype: f"({a}/{b})", BinaryOps.MOD: lambda a,b,dtype: f"({a}%{b})",
    BinaryOps.MAX: lambda a,b,dtype: f"{add_parens(a)}.max({b})" if not bool(re.match(r'^-?\d+\.\d+$', a)) else f"{a}_{RUST_TYPE_MAP[dtype]}.max({b})",
    BinaryOps.CMPLT: lambda a,b,dtype: f"({add_parens(a,on_cast=True)} < {add_parens(b,on_cast=True)})",
    BinaryOps.CMPEQ: lambda a,b,dtype: f"({add_parens(a,on_cast=True)} == {add_parens(b,on_cast=True)})",
    BinaryOps.XOR: lambda a,b,dtype: f"({add_parens(a,on_cast=True)} ^ {add_parens(b,on_cast=True)})",
    TernaryOps.WHERE: lambda a,b,c,dtype: f"(if {a} {{{b}}} else {{{c}}})" }

  # returns a str expression of the casted xs with the given type
  def render_cast(self, x:str, src_dtype:DType, dst_dtype:DType, bitcast=False, force_cast=False) -> str:
    print(f"render_cast x: {x} src_dtype: {src_dtype} dst_dtype: {dst_dtype} bitcast: {bitcast}", file=sys.stderr)
    if bitcast and (is_float(dst_dtype) or is_float(src_dtype)):
      val = f"{x}.to_bits()" if is_float(src_dtype) else f"{render_dtype(dst_dtype)}::from_bits({rust_cast(x,to_unsigned(dst_dtype),src_dtype=src_dtype,force_cast=True)})"
      return rust_cast(val, dst_dtype, src_dtype, force_cast=force_cast)
      # elif is_float(dst_dtype):
      #   return rust_cast(f"{render_dtype(dst_dtype)}::from_bits({rust_cast(x,to_unsigned(dst_dtype),src_dtype=src_dtype,force_cast=True)})", dst_dtype, src_dtype, force_cast=force_cast)
    return rust_cast(x, dst_dtype, src_dtype, force_cast=force_cast)
    # if src_dtype[0] is not None and src_dtype[0] == var_dtype: return x[0] # no cast needed
    # if var_dtype is dtypes.bool: return f"({x[0]} != { '0.0' if src_dtype[0] is dtypes.float else '0' })"
    # return f"({x[0]}{' as usize' if dtypes.is_float(var_dtype) and (detect_bool(x[0]) or src_dtype[0] is dtypes.bool) else ''} as {render_dtype(var_dtype)})"

  # returns a str expression of the const with the given type
  def render_const(self, x:Union[float,int,bool], var_dtype) -> str:
    print(f"render_const x: {x} var_dtype: {var_dtype}", file=sys.stderr)
    if math.isnan(x): val = f"{render_dtype(var_dtype)}::NAN"
    elif math.isinf(x): val = f"{render_dtype(var_dtype)}::{'NEG_INFINITY' if x < 0 else 'INFINITY'}"
    else: val = f"{float(x)}" if dtypes.is_float(var_dtype) else f"{int(x)}" if dtypes.is_int(var_dtype) else f"{bool(x)}".lower()
    return self.render_cast(val, None, var_dtype)
    # return (self.render_cast([val]*var_dtype.count, [None]*var_dtype.count, var_dtype)
    #   if var_dtype.count > 1 or var_dtype not in [dtypes.float, dtypes.int, dtypes.bool] else val)

  # returns a str expression of the loaded value with the output type
  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    return f"{buf_name}[{self.render_index(idx)}]"

  # def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool,bool,int]]], uops:UOpGraph, prefix=None) -> str:
  #   print(f"render_kernel function_name: {function_name} kernel: {kernel} bufs: {bufs} prefix: {prefix}", file=sys.stderr)
  #   for name,a in bufs:
  #     print(f"render_kernel name: {name} a: {a}", file=sys.stderr)
  #   buftypes = []
  #   for name,(dtype,mutable,var,size) in bufs:
  #     if var:
  #       buftypes.append((name,render_dtype(dtype)))
  #     else:
  #       buftypes.append((name,("&mut " if mutable else "&")+"["+render_dtype(dtype)+f"; {size}]"))
  #   prg = ''.join([f"{self.kernel_prefix}fn {function_name}(",] +
  #   [', '.join([f'{name}: {t}' for name,t in buftypes])] +
  #   [") {\n"] + ['\n'.join(kernel), "\n}"])
  #   return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  def render_kernel(self, function_name:str, uops:UOpGraph) -> str:
    print(f"render_kernel function_name: {function_name} kernel: {self.kernel} bufs: {self.bufs}", file=sys.stderr)
    for name,a in self.bufs:
      print(f"render_kernel name: {name} a: {a}", file=sys.stderr)
    buftypes = []
    for name,(dtype,mutable,var,size) in self.bufs:
      if var:
        buftypes.append((name,render_dtype(dtype)))
      else:
        buftypes.append((name,("&mut " if mutable else "&")+"["+render_dtype(dtype)+f"; {size}]"))
    prg = ''.join([f"{self.kernel_prefix}fn {function_name}(",] +
    [', '.join([f'{name}: {t}' for name,t in buftypes])] +
    [") {\n"] + ['\n'.join(self.kernel), "\n}"])
    return prg

  def render_index(self, idx:str) -> str:
    return f"({idx}) as usize" if detect_expression(idx) else f"{idx} as usize"

  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx:str, idx_dtype:DType, local=False) -> str:
    print(f"render_store buf_name: {buf_name} buf_dtype: {buf_dtype} var_name: {var_name} var_dtype: {var_dtype} idx: {idx} idx_dtype: {idx_dtype} local: {local}", file=sys.stderr)
    return f"{buf_name}[{rust_cast(idx,idx_dtype,idx=True)}] = {rust_cast(var_name,buf_dtype,var_dtype)};"

  def render_local(self, name:str, dtype:DType, size:int): return f"{dtype.name} {name}[{size}];"
  def render_dtype(self, dtype:DType) -> str: return render_dtype(dtype)
  def render_alu(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType) -> str:
    return f"let {buf_name}:{render_dtype(buf_dtype)} = {var_name}; //{var_dtype} {buf_dtype}" if not var_dtype is dtypes.bool else f"let {buf_name} = {var_name}; //{var_dtype} {buf_dtype}"

  def render_loop(self, u) -> str:
    print(f"render_loop u: {u}", file=sys.stderr)
    print(f"ssa: {self.ssa(u,'ridx')}", file=sys.stderr)
    print(f"r: {self.r}", file=sys.stderr)
    return f"for {(expr := self.ssa(u,'ridx'))} in {self.r[u.vin[0]]}..{self.r[u.vin[1]]} {{"

  # def uop_store(self, u):
  #   vin = u.vin
  #   assert vin[0].dtype is not None and vin[2].dtype is not None
  #   if len(vin) > 3: self.kk(f"if ({self.r[vin[3]]}) {{")
  #   self.kk(self.render_store(self.r[vin[0]], vin[0].dtype, self.r[vin[2]], vin[2].dtype, strip_parens(self.r[vin[1]]), vin[1].dtype, vin[0].uop is UOps.DEFINE_LOCAL))
  #   if len(vin) > 3: self.kk("}")

  def uop_alu(self, u):
    if u.arg in {BinaryOps.ADD,BinaryOps.MUL,BinaryOps.XOR}: operands = [strip_parens(self.r[v]) if v.arg == u.arg else self.r[v]for v in u.vin]
    else: operands = [self.r[v] for v in u.vin]
    val = self.code_for_op[u.arg](*operands, u.dtype)
    assert self.child_count[u] != 0, f"childless ALU op found {u}"
    if self.child_count[u] <= 1 and u.arg != BinaryOps.MAX and not getenv("EXPAND_SSA"): self.r[u] = val
    else: self.kk(self.render_alu(self.ssa(u,'alu'),u.dtype,val,u.vin[0].dtype))

  def uop_load(self, u):
    val = self.render_load(u.dtype, self.r[u.vin[0]], u.vin[0].dtype, strip_parens(self.r[u.vin[1]]), u.vin[0].uop is UOps.DEFINE_LOCAL)
    if len(u.vin) > 3: val = self.code_for_op[TernaryOps.WHERE](self.r[u.vin[2]], val, self.r[u.vin[3]], u.dtype)
    self.kk(f"let {self.ssa(u,'val')} = {val};")

  def uop_cast(self, u, bitcast=False):
    if len(u.vin) != 1: raise NotImplementedError("vectorized cast not implemented in rust")
    val = self.render_cast(self.r[u.vin[0]], u.vin[0].dtype, u.dtype, bitcast=bitcast, force_cast=True)
    if self.child_count[u] <= 1: self.r[u] = val
    else: self.kk(f"let {self.ssa(u,'cast')} = {val};")

  def uop_define_global(self, u):
    print (f"self.bufsizes: {self.bufsizes} self.bufs: {self.bufs} arg: {u.arg}")
    assert len(self.bufs) == u.arg[0], f"missed a global buffer {len(self.bufs)} {u.arg}"
    if len(self.bufs) < len(self.bufsizes): print(f"more buffers than we have sizes for len(bufs): {len(self.bufs)} len(bufsizes):{len(self.bufsizes)}")
    if len(self.bufs) >= len(self.bufsizes): self.bufsizes.append(u.arg[3] if len(u.arg) > 3 else 1)
    self.bufs.append((u.arg[1], (u.dtype,u.arg[2],False,self.bufsizes[len(self.bufs)])))
    self.r[u] = u.arg[1]

  def uop_define_acc(self, u): self.kk(f"let mut {self.ssa(u,'acc')} = {self.render_const(u.arg, u.dtype)} as {self.render_dtype(u.dtype)};")
  def uop_const(self, u):  self.r[u] = self.render_const(u.arg, u.dtype) if u.arg >= 0 else f"{self.render_const(u.arg, u.dtype)}"
  def uop_gep(self, u):
    assert u.vin[0].dtype is not None
    from_ssa = u.vin[0].uop in {UOps.LOAD, UOps.WMMA, UOps.DEFINE_ACC}
    self.r[u] = (self.r[u.vin[0]] if from_ssa else f"{(self.r[u.vin[0]])}") + (f"[{u.arg}]" if u.vin[0].dtype.count > 4 else f".{'xyzw'[u.arg]}")
