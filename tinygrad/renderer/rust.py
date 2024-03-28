from typing import Dict, List, Optional, NamedTuple, Tuple, Union, DefaultDict, cast, Literal, Callable
import math, re
from collections import defaultdict, Counter
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.helpers import strip_parens, getenv, DEBUG
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType
from tinygrad.codegen.uops import UOpGraph
from functools import reduce
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

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool,bool,int]]], uops:UOpGraph, prefix=None) -> str:
    print(f"render_kernel function_name: {function_name} kernel: {kernel} bufs: {bufs} prefix: {prefix}", file=sys.stderr)
    for name,a in bufs:
      print(f"render_kernel name: {name} a: {a}", file=sys.stderr)
    buftypes = []
    for name,(dtype,mutable,var,size) in bufs:
      if var:
        buftypes.append((name,render_dtype(dtype)))
      else:
        buftypes.append((name,("&mut " if mutable else "&")+self.buffer_prefix+"["+render_dtype(dtype)+f"; {size}]"+self.buffer_suffix))
    #buftypes = [(name,("&mut " if mutable else "&")+self.buffer_prefix+"["+render_dtype(dtype)+f"; {size}]"+self.buffer_suffix) ]
    prg = ''.join([f"{self.kernel_prefix}fn {function_name}(",] +
    [', '.join([f'{name}: {t}' for name,t in buftypes] + self.extra_args)] +
    [") {\n"] + ['\n'.join(kernel), "\n}"])
    return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  def render_index(self, idx:str) -> str:
    return f"({idx}) as usize" if detect_expression(idx) else f"{idx} as usize"

  # returns a str statement that does the store
  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx:str, idx_dtype:DType, local=False) -> str:
    #return f"{buf_name}[{self.render_index(idx)}] = {var_name} as {render_dtype(buf_dtype)};"
    print(f"render_store buf_name: {buf_name} buf_dtype: {buf_dtype} var_name: {var_name} var_dtype: {var_dtype} idx: {idx} idx_dtype: {idx_dtype} local: {local}", file=sys.stderr)
    return f"{buf_name}[{rust_cast(idx,idx_dtype,idx=True)}] = {rust_cast(var_name,buf_dtype,var_dtype)};"

  def render_local(self, name:str, dtype:DType, size:int): return self.smem_align + self.smem_prefix + f"{dtype.name} {name}[{size}];"
  def render_dtype(self, dtype:DType) -> str: return render_dtype(dtype)
  def render_alu(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType) -> str:
    return f"let {buf_name}:{render_dtype(buf_dtype)} = {var_name}; //{var_dtype} {buf_dtype}" if not var_dtype is dtypes.bool else f"let {buf_name} = {var_name}; //{var_dtype} {buf_dtype}"

  
def print_uop_graph(uop_graph):
  print("uop graph:")
  for u in uop_graph:
    uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
    print(f"  {uop} {dtype} {vin} {args}")

def uops_to_rust(lang:RustLanguage, function_name:str, uops:UOpGraph, outputs, inputs) -> str:
  kernel = []
  bufsizes = [b.size for b in outputs+inputs]
  maxbufsize = 0 if len(bufsizes) == 0 else max(bufsizes)
  bufs: List[Tuple[str, Tuple[DType, bool, int]]] = []
  depth = 1
  print(f"outputs: {outputs} inputs: {inputs} bufsizes: {bufsizes}")
  print_uop_graph(uops)
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

  # #if len(bufsizes) == 0: maxbufsize = max([u.arg for u in [u for i,u in uops() if u.uop is UOps.CONST] if isinstance(u.arg, int)])+1
  # if len(bufsizes) == 0:
  #   c = []
  #   d = [u.vin for i,u in enumerate(uops) if u.uop is UOps.LOOP and len(u.vin) == 2]
  #   print(f"d: {d}" )
  #   for a in d:
  #     if isinstance(a[1].arg, int):
  #       c.append(a)
  #     else
  #   #     if len(a) == 0:
  #   #       c.append(a)
  #   #     else:
  #   #       for b in a:
  #   #       print(b.arg)
  #   print(f"maxbufsize: {maxbufsize}")

  for u in uops:
    uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
    print(f"uop: {uop} dtype: {dtype} vin: {vin} args: {args}")
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
        if len(vin) != 1: raise NotImplementedError("vectorized cast not implemented in rust")
        val = lang.render_cast(r[vin[0]], vin[0].dtype, dtype, bitcast=(uop is UOps.BITCAST), force_cast=True)
        if child_count[u] <= 1: r[u] = val
        else: kk(f"let {ssa(u,'cast')} = {val};")
      elif uop is UOps.DEFINE_LOCAL:
        kk(lang.render_local(args[0], dtype, args[1]))
        r[u] = args[0]
      elif uop is UOps.DEFINE_VAR:
        bufs.append((args.expr, (dtype,False,True,0)))
        r[u] = args.expr
      elif uop is UOps.DEFINE_GLOBAL:
        print (f"bufsizes: {bufsizes} bufs: {bufs} args: {args}")
        assert len(bufs) == args[0], f"missed a global buffer {len(bufs)} {args}"
        if len(bufs) < len(bufsizes): print(f"more buffers than we have sizes for len(bufs): {len(bufs)} len(bufsizes):{len(bufsizes)}")
        if len(bufs) >= len(bufsizes): bufsizes.append(args[3] if len(args) > 3 else 1)
        bufs.append((args[1], (dtype,args[2],False,bufsizes[len(bufs)])))
        r[u] = args[1]
      elif uop is UOps.DEFINE_ACC: kk(f"let mut {ssa(u,'acc')} = {lang.render_const(args, dtype)} as {lang.render_dtype(dtype)};")
      elif uop is UOps.CONST: r[u] = lang.render_const(args, dtype) if args >= 0 else f"{lang.render_const(args, dtype)}"
      elif uop is UOps.GEP:
        assert vin[0].dtype is not None
        from_ssa = vin[0].uop in {UOps.LOAD, UOps.WMMA, UOps.DEFINE_ACC}
        r[u] = (r[vin[0]] if from_ssa else f"{(r[vin[0]])}") + (f"[{args}]" if vin[0].dtype.count > 4 else f".{'xyzw'[args]}")
      else: raise RuntimeError(f"failed to render {uop}")

  return lang.render_kernel(function_name, kernel, bufs, uops)
