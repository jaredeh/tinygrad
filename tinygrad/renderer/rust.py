from typing import Dict, List, Tuple, Union, Literal, Callable, cast
import math, re
from collections import defaultdict, Counter
from tinygrad.opt import tc
from tinygrad.uop.ops import GroupOp, Ops, UOp, PatternMatcher, UPat
from tinygrad.helpers import strip_parens, getenv, prod, dedup, DEBUG
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType, to_dtype
from tinygrad.renderer import Renderer

RUST_TYPE_MAP = {dtypes.long:["i64",8], dtypes.ulong:["u64",8], dtypes.float64:["f64",8], dtypes.double:["f64",8],
                 dtypes.int:["i32",4], dtypes.uint32:["u32",4], dtypes.int32:["i32",4], dtypes.float:["f32",4],
                 dtypes.int16:["i16",2], dtypes.uint16:["u16",2], dtypes.short:["i16",2], dtypes.ushort:["u16",2],
                 dtypes.int8:["i8",1], dtypes.uint8:["u8",1], dtypes.char:["i8",1], dtypes.uchar:["u8",1], dtypes.bool:["bool",1]}

RUST_TYPE_MAP = {
    "long": "i64", "ulong": "u64", "float64": "f64", "double": "f64", "half": "f16",
    "int": "i32", "uint": "u32", "uint32": "u32", "int32": "i32", "float": "f32",
    "int16": "i16", "uint16": "u16", "unsigned short": "u16", "short": "i16", "ushort": "u16", "unsigned int": "u32", "unsigned long": "u64",
    "int8": "i8", "uint8": "u8", "char": "i8", "signed char": "i8", "uchar": "u8", "unsigned char": "u8", "bool": "bool"
}

def detect_bool(x:str) -> bool:
  if len([1 for c in [' < ', ' && ', ' == '] if c in x]) != 1: return False
  if len([1 for c in [' as '] if c in x]) != 0: return False
  return True

def detect_expression(x:str) -> bool: return any(not char.isalnum() and char != '_' for char in x)
def to_signed(x:str) -> DType:
  return [y for y in RUST_TYPE_MAP.keys() if RUST_TYPE_MAP[y][1] == RUST_TYPE_MAP[x][1] and not dtypes.is_unsigned(y) and not dtypes.is_float(y)][0]
def to_unsigned(x:DType) -> DType:
  return [y for y in dtypes.fields().values() if x.itemsize == y.itemsize and dtypes.is_unsigned(y) and not dtypes.is_float(y)][0]
def detect_as_cast(x:str): return bool(re.search(r'\sas\s[a-zA-Z0-9]+$', x))
def detect_numeric(x:str) -> bool:
  return not any(not char.isdigit() and char != '-' and char != '.' for char in x)
def detect_neg_const(x:str) -> bool: return detect_numeric(x) and x[0] == '-'
def negate_const(x:str) -> str: return f"{-int(x)}"
def has_parens(x:str) -> bool: return x[0] == '(' and x[-1] == ')'
def add_parens(x:str, on_cast:bool=False) -> str:
  return f"({x})" if detect_expression(x) and not has_parens(x) else f"({x})" if (detect_as_cast(x) and on_cast) else f"{x}"
def render_dtype(var_dtype:DType) -> str:
  if var_dtype.name not in RUST_TYPE_MAP:
    raise ValueError(f"Unknown dtype: {var_dtype} name: {var_dtype.name}")
  return RUST_TYPE_MAP[var_dtype.name]
def is_float(dtype) -> bool: return False if dtype is None else dtypes.is_float(dtype)
def is_bool(dtype) -> bool: return False if dtype is None else dtypes.is_bool(dtype) or dtype.name == "bool"
def is_unsigned(dtype) -> bool: return False if dtype is None else dtypes.is_unsigned(dtype)
def rust_cast(x:str, dst_dtype:DType, src_dtype:DType=None, force_cast=False, idx=False, ops=False):
  if DEBUG >= 6: print(f"rust_cast: {x}, {dst_dtype}, {src_dtype}, {force_cast}, {idx}, {ops}")
  #print(f"dst_dtype: {x}, {dst_dtype.max}")
  val = x
  if detect_numeric(x):
    if is_float(dst_dtype) or is_float(src_dtype) or ops:
      val = f"{x}_{render_dtype(dst_dtype)}"
    elif int(x) >= 0:
        val = f"{dst_dtype.max & int(x)}"
    elif is_unsigned(dst_dtype) and int(x) < 0:
        val = f"{dst_dtype.max}"
    if idx: return val
  if dtypes.is_bool(dst_dtype):
    return val if is_bool(src_dtype) else f"{add_parens(val)} != { '0.0' if is_float(src_dtype) else '0' }" if val not in ["false","true"] else val
  if src_dtype is not None and src_dtype == dst_dtype: return f"{val} as usize" if idx else val
  if is_bool(src_dtype) or detect_bool(val) and src_dtype is not None:
    val = f"{add_parens(val)}" if not is_float(dst_dtype) else f"{add_parens(add_parens(val+" as usize"))}"
    force_cast = True
  if idx: return f"{add_parens(val)} as   usize"
  if detect_neg_const(x) and detect_expression(x) and not detect_as_cast(x) and is_unsigned(src_dtype):
    val = f"{rust_cast(val,to_signed(src_dtype),force_cast=True)}"
  if is_unsigned(dst_dtype) != is_unsigned(src_dtype):
    val = f"{val} as {render_dtype(dst_dtype)}"
  return f"{val} as {render_dtype(dst_dtype)}" if force_cast else val

base_rewrite = PatternMatcher([
  (UPat(Ops.DEFINE_REG, name="x"), lambda ctx, x: ctx[x.src[0]]),
  (UPat(Ops.ASSIGN, name="x"), lambda ctx, x: f"{ctx[x.src[0]]} = {ctx[x.src[1]]};"),
  (UPat(Ops.IF, name="x"), lambda ctx, x: f"if {ctx[x.src[0]]} {{"),
  (UPat((Ops.ENDIF, Ops.ENDRANGE)), lambda ctx: "}"),
  # r method accesses
  (UPat(Ops.RANGE, name="x"), lambda ctx, x: f"for {ctx[x]} in 0..{ctx[x.src[0]]} {{"),
  (UPat(Ops.CAST, name="x"), lambda ctx, x: f"{ctx.render_cast(ctx[x.src[0]], x.src[0].dtype, x.dtype, force_cast=True)}"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx, x: f"{ctx.render_cast(ctx[x.src[0]], x.src[0].dtype, x.dtype, bitcast=True)}"),
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx, x: f"let mut {ctx[x]} = [{f'0.0_{ctx.render_dtype(x.dtype.base)}' if dtypes.is_float(x.dtype.base) else '0'}; {x.dtype.size}];"),
  (UPat(Ops.BARRIER), lambda ctx: ctx.barrier),
  (UPat(Ops.NOOP, name="x"), lambda ctx, x: ctx[x.src[0]]),
  #(UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f"{ctx.code_for_workitem[x.arg[0][0]](x.arg[0][-1])}; /* {x.arg[1]} */"),
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f"x.arg[0][0]={x.arg[0][0]} x.arg[0][-1]={x.arg[0][-1]}; /* {x.arg[1]} */"),
  # const
  (UPat(Ops.CONST, arg=math.inf, name="x"), lambda ctx, x: f"{ctx.render_dtype(x.dtype)}::INFINITY"),
  (UPat(Ops.CONST, arg=-math.inf, name="x"), lambda ctx, x: f"{ctx.render_dtype(x.dtype)}::NEG_INFINITY"),
  (UPat(Ops.CONST, arg=math.nan, name="x"), lambda ctx, x: f"{ctx.render_dtype(x.dtype)}::NAN"),
  (UPat(Ops.CONST, dtype=dtypes.float, name="x"), lambda ctx, x: f"{ctx.render_dtype(x.dtype)}::NAN" if math.isnan(x.arg) else f"{float(x.arg)}_{ctx.render_dtype(x.dtype)}"),
  (UPat(Ops.CONST, (dtypes.int64, dtypes.uint64, dtypes.int32,dtypes.uint32,), name="x"), lambda ctx, x: f"{x.arg}"),
  (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx, x: "true" if x.arg else "false"),
  # consts are rendered to larget type and casted
  (UPat(Ops.CONST, (dtypes.bfloat16, dtypes.half), name="x"), lambda ctx, x: f"{x.arg}"),
  (UPat(Ops.CONST, (dtypes.uint8, dtypes.uint16), name="x"), lambda ctx, x: f"{x.arg}"),
  (UPat(Ops.CONST, (dtypes.int8, dtypes.int16), name="x"), lambda ctx, x: f"{x.arg}"),
  # default const render
  (UPat(Ops.CONST, name="x"), lambda ctx, x: str(x.arg)),
  # new load/store
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var('idx')), allow_any_len=True),
    lambda ctx, buf, idx: f"{ctx[buf]}[{ctx[idx]} as    usize]"),
    #lambda ctx, buf, idx: f"{ctx[buf]}[{ctx[idx] if idx.op == Ops.ADD else strip_parens(ctx[idx])} as    usize]"),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), UPat(), UPat.var("gate"))).or_casted("bidx"), UPat.var("var")), allow_any_len=True),
    lambda ctx, bidx, var, gate: f"if {ctx[gate]} {{ {ctx[bidx]} }} else {{ {ctx[var]} }}"),
  (UPat(Ops.LOAD, src=(UPat.var('bidx'),), allow_any_len=True), lambda ctx, bidx: f"{ctx[bidx]}"),
  (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True), lambda ctx, bidx, var: f"{ctx[bidx]} = {ctx.render_cast(ctx[var], bidx.dtype, var.dtype)};"),
  # alu/gep
  # TODO: look for left-associative
  (UPat(GroupOp.ALU, name="x"), lambda ctx, x: ctx.code_for_op[x.op](
      *([strip_parens(ctx[v]) if v.op == x.op and x.op in {Ops.ADD, Ops.MUL, Ops.XOR, Ops.OR, Ops.AND} else ctx[v] for v in x.src]), x.dtype)),
  (UPat(Ops.GEP, name="x"), lambda ctx,x: ctx[x.src[0]] + \
    (f"[{x.arg[0]}]")),
  # (UPat(Ops.GEP, name="x"), lambda ctx,x: ctx[x.src[0]] + \
  #   (f"[{x.arg[0]}]" if x.src[0].dtype.count > ctx.gep_arr_threshold else f".{'xyzwabcd'[x.arg[0]]}")),
  # custom passes through with format

])

extra_pm = PatternMatcher([
  (UPat(Ops.BITCAST, name="x"),
    lambda x: UOp(Ops.BITCAST, x.dtype, (UOp(Ops.NOOP, x.src[0].dtype, x.src),)) if x.src[0].op not in {Ops.NOOP, Ops.LOAD, Ops.CUSTOM} else None),
  (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
])

def uops_to_dtypes(uops: List[UOp]) -> List[DType]:
  return dedup(u.dtype for u in uops if not isinstance(u.dtype, (ImageDType, PtrDType)))

class RustRenderer(Renderer):
  device = "RUST"
  has_local = False
  kernel_typedef: str = '#![feature(f16)]\n#[no_mangle]\npub extern "C" fn'
  buffer_prefix: str = "&mut "
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = ""
  barrier: str = ""
  code_for_workitem: Dict[Literal["g", "l", "i"], Callable] = {}
  extra_args: List[str] = []
  supports_float4 = False
  float4: str | None = None
  float4_style: tuple[str, str] = ('(', ')')
  gep_arr_threshold: int = 4
  type_map: Dict[DType, str] = RUST_TYPE_MAP
  infinity: str = "INFINITY"
  nan: str = "NAN"
  code_for_op: Dict = {
    Ops.SQRT: lambda x, dtype: f"{add_parens(rust_cast(x, dtype))}.sqrt()",
    Ops.RECIP: lambda x, dtype: f"1.0/{add_parens(rust_cast(x, dtype))}",
    Ops.NEG: lambda x, dtype: f"(!{x})" if dtype is dtypes.bool else f"-({x})",
    Ops.EXP2: lambda x, dtype: f"{add_parens(rust_cast(x, dtype))}.exp2()",
    Ops.LOG2: lambda x, dtype: f"{add_parens(rust_cast(x, dtype))}.log2()",
    Ops.SIN: lambda x, dtype: f"{add_parens(rust_cast(x, dtype))}.sin()",
    Ops.AND: lambda a, b, dtype: f"({a} && {b})" if dtype == dtypes.bool else f"({a} & {b})",
    Ops.XOR: lambda a, b, dtype: f"({a} ^ {rust_cast(b,dtype)})",
    Ops.OR: lambda a, b, dtype: f"({a} | {b})",
    Ops.ADD: lambda a, b, dtype: f"( {a} || {b} )" if dtype == dtypes.bool else f"({a} - {negate_const(b)})" if detect_neg_const(b) and dtypes.is_unsigned(dtype) else f"({a}+{rust_cast(b,dtype)})",
    Ops.SUB: lambda a, b, dtype: f"({rust_cast(a,dtype,force_cast=True)}).wrapping_sub({b})" if dtypes.is_int(dtype) else f"({a}-{b})",
    Ops.MUL: lambda a, b, dtype: f"({rust_cast(a, dtype, ops=True)}*{rust_cast(b, dtype, ops=True)})" if dtype != dtypes.bool else f"({a} && {b})",
    Ops.MOD: lambda a, b, dtype: f"({a}%{b})",
    Ops.IDIV: lambda a, b, dtype: f"({a}/{b})",
    Ops.CMPNE: lambda a, b, dtype: f"({add_parens(a, on_cast=True)} != {add_parens(b, on_cast=True)})",
    Ops.SHR: lambda a, b, dtype: f"({add_parens(a)}>>{b})",
    Ops.SHL: lambda a, b, dtype: f"({add_parens(a)}<<{b})",
    Ops.CMPLT: lambda a, b, dtype: f"({add_parens(a, on_cast=True)} < {add_parens(b, on_cast=True)})",
    Ops.WHERE: lambda a, b, c, dtype: f"(if {a} {{ {rust_cast(b,dtype)} }} else {{ {rust_cast(c,dtype)} }})"
  }
  string_rewrite = base_rewrite
  extra_matcher = extra_pm

  # returns a str expression of the casted xs with the given type
  def render_cast(self, x:str, src_dtype:DType, dst_dtype:DType, bitcast=False, force_cast=False) -> str:
    if DEBUG >= 6: print(f"render_cast(x={x}, src_dtype={src_dtype}, dst_dtype={dst_dtype}, bitcast={bitcast}, force_cast={force_cast}")
    if x is None:
      raise ValueError("x cannot be None")
    if bitcast and (is_float(dst_dtype) or is_float(src_dtype)):
      if is_float(src_dtype):
        val = f"{x}.to_bits()"
      else:
        val = f"{render_dtype(dst_dtype)}::from_bits({rust_cast(x,to_unsigned(dst_dtype),src_dtype=src_dtype,force_cast=True)})"
      return add_parens(rust_cast(val, dst_dtype, src_dtype, force_cast=True))
    return rust_cast(x, dst_dtype, src_dtype, force_cast=force_cast)

  # returns a str expression of the const with the given type
  def render_const(self, x:Union[float,int,bool], var_dtype) -> str:
    if math.isnan(x): val = f"{render_dtype(var_dtype)}::NAN"
    elif math.isinf(x): val = f"{render_dtype(var_dtype)}::{'NEG_INFINITY' if x < 0 else 'INFINITY'}"
    else: val = f"{float(x)}" if dtypes.is_float(var_dtype) else f"{int(x)}" if dtypes.is_int(var_dtype) else f"{bool(x)}".lower()
    return self.render_cast(val, None, var_dtype)

  # returns a str expression of the loaded value with the output type
  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    return f"{buf_name}[{self.render_index(idx)}]"

  def render_kernel(self, function_name: str, kernel: List[str], bufs: List[tuple[str, tuple[DType, bool]]], uops: List[UOp], prefix=None) -> str:
    buftypes = {}
    for name,(dtype, mutable) in bufs:
      if name in buftypes.keys():
        print(f"warning: buffer {name} is already defined {name} {dtype} {mutable} ")
        raise
      if isinstance(dtype, PtrDType):
        buftypes[name] = ("&mut " if mutable else "&")+"["+render_dtype(dtype)+f"; {dtype.size}]"
      else:
        buftypes[name] = render_dtype(dtype)

    prg = ''.join([f"{self.kernel_typedef} {function_name}(",] +
                  [', '.join([f'{name}: {t}' for name, t in buftypes.items()] + self.extra_args)] +
                  [") {\n"] + ['\n'.join(kernel), "\n}"])
    return prg if prefix is None else "\n".join(prefix) + f"\n{prg}"

  # def arender_kernel(self, function_name:str, kernel:list[str], bufs:list[tuple[str,tuple[DType,bool]]], uops:list[UOp], prefix=None) -> str:
  #   buftypes = {}
  #   for name,(dtype,mutable,var,size) in bufs:
  #     if name in buftypes.keys():
  #       print(f"warning: buffer {name} is already defined {name} {dtype} {mutable} {var} {size}")
  #       raise
  #     if var:
  #       buftypes[name] = render_dtype(dtype)
  #     else:
  #       buftypes[name] = ("&mut " if mutable else "&")+"["+render_dtype(dtype)+f"; {size}]"
  #   prg = ''.join([f"{self.kernel_prefix}fn {function_name}(",] +
  #   [', '.join([f'{name}: {t}' for name,t in buftypes.items()])] +
  #   [") {\n"] + ['\n'.join(kernel), "\n}"])
  #   return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  def render_index(self, idx:str) -> str:
    return f"({idx}) as     usize" if detect_expression(idx) else f"{idx} as     usize"

  # returns a str statement that does the store
  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx:str, idx_dtype:DType, local=False) -> str:
    return f"{buf_name}[{rust_cast(idx,idx_dtype,idx=True)}] = {rust_cast(var_name,buf_dtype,var_dtype)};"

  def render_dtype(self, dtype:DType) -> str: return render_dtype(dtype)
  def render_alu(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType) -> str:
    return f"let {buf_name}:{render_dtype(buf_dtype)} = {var_name};" if not var_dtype is dtypes.bool else f"let {buf_name} = {var_name};"

  def __getitem__(self, key): return self.r[key]

  def _render(self, uops: List[UOp]) -> tuple[str, List[str], List[tuple[str, tuple[DType, bool]]]]:
    r: Dict[UOp, str] = {}
    self.r = r
    child_count = Counter(v for ru in uops for v in ru.src)
    bufs: Dict[UOp, tuple[str, tuple[DType, bool]]] = {}
    kernel = []
    depth = 1
    c: defaultdict[str, int] = defaultdict(int)
    name = "test"

    for u in uops:
      #print(f"0r={r}")
      #if DEBUG >= 6: print(f"u={u}\n")
      if DEBUG >= 6: print(f"u.op={u.op}\n")
      if u.op is Ops.SINK:
        if u.arg is not None:
          name = u.arg.function_name
        continue
      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        r[u] = f"data{u.arg}" if u.op is Ops.DEFINE_GLOBAL else u.arg[0]
        bufs[u] = (r[u], (u.dtype, False))
        continue

      if u.op is Ops.STORE:
        for up in u.src[0].toposort():
          if up.op is Ops.DEFINE_GLOBAL:
            bufs[up] = (bufs[up][0], (bufs[up][1][0], True))

      prefix = None
      if u.op is Ops.SPECIAL:
        r[u] = u.arg[0]
      elif u.op is Ops.RANGE:
        r[u] = f"ridx{u.arg}"
      else:
        prefix = {Ops.WMMA: "wmma", Ops.DEFINE_LOCAL: "temp", Ops.CONST: "const",
                  Ops.CAST: "cast", Ops.BITCAST: "cast", Ops.GEP: "gep", Ops.VECTORIZE: "cast",
                  Ops.INDEX: "bidx", Ops.DEFINE_REG: "acc", Ops.LOAD: "val"}.get(u.op, "alu")
        if DEBUG >= 6: print(f"  prefix={prefix}")
        if DEBUG >= 6: print(f"  c={c}")
        if DEBUG >= 6: print(f"  c[prefix]={c[prefix]}")
        r[u] = f"{prefix}{c[prefix]}"

      l = cast(str, self.string_rewrite.rewrite(u, ctx=self))
      assert l is not None, f"failed to render {u.op} {u.dtype} {[(x.op, x.dtype) for x in u.src]} {u.arg}\n u={u}"

      if u.op in {Ops.ENDIF, Ops.ENDRANGE}:
        depth -= 1
      if (u.op is not Ops.CAST or u.dtype.count == 1) and (u.op in {Ops.CONST, Ops.GEP, Ops.INDEX, Ops.CUSTOMI} or
        (u.op in {Ops.VECTORIZE, *(GroupOp.ALU - {Ops.WHERE}), Ops.CAST, Ops.BITCAST} and
          child_count[u] == 1 and not getenv("EXPAND_SSA"))):
        r[u] = l
      else:
        if u.op in {Ops.RANGE, Ops.ASSIGN, Ops.DEFINE_LOCAL, Ops.STORE} or u.dtype == dtypes.void:
          if u.op is Ops.ASSIGN:
            r[u] = r[u.src[0]]
        elif u.op is Ops.DEFINE_REG:
          l = f"let mut {r[u]} = {l};"
        else:
          l = f"let {r[u]} = {l};" if u.op is not Ops.SPECIAL else l
        kernel.append("  " * depth + l)
        if prefix:
          c[prefix] += 1
      if u.op in {Ops.IF, Ops.RANGE}:
        depth += 1
      if DEBUG >= 6: print(f"  l={l}")
      #if DEBUG >= 6: print(f"  kernel={kernel}")
      #print(f"  1r={r}")
      if DEBUG >= 6: print("\n\n")
    del self.r
    return (name, kernel, list(bufs.values()))

  def render(self, uops: list[UOp]) -> str:
    return self.render_kernel(*self._render(uops), uops)
