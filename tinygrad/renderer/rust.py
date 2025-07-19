from typing import Dict, List, Tuple, Union, Literal, Callable, cast
import math, re, os
from collections import defaultdict, Counter
from tinygrad.uop.ops import GroupOp, Ops, UOp, PatternMatcher, UPat
from tinygrad.helpers import strip_parens, getenv, prod, dedup, DEBUG
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType, to_dtype
from tinygrad.renderer.cstyle import CStyleLanguage

RUST_TYPE_MAP = {
    "float64": "f64", "double": "f64",
    "half": "f16", "half2": "Float16x2", "half4": "Float16x4",
    "float": "f32", "float2": "Float32x2", "float4": "Float32x4",
    "signed char": "i8", "int8": "i8", "char": "i8",
    "short": "i16", "int16": "i16",
    "int": "i32", "int32": "i32",
    "long": "i64",
    "unsigned char": "u8", "uint8": "u8", "uchar": "u8",
    "unsigned short": "u16", "uint16": "u16", "ushort": "u16",
    "unsigned int": "u32", "uint32": "u32", "uint": "u32",
    "unsigned long": "u64", "ulong": "u64",
    "bool": "bool", "void": "void"
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
def add_parens(x:str, on_cast:bool=False) -> str:
  return f"({x})" if detect_expression(x) and not (x[0] == '(' and x[-1] == ')') else f"({x})" if (detect_as_cast(x) and on_cast) else f"{x}"
def render_dtype(var_dtype:DType) -> str:
  if var_dtype.name not in RUST_TYPE_MAP:
    raise ValueError(f"Unknown dtype: {var_dtype} name: {var_dtype.name}")
  return RUST_TYPE_MAP[var_dtype.name]
def is_positive_integer(x:str) -> bool: return bool(re.search(r'^\d+$', x))
def is_float(dtype) -> bool: return False if dtype is None else dtypes.is_float(dtype)
def is_floatx(dtype) -> bool: return False if dtype is None else (dtype.name in ["float4","float2","half4","half2"])
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
    val = f"{add_parens(val)}" if not is_float(dst_dtype) else f"{add_parens(add_parens(val)+" as usize")}"
    force_cast = True
  if idx: return f"{val}" if is_positive_integer(val) else f"{add_parens(val)} as usize"
  if detect_neg_const(x) and detect_expression(x) and not detect_as_cast(x) and is_unsigned(src_dtype):
    val = f"{rust_cast(val,to_signed(src_dtype),force_cast=True)}"
  if is_unsigned(dst_dtype) != is_unsigned(src_dtype):
    val = f"{val} as {render_dtype(dst_dtype)}"
  return f"{val} as {render_dtype(dst_dtype)}" if force_cast else val

rust_rewrite = PatternMatcher([
  (UPat(Ops.DEFINE_REG, name="x"), lambda ctx, x: f"let mut {ctx[x]}:{render_dtype(x.dtype)} = {ctx[x.src[0]]}"),
  (UPat(Ops.ASSIGN, name="x"), lambda ctx, x: f"{ctx[x.src[0]]} = {ctx[x.src[1]]};"),
  (UPat(Ops.IF, name="x"), lambda ctx, x: f"if {ctx[x.src[0]]} {{"),
  (UPat((Ops.ENDIF, Ops.ENDRANGE)), lambda ctx: "}"),
  (UPat(Ops.WMMA, name="x"), lambda ctx,x: f"__{x.arg[0]}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]})"),
  (UPat(Ops.RANGE, name="x"), lambda ctx, x: f"for {ctx[x]} in 0..{ctx[x.src[0]]} {{"),
  (UPat(Ops.VECTORIZE, name="x"),
   lambda ctx,x: f"{ctx.float4_style[0]}{','.join([ctx[y] for y in x.src])}{ctx.float4_style[1]}"),
  (UPat(Ops.CAST, name="x"), lambda ctx, x: f"{ctx.render_cast(ctx[x.src[0]], x.src[0].dtype, x.dtype, force_cast=True)}"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx, x: f"{ctx.render_cast(ctx[x.src[0]], x.src[0].dtype, x.dtype, bitcast=True)}"),
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx, x: f"let mut {ctx[x]} = [{f'0.0_{ctx.render_dtype(x.dtype.base)}' if dtypes.is_float(x.dtype.base) else '0'}; {x.dtype.size}];"),
  (UPat(Ops.BARRIER), lambda ctx: ctx.barrier),
  (UPat(Ops.NOOP, name="x"), lambda ctx, x: ctx[x.src[0]]),
  (UPat(Ops.WHERE, name="x"), lambda ctx, x: f"(if {ctx[x.src[0]]} {{ {ctx.render_cast(ctx[x.src[1]], x.src[1].dtype, x.dtype)} }} else {{ {ctx.render_cast(ctx[x.src[2]], x.src[2].dtype, x.dtype)} }})"),
  (UPat(Ops.XOR, name="x"), lambda ctx, x: f"({ctx[x.src[0]]} ^ {ctx.render_cast(ctx[x.src[1]], x.src[1].dtype, x.dtype)})"),
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f"x.arg[0][0]={x.arg[0][0]} x.arg[0][-1]={x.arg[0][-1]}; /* {x.arg[1]} {x.flargp} */"),
  (UPat(Ops.CONST, arg=math.inf, name="x"), lambda ctx, x: f"{ctx.render_dtype(x.dtype)}::INFINITY"),
  (UPat(Ops.CONST, arg=-math.inf, name="x"), lambda ctx, x: f"{ctx.render_dtype(x.dtype)}::NEG_INFINITY"),
  (UPat(Ops.CONST, arg=math.nan, name="x"), lambda ctx, x: f"{ctx.render_dtype(x.dtype)}::NAN"),
  (UPat(Ops.CONST, dtype=dtypes.float, name="x"), lambda ctx, x: f"{ctx.render_dtype(x.dtype)}::NAN" if math.isnan(x.arg) else f"{float(x.arg)}_{ctx.render_dtype(x.dtype)}"),
  (UPat(Ops.CONST, (dtypes.int64, dtypes.uint64, dtypes.int32,dtypes.uint32,), name="x"), lambda ctx, x: f"{x.arg}"),
  (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx, x: "true" if x.arg else "false"),
  (UPat(Ops.CONST, name="x"), lambda ctx, x: f"{ctx.render_dtype(x.dtype)}::NAN" if math.isnan(x.arg) else f"{x.arg}"),
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var('idx')), allow_any_len=True),
    lambda ctx, buf, idx: ctx.render_index(ctx[buf], buf.dtype, ctx[idx], idx.dtype)),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), UPat(), UPat.var("gate"))).or_casted("bidx"), UPat.var("var")), allow_any_len=True),
    lambda ctx, bidx, var, gate: f"if {ctx[gate]} {{ {ctx.render_cast(ctx[bidx],bidx.dtype,bidx.dtype)} }} else {{ {ctx.render_cast(ctx[var],var.dtype,var.dtype)} }}"),
  (UPat(Ops.LOAD, name='x'), lambda ctx, x: f"let {ctx[x]}:{render_dtype(x.dtype)} = {ctx[x.src[0]]}"),
  (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True), lambda ctx, bidx, var: ctx._render_store(bidx,var)),
  (UPat(Ops.STORE, name='x'), lambda ctx, x: ctx._render_store2(ctx[x], x.dtype, ctx[x.src[0]], x.src[0].dtype)),
  (UPat(GroupOp.ALU, name="x"), lambda ctx, x: ctx.code_for_op[x.op](
      *([strip_parens(ctx[v]) if v.op == x.op and x.op in {Ops.ADD, Ops.MUL, Ops.XOR, Ops.OR, Ops.AND} else ctx[v] for v in x.src]), x.dtype)),
  (UPat(Ops.GEP, name="x"), lambda ctx,x: ctx[x.src[0]] + \
    (f".0[{x.arg[0]}]" if is_floatx(x.src[0].dtype) else f"[{x.arg[0]}]")),
])

rust_extra_pm = PatternMatcher([
  (UPat(Ops.BITCAST, name="x"),
    lambda x: UOp(Ops.BITCAST, x.dtype, (UOp(Ops.NOOP, x.src[0].dtype, x.src),)) if x.src[0].op not in {Ops.NOOP, Ops.LOAD, Ops.CUSTOM} else None),
  (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
])

def uops_to_dtypes(uops: List[UOp]) -> List[DType]:
  return dedup(u.dtype for u in uops if not isinstance(u.dtype, (ImageDType, PtrDType)))

class RustRenderer(CStyleLanguage):
  device = "RUST"
  has_local = False
  kernel_typedef: str = '#[no_mangle]\npub extern "C" fn'
  buffer_prefix: str = "&mut "
  supports_float4: bool = True
  float4_style = ('[', ']')
  gep_arr_threshold = 0
  code_for_op: dict = {
    Ops.SQRT: lambda x, dtype: f"{add_parens(rust_cast(x, dtype, None))}.sqrt()",
    Ops.RECIP: lambda x, dtype: f"1.0/{add_parens(rust_cast(x, dtype))}",
    Ops.NEG: lambda x, dtype: f"(!{x})" if dtype is dtypes.bool else f"-({x})",
    Ops.EXP2: lambda x, dtype: f"{add_parens(rust_cast(x, dtype))}.exp2()",
    Ops.LOG2: lambda x, dtype: f"{add_parens(rust_cast(x, dtype))}.log2()",
    Ops.SIN: lambda x, dtype: f"{add_parens(rust_cast(x, dtype))}.sin()",
    Ops.AND: lambda a, b, dtype: f"({a} && {b})" if dtype == dtypes.bool else f"({a} & {b})",
    Ops.OR: lambda a, b, dtype: f"({add_parens(a)} | {add_parens(b)})",
    Ops.ADD: lambda a, b, dtype: f"( {a} || {b} )" if dtype == dtypes.bool else f"({a} - {-int(b)})" if detect_neg_const(b) and dtypes.is_unsigned(dtype) else f"({a}+{rust_cast(b,dtype)})",
    Ops.SUB: lambda a, b, dtype: f"({rust_cast(a,dtype,force_cast=True)}).wrapping_sub({b})" if dtypes.is_int(dtype) else f"({a}-{b})",
    Ops.MUL: lambda a, b, dtype: f"({rust_cast(a, dtype, ops=True)}*{rust_cast(b, dtype, ops=True)})" if dtype != dtypes.bool else f"({a} && {b})",
    Ops.MOD: lambda a, b, dtype: f"({a}%{b})",
    Ops.IDIV: lambda a, b, dtype: f"({a}/{b})",
    Ops.CMPNE: lambda a, b, dtype: f"({add_parens(a, on_cast=True)} != {add_parens(b, on_cast=True)})",
    Ops.SHR: lambda a, b, dtype: f"({add_parens(a)}>>{b})",
    Ops.SHL: lambda a, b, dtype: f"({add_parens(a)}<<{b})",
    Ops.CMPLT: lambda a, b, dtype: f"({add_parens(a, on_cast=True)} < {add_parens(b, on_cast=True)})"
  }
  string_rewrite = rust_rewrite
  #extra_matcher = rust_extra_pm

  def _render_store(self, d, s) -> str:
    if DEBUG >= 6: print(f"_render_store()")
    src = self[s]
    src_dtype = s.dtype
    dst = None
    dst_dtype = d.dtype
    try:
      # cast0 which isn't gonna work for us here going to figure out what the underlying thing is
      if d.op == Ops.CAST and d.src[0].op == Ops.INDEX and d.src[0].src[0].op == Ops.DEFINE_GLOBAL:
        idx = None
        if d.src[0].src[1].op == Ops.CONST:
          idx = f"{int(d.src[0].src[1].arg)}..{int(d.src[0].src[1].arg)+s.dtype.count}"
        else:
          sidx = f"{self[d.src[0].src[1]]}"
          if not sidx.endswith("as usize)") and not sidx.endswith("as usize"):
            sidx = f"({sidx} as usize)"
          idx = f"{sidx}..{sidx}+{s.dtype.count}"
        if idx is not None: dst = f"data{d.src[0].src[0].arg}[{idx}]"
      if dst is None: raise
      return f"{dst}.copy_from_slice(&{self.render_cast(src, d.dtype, src_dtype)}.0);"
    except:
      dst = self[d]
    #f"{'*' if bidx.dtype.size == -1 else ''}{ctx[bidx]} = {ctx.render_cast(ctx[var], bidx.dtype, var.dtype)};"
    if is_floatx(src_dtype) and dst.startswith(self.render_dtype(src_dtype)):
      return f"{self.floatx_isolate_array(dst, src_dtype)}.copy_from_slice(&{self.render_cast(src, dst_dtype, src_dtype)}.0);"
    return f"{dst} = {self.render_cast(src, dst_dtype, src_dtype)};"

  def render_index(self, x:str, xdtype:DType, i:str, idtype:DType) -> str:
    if DEBUG >= 6: print(f"render_index(x={x}, xdtype={xdtype}, i={i}, idtype={idtype}")
    if xdtype.size < 0:
      if str(i) != "0":
        self.tweak['unsafe'] = True
        return f"*{x}.add({i}{' as usize' if bool(re.search(r'\D', i)) else ''})"
      else: return f"*{x}"
    if is_positive_integer(i): return f"{x}[{i}]"
    return f"{x}[{i} as usize]"

  def floatx_isolate_array(self, x:str,dtype:DType) -> str:
    match = re.match(r"^([a-zA-Z][a-zA-Z0-9]*\[.*\]).try_into\(\).unwrap\(\)\)$", x.lstrip(f"{self.render_dtype(dtype)}("))
    if match is None: raise ValueError(f"Invalid floatx input: {x}")
    return match.group(1)

  def floatx_rewrite_input(self, x:str, dst_dtype:DType) -> str:
    match = re.match(r"^[a-zA-Z][a-zA-Z0-9]*\[(.*)\]$", x)
    #if match is None: raise ValueError(f"Invalid float4 input: {x}")
    if match is None: return x
    idx = match.group(1)
    if is_positive_integer(idx): return x.replace(f"[{idx}]", f"[{idx}..{int(idx)+dst_dtype.count}]")
    match = re.fullmatch(r"\((\w+)(?:\+(\d+))?\) as usize", idx)
    if match:
        var, offset = match.groups()
        offset = int(offset or 0)+dst_dtype.count
        idx2 = f"({var}+{offset}) as usize"
        return x.replace(f"[{idx}]", f"[{add_parens(idx)}..{add_parens(idx2)}]")
    return x.replace(f"[{idx}]", f"[{add_parens(idx)}..{add_parens(idx)}+{dst_dtype.count}]")

  def render_cast(self, x:str, src_dtype:DType, dst_dtype:DType, bitcast=False, force_cast=False, preservenumber=False) -> str:
    if DEBUG >= 6: print(f"render_cast(x={x}, src_dtype={src_dtype}, dst_dtype={dst_dtype}, bitcast={bitcast}, force_cast={force_cast}, preservenumber={preservenumber}")
    if DEBUG >= 6: print(f" isinstance(dst_dtype, PtrDType) {isinstance(dst_dtype, PtrDType)} is_floatx(dst_dtype) {is_floatx(dst_dtype)}")
    if DEBUG >= 6: print(f" isinstance(src_dtype, PtrDType) {isinstance(src_dtype, PtrDType)} is_floatx(src_dtype) {is_floatx(src_dtype)}")

    if x is None:
      raise ValueError("x cannot be None")
    if x.startswith("if"): return x
    if bitcast and (is_float(dst_dtype) or is_float(src_dtype)):
      if is_float(src_dtype): val = f"{x}.to_bits()"
      else:
        val = f"{render_dtype(dst_dtype)}::from_bits({rust_cast(x,to_unsigned(dst_dtype),src_dtype=src_dtype,force_cast=True)})"
      return add_parens(rust_cast(val, dst_dtype, src_dtype, force_cast=True))

    if is_floatx(dst_dtype):
      if isinstance(src_dtype, PtrDType):
        y = self.floatx_rewrite_input(x, dst_dtype)
        if y == x and not "[" in x: return f"{y}"
      if x.startswith(render_dtype(dst_dtype)): return x
      if bool(re.search(r'^[a-z][a-z0-9]*$',x)): return x
      if isinstance(dst_dtype, PtrDType) and not isinstance(src_dtype, PtrDType):
        print(f"WARNING: unsafe")
        return f"unsafe {{ &mut *({self.floatx_rewrite_input(x, dst_dtype)}.as_mut_ptr() as *mut {render_dtype(dst_dtype)}) }}"
      return f"{render_dtype(dst_dtype)}({self.floatx_rewrite_input(x, dst_dtype)}.try_into().unwrap())"

    if preservenumber and detect_numeric(x):
      return x
    return rust_cast(x, dst_dtype, src_dtype, force_cast=force_cast)

  def render_kernel(self, function_name: str, kernel: List[str], bufs: List[tuple[str, tuple[DType, bool]]], uops: List[UOp], prefix=None) -> str:
    struct_defs = set()
    buftypes = {}
    for name,(dtype, mutable) in bufs:
      if name in buftypes.keys():
        print(f"warning: buffer {name} is already defined {name} {dtype} {mutable} ")
        raise
      if isinstance(dtype, PtrDType):
        if self.tweak['unsafe']:
          buftypes[name] = ("*mut " if mutable else "*") + (render_dtype(dtype) if dtype.size == -1 else f"[{render_dtype(dtype)}; {dtype.size}]")
        else:
          buftypes[name] = ("&mut " if mutable else "&") + (render_dtype(dtype) if dtype.size == -1 else f"[{render_dtype(dtype)}; {dtype.size}]")
      else:
        buftypes[name] = render_dtype(dtype)

    # hack to allow us to reuse CstyleLanuage Renderer, Rust handles loads and store differently than _render() hardcode
    for i,line in enumerate(kernel):
      if len(line.split(" = let")) > 1: kernel[i] = f"{' '*(len(line)-len(line.lstrip()))}let{line.split(" = let")[1]}" #DEFINE_REG, DEFINE_LOCAL,LOAD
      elif bool(re.search(r'^\s+Float\d+', line)):
        src = re.sub(r'^(\s*)(\w+)\s+(\w+)$', r'\1let \3:\2', line.split(' = ')[0])
        sz,l = map(int, re.match(r'^\s+Float(\d+)x(\d+)', line).groups())
        dtype = dtypes.half.vec(l) if sz == 16 else (dtypes.float.vec(l) if sz == 32 else None)
        if dtype is None: raise NotImplementedError(f"unknown Float type in: {line} a: {sz} {l}")
        dst = self.render_cast(line.split(' = ')[1].rstrip(';'), dtype, dtype)
        kernel[i] = f"{src} = {dst};"
      else: kernel[i] = re.sub(r'^(\s*)(\w+)\s+(\w+)\s*=', r'\1let \3:\2 =', line)

    for dt in uops_to_dtypes(uops):
      # emit struct types like Float32x4 etc if needed
      if dt.count > 1:
        vecname = f"{render_dtype(dt)}"
        base = render_dtype(dt.scalar())
        struct_defs.add(f"#[repr(align(16))]\n#[derive(Clone, Copy)]\nstruct {vecname}([{base}; {dt.count}]);\n")
      if render_dtype(dt) == "f16" or render_dtype(dt).startswith("Float16"): self.tweak["kernel"]["#![feature(f16)]"] = True

    preamble = "\n".join([k for k,v in self.tweak["kernel"].items() if v]) + "\n"
    preamble += "\n".join(sorted(struct_defs)) + "\n" if struct_defs else ""

    ktype = self.kernel_typedef.replace('pub','pub unsafe') if self.tweak['unsafe'] else self.kernel_typedef
    if self.tweak['unsafe']: print("WARNING: unsafe function")
    prg = ''.join([preamble, f"{ktype} {function_name}(",] +
                  [', '.join([f'{name}: {t}' for name, t in buftypes.items()] + self.extra_args)] +
                  [") {\n"] + ['\n'.join(kernel), "\n}"])
    if DEBUG >= 6: print(f"prg={prg}")
    import sys
    sys.stdout.flush()
    return prg if prefix is None else "\n".join(prefix) + f"\n{prg}"

  def render_dtype(self, dtype:DType, mutable=True) -> str: return render_dtype(dtype)

  def _render(self, uops:list[UOp]) -> tuple[str, list[str], list[tuple[str,tuple[DType,bool]]]]:
    self.tweak = {'unsafe': False, 'kernel': {'#![feature(f16)]':False}}
    r: dict[UOp, str] = {}
    self.r = r

    child_count = Counter(v for ru in uops for v in ru.src)
    bufs: dict[UOp, tuple[str, tuple[DType, bool]]] = {}
    kernel = []
    depth = 1
    c: defaultdict[str, int] = defaultdict(int)
    name = "test"
    for u in uops:
      #print(f"0r={r}")
      if DEBUG >= 6: print(f"{'!'*70}\nu={u}\n")
      if DEBUG >= 6: print(f"u.op={u.op}\n")
      if u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
        continue
      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        r[u] = f"data{u.arg}" if u.op is Ops.DEFINE_GLOBAL else u.arg[0]
        bufs[u] = (r[u], (u.dtype, False))
        continue

      # mark buffers that we store to writable
      if u.op is Ops.STORE:
        for up in u.src[0].toposort():
          if up.op is Ops.DEFINE_GLOBAL: bufs[up] = (bufs[up][0], (bufs[up][1][0], True))

      # naming
      prefix = None
      if u.op is Ops.SPECIAL: r[u] = u.arg[0]
      elif u.op is Ops.RANGE: r[u] = f"ridx{u.arg}"
      else:
        prefix = {Ops.WMMA: "wmma", Ops.DEFINE_LOCAL: "temp", Ops.CONST: "const",
                  Ops.CAST: "cast", Ops.BITCAST: "cast", Ops.GEP: "gep", Ops.VECTORIZE: "cast", Ops.NOOP: "precast",
                  Ops.INDEX: "bidx", Ops.DEFINE_REG: "acc", Ops.LOAD: "val"}.get(u.op, "alu")
        if DEBUG >= 6: print(f"  prefix={prefix}")
        if DEBUG >= 6: print(f"  c={c}")
        if DEBUG >= 6: print(f"  c[prefix]={c[prefix]}")
        r[u] = f"{prefix}{c[prefix]}"

      l = cast(str, self.string_rewrite.rewrite(u, ctx=self))
      assert l is not None, f"failed to render {u.op} {u.dtype} {[(x.op,x.dtype) for x in u.src]} {u.arg}"

      if u.op in {Ops.ENDIF, Ops.ENDRANGE}: depth -= 1
      if (u.op is not Ops.CAST or u.dtype.vcount == 1) and (u.op in {Ops.CONST, Ops.GEP, Ops.INDEX, Ops.CUSTOMI} or \
        (u.op in {Ops.VECTORIZE, *(GroupOp.ALU-{Ops.WHERE}), Ops.CAST, Ops.BITCAST} and child_count[u] == 1 and not getenv("EXPAND_SSA"))):
        r[u] = l
      else:
        if u.op in {Ops.RANGE, Ops.ASSIGN, Ops.DEFINE_LOCAL, Ops.STORE} or u.dtype == dtypes.void:
          if u.op is Ops.ASSIGN: r[u] = r[u.src[0]]
        # elif u.op is Ops.DEFINE_REG:
        #   l = f"let mut {r[u]}:{render_dtype(u.dtype)} = {l};"
        # else:
        #   l = f"let {r[u]}:{render_dtype(u.dtype)} = {l};"
        #   if u.op is Ops.SPECIAL: raise NotImplementedError(f"special op {u.op} {u.arg}")
        else:
          l = f"{self.render_dtype(u.dtype)} {r[u]} = {l}" + (";" if u.op is not Ops.SPECIAL else "")
        kernel.append("  "*depth + l)
        if prefix: c[prefix] += 1  # if it was used, increment
      if u.op in {Ops.IF, Ops.RANGE}: depth += 1
      if DEBUG >= 6: print(f"  l={l}")
      #if DEBUG >= 6: print(f"  kernel={kernel}")
      #print(f"  1r={r}")
      if DEBUG >= 6: print("\n\n")
    del self.r

    # NOTE: this relies on bufs dict preserving order
    return (name, kernel, list(bufs.values()))

