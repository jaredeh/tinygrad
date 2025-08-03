from typing import Dict, List, Tuple, Union, Literal, Callable, cast
import math, re, os
from collections import defaultdict, Counter
from tinygrad.uop.ops import GroupOp, Ops, UOp, PatternMatcher, UPat
from tinygrad.helpers import strip_parens, getenv, prod, dedup, DEBUG
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType, to_dtype
from tinygrad.renderer.cstyle import CStyleLanguage

RUST_TYPE_MAP = {
    "signed char": "i8", "short": "i16", "int": "i32", "long": "i64",
    "unsigned char": "u8", "unsigned short": "u16", "unsigned int": "u32", "unsigned long": "u64",
    "half": "f16", "half2": "Float16x2", "half4": "Float16x4",
    "float": "f32", "float2": "Float32x2", "float4": "Float32x4",
    "double": "f64", "bool": "bool", "void": "void"
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
def is_float(dtype) -> bool: return False if dtype is None else dtypes.is_float(dtype) or dtypes.is_float(dtype.base)
def is_floatx(dtype) -> bool: return False if dtype is None else (dtype.name in ["float4","float2","half4","half2"])
def is_bool(dtype) -> bool: return False if dtype is None else dtypes.is_bool(dtype) or dtype.name == "bool"
def is_unsigned(dtype) -> bool: return False if dtype is None else dtypes.is_unsigned(dtype)
def rust_cast(x:str, dst_dtype:DType, src_dtype:DType=None, force_cast=False, idx=False, ops=False):
  if os.environ.get("RUSTDEBUG", False): print(f"rust_cast: {x}, {dst_dtype}, {src_dtype}, {force_cast}, {idx}, {ops}")
  #print(f"dst_dtype: {x}, {dst_dtype.max}")
  val = x
  if detect_numeric(x):
    if is_float(dst_dtype) or is_float(src_dtype) or ops:
      val = f"{f"{x}.0" if is_positive_integer(x) and is_float(dst_dtype) else x}_{render_dtype(dst_dtype)}"
    elif is_bool(dst_dtype):
      val = "false" if x in ["0", "0.0"] else "true"
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
  (UPat(Ops.DEFINE_REG, name="x"), lambda ctx, x: f"let mut {ctx[x]}:{render_dtype(x.dtype)};" if not isinstance(x.dtype, PtrDType) else f"let mut {ctx[x]}:[{render_dtype(x.dtype)}; {x.dtype.size}] = [{rust_cast("0", x.dtype)}; {x.dtype.size}];"),
  (UPat(Ops.IF, name="x"), lambda ctx, x: f"if {ctx[x.src[0]]} {{"),
  (UPat((Ops.ENDIF, Ops.ENDRANGE)), lambda ctx: "}"),
  (UPat(Ops.WMMA, name="x"), lambda ctx,x: f"__{x.arg[0]}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]})"),
  (UPat(Ops.RANGE, name="x"), lambda ctx, x: f"for {ctx[x]} in 0..{ctx[x.src[0]]} {{"),
  (UPat(Ops.VECTORIZE, name="x"),
   lambda ctx,x: f"{ctx.float4_style[0]}{','.join([ctx[y] for y in x.src])}{ctx.float4_style[1]}"),
  (UPat(Ops.CAST, name="x"), lambda ctx, x: f"{ctx.render_cast(ctx[x.src[0]], x.src[0].dtype, x.dtype, force_cast=True)}"),
  (UPat(Ops.PRECAST, name="x"), lambda ctx,x: ctx[x.src[0]]),
  (UPat(Ops.BITCAST, name="x"), lambda ctx, x: f"{ctx.render_cast(ctx[x.src[0]], x.src[0].dtype, x.dtype, bitcast=True)}"),
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx, x: f"let mut {ctx[x]} = [{f'0.0_{ctx.render_dtype(x.dtype.base)}' if dtypes.is_float(x.dtype.base) else '0'}; {x.dtype.size}];"),
  (UPat(Ops.BARRIER), lambda ctx: ctx.barrier),
  (UPat(Ops.WHERE, name="x"), lambda ctx, x: f"(match {ctx[x.src[0]]} {{ true => {ctx.render_cast(ctx[x.src[1]], x.src[1].dtype, x.dtype)}, false => {ctx.render_cast(ctx[x.src[2]], x.src[2].dtype, x.dtype)} }})"),
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

  def _render_store(self, d, s) -> str:
    if os.environ.get("RUSTDEBUG", False): print(f"_render_store()")
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
    if is_floatx(src_dtype) and dst.startswith(self.render_dtype(src_dtype)):
      return f"{self.floatx_isolate_array(dst, src_dtype)}.copy_from_slice(&{self.render_cast(src, dst_dtype, src_dtype)}.0);"
    return f"{dst} = {self.render_cast(src, dst_dtype, src_dtype)};"

  def render_index(self, x:str, xdtype:DType, i:str, idtype:DType) -> str:
    if os.environ.get("RUSTDEBUG", False): print(f"render_index(x={x}, xdtype={xdtype}, i={i}, idtype={idtype}")
    if xdtype.size < 0:
      if str(i) != "0":
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
    if os.environ.get("RUSTDEBUG", False): print(f"render_cast(x={x}, src_dtype={src_dtype}, dst_dtype={dst_dtype}, bitcast={bitcast}, force_cast={force_cast}, preservenumber={preservenumber}")
    if os.environ.get("RUSTDEBUG", False): print(f" isinstance(dst_dtype, PtrDType) {isinstance(dst_dtype, PtrDType)} is_floatx(dst_dtype) {is_floatx(dst_dtype)}")
    if os.environ.get("RUSTDEBUG", False): print(f" isinstance(src_dtype, PtrDType) {isinstance(src_dtype, PtrDType)} is_floatx(src_dtype) {is_floatx(src_dtype)}")

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
    # Check for unsafeness
    unsafe = False
    for line in kernel:
      if "unsafe" in line: unsafe = True; break

    # Process input buffers
    buftypes = {}
    for name,(dtype, mutable) in bufs:
      if name in buftypes.keys():
        print(f"warning: buffer {name} is already defined {name} {dtype} {mutable} ")
        raise
      if isinstance(dtype, PtrDType):
        if dtype.size == -1:
          unsafe = True
          buftypes[name] = ("*mut " if mutable else "*const ") + (render_dtype(dtype) if dtype.size == -1 else f"[{render_dtype(dtype)}; {dtype.size}]")
        else:
          buftypes[name] = ("&mut " if mutable else "&") + (render_dtype(dtype) if dtype.size == -1 else f"[{render_dtype(dtype)}; {dtype.size}]")
      else:
        buftypes[name] = render_dtype(dtype)

    # Hack to allow us to reuse CstyleLanuage Renderer, Rust handles loads and store differently than _render() hardcode
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

    # Walk uops graph to check for features and struct defs
    struct_defs = set()
    features = set()
    #features.add("#![no_builtins]\n")
    for dt in uops_to_dtypes(uops):
      # emit struct types like Float32x4 etc if needed
      if dt.count > 1:
        vecname = f"{render_dtype(dt)}"
        base = render_dtype(dt.scalar())
        struct_defs.add(f"#[repr(align({dt.count*dt.itemsize}))]\n#[derive(Clone, Copy)]\nstruct {vecname}([{base}; {dt.count}]);\n")
      # add f16 feature if needed
      if render_dtype(dt) == "f16" or render_dtype(dt).startswith("Float16"): features.add("#![feature(f16)]\n")

    preamble = "\n".join(features) + "\n" + f"{"\n".join(sorted(struct_defs)) + "\n" if struct_defs else ""}"
    ktype = self.kernel_typedef.replace('pub','pub unsafe') if unsafe else self.kernel_typedef

    # Piece together kernel
    prg = ''.join([preamble, f"{ktype} {function_name}(",] +
                  [', '.join([f'{name}: {t}' for name, t in buftypes.items()] + self.extra_args)] +
                  [") {\n"] + ['\n'.join(kernel), "\n}"])
    if os.environ.get("RUSTDEBUG", False): print(f"prg={prg}")
    return prg if prefix is None else "\n".join(prefix) + f"\n{prg}"

  def render_dtype(self, dtype:DType, mutable=True) -> str: return render_dtype(dtype)
