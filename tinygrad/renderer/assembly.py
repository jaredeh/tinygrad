from typing import Callable, DefaultDict, Dict, List, Union, NamedTuple, Set, Optional
import functools, struct
from collections import defaultdict
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.dtype import dtypes, DType, PtrDType, INVERSE_DTYPES_DICT
from tinygrad.codegen.uops import UOpGraph, PatternMatcher

def render_val(x, dtype):
  if dtypes.is_float(dtype):
    if dtype == dtypes.double: return "0d%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
    elif dtype == dtypes.half: return "0x%02X%02X" % tuple(struct.pack("e",x)[::-1])
    return "0f%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
  return str(int(x)) + ("U" if dtypes.is_unsigned(dtype) else "")

class AssemblyLanguage(NamedTuple):
  kernel_prefix: str = ""
  barrier: str = ""
  load_global: bool = False
  label_prefix: str = ""
  gid: List[str] = []
  gdim: List[str] = []
  lid: List[str] = []
  const_requires_mov: List[DType] = [] # list of dtypes for which creating a const requires a move
  asm_for_op: Dict[Op, Callable[...,str]] = {}
  types: Dict[DType, str] = INVERSE_DTYPES_DICT
  supports_half: List[Op] = []
  matcher = Optional[PatternMatcher] = None

  def render_const(self, x:Union[float,int,bool], dtype, mov=None) -> Union[List[str], str]: raise NotImplementedError()
  def render_local(self, dest, name, size, dtype) -> List[str]: raise NotImplementedError()

  def render_loop(self, idx, start, label, acc=None) -> List[str]: raise NotImplementedError()
  def render_bra(self, b1, pred=None, b2=None) -> List[str]: raise NotImplementedError()
  def render_gep(self, loc, base, offset, dtype, gate=None) -> List[str]: raise NotImplementedError()
  def render_load(self, loc, dest, dtype, gate=None, alt=None, ss="", offset=0) -> List[str]: raise NotImplementedError()
  def render_store(self, loc, val, dtype, gate=None, ss="", offset=0) -> List[str]: raise NotImplementedError()
  def render_cast(self, d:str, a:str, dtype:DType, atype:DType, bitcast=False, pred=False) -> List[str]: raise NotImplementedError()

  def render_kernel(self, kernel, function_name, bufs, regs) -> str: raise NotImplementedError()
  def mem_type(self, dtype) -> str: raise NotImplementedError()


  def reset(self, uops:UOpGraph, outputs, inputs) -> None:
    super().reset(uops, outputs, inputs)
    self.c_label: DefaultDict[str, int] = defaultdict(int)
    self.r_label: Dict[UOp, str] = {}
    # here we do a pretransform on UOps to fix some shortcomings of PTX
    # all uops must be a register
    self.replace: Dict[UOp, UOp] = {}
    self.seen: Set[UOp] = set()

    def eq_rep(root, x, y):
      root.arg = BinaryOps.XOR
      new = uops.add(UOps.ALU, dtypes.bool, (root,), arg=UnaryOps.NEG, insert_before=uops.uops.index(root)+1)
      return new

    def lt_rep(x, y):
      new = uops.add(UOps.ALU, dtypes.bool, (u.vin[0],), arg=UnaryOps.NEG, insert_before=uops.uops.index(u))
      u.vin = (new, u.vin[1])
      u.arg = BinaryOps.MUL

    def ld_rep(root, x, y):
      root.dtype = dtypes.uint8
      new = uops.add(UOps.CAST, dtypes.bool, (root,), insert_before=uops.uops.index(root)+1)
      ptr_ar(root)
      return new

    def gate_rep(root, x, y, z, k):
      new = uops.add(UOps.CAST, dtypes.uint8, (k,), insert_before=uops.uops.index(root))
      root.vin = (x,y,z,new)
      return ld_rep(root,x,y)

    def ptr_ar(root):
      assert root.arg in {'.shared', '.global', None}
      if root.arg is None: root.arg = '.shared' if root.vin[0].uop == UOps.DEFINE_LOCAL else '.global'  # move this to the argL
      val = uops.add(UOps.CONST, dtypes.int, tuple(), arg=root.vin[0].dtype.itemsize, insert_before=uops.uops.index(root))
      ptr = uops.add(UOps.ALU, dtypes.int, (root.vin[1], val), arg=BinaryOps.MUL, insert_before=uops.uops.index(root))
      if ptr.uop == UOps.CONST: root.vin = (root.vin[0], ptr) + root.vin[2:]
      else:
        zero = uops.add(UOps.CONST, dtypes.int, tuple(), arg=0, cachable=False, insert_before=uops.uops.index(root))
        bptr = uops.add(UOps.CAST, dtypes.uint64, (ptr,), insert_before=uops.uops.index(root))
        fptr = uops.add(UOps.ALU, dtypes.uint64, (root.vin[0], bptr), arg=BinaryOps.ADD, insert_before=uops.uops.index(root))
        root.vin = (fptr, zero) + root.vin[2:]

    self.matcher = PatternMatcher([
      ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.CMPEQ, "vin": ({"__name__": "x", "dtype": dtypes.bool},{"__name__": "y"})}, eq_rep),
      ({"uop": UOps.ALU, "arg": BinaryOps.CMPLT, "vin": ({"__name__": "x", "dtype": dtypes.bool},{"__name__": "y"})}, lt_rep),
      ({"__name__": "root", "uop": UOps.LOAD,"dtype": dtypes.bool,
        "vin": ({"__name__": "x"},{"__name__": "y"},{"__name__": "z"},{"__name__": "k"})}, gate_rep),
      ({"__name__": "root", "uop": UOps.LOAD,"dtype": dtypes.bool, "vin": ({"__name__": "x"},{"__name__": "y"})}, ld_rep),
      ({"__name__": "root", "uop": UOps.STORE, "vin": {}}, ptr_ar),
      ({"__name__": "root", "uop": UOps.LOAD, "vin": {}}, ptr_ar),
    ])

    for u in uops:
      if u in self.seen: continue
      self.seen.add(u)
      for o,n in self.replace.items():
        if o in u.vin and u is not n:
          u.vin = tuple(n if x == o else x for x in u.vin)
      if rew := self.matcher.rewrite(u): self.replace[u] = rew
    uops.remove_childless(set(x for x in uops if x.uop in {UOps.DEFINE_GLOBAL, UOps.PHI, UOps.ENDIF, UOps.ENDLOOP, UOps.STORE}))

  def ssa_label(self, u, prefix):
    self.c_label[prefix] += 1
    self.r_label[u] = f"{self.label_prefix}{prefix}_{self.c_label[prefix]-1}"
    return self.r_label[u]

  def const(self, x:Union[float,int,bool], dtype, mov=False):
    if mov or dtype in self.const_requires_mov:
      self.kk(*self.render_const(x, dtype, mov=(out:=self.ssa(None, 'const', self.types[dtype]))))
      return out
    return self.render_const(x, dtype)

  def cast(self, a, dtype:DType, atype:DType, bitcast=False, u=None, pred=False):
    if atype == dtype:
      if u: self.r[u] = a
      return a
    self.kk(*self.render_cast((ret:=self.ssa(u, 'cast', self.types[dtype])), a, dtype, atype, bitcast))
    return ret

  def render_if(self, u):
    assert u.vin[0].dtype is not None
    return self.render_bra(lb:=self.ssa_label(u, 'if'), self.cast(self.r[u.vin[0]], dtypes.bool, u.vin[0].dtype, u=u, pred=True), f"{lb}_true"), f"{lb}_true:")

  def render_phi(self, u): return f"mov.b{self.types[u.dtype][1:]} {self.r[u.vin[0]]}, {self.r[u.vin[1]]};"

  def render_define_var(self, u): return self.render_load(u.arg.expr, self.ssa(u, 'dat', dtype=self.types[u.dtype]), u.dtype, ss=".param")

  def render_define_global(self, u):
    dt = u.dtypes.ulong if u.dtype.__class__ == PtrDType else u.dtype
    return self.render_load(u.arg[1], self.ssa(u, 'dat', dtype=self.types[dt]), dt, ss=".param")

  def uop_endloop(self, u):
    self.kk(self.asm_for_op[BinaryOps.ADD](self.r[u.vin[0]], self.r[u.vin[0]], "1", dtypes.int, self.types[dtypes.int]),
        self.asm_for_op[BinaryOps.CMPLT](pred:=self.ssa(None, "pred", "pred"), self.r[u.vin[0]], self.r[u.vin[0].vin[1]], dtypes.int, self.types[dtypes.int]))
    self.kk(*self.render_bra(self.r_label[u.vin[0]], pred, f"{self.r_label[u.vin[0]]}_exit"), f"{self.r_label[u.vin[0]]}_exit:")

  def uop_endif(self, u): self.kk(f"{self.r_label[u.vin[0]]}:")

  def uop_store(self, u):
    assert u.vin[0].dtype is not None and u.vin[1].dtype is not None and u.vin[2].dtype is not None
    if u.vin[2].dtype.count > 1:
      self.kk((f"@{r[u.vin[3]]} " if len(u.vin)>3 else "") + \
          f"st{u.arg}.v{u.vin[2].dtype.count}.{self.mem_type(u.vin[2].dtype.scalar())} [{self.r[u.vin[0]]}+{u.vin[1].arg}], {{{', '.join(self.r[u.vin[2]])}}};")
    else:
      self.kk(*self.render_store(self.r[u.vin[0]], self.r[u.vin[2]], u.vin[2].dtype, gate=self.r[u.vin[3]] if len(u.vin)>3 else None, ss=u.arg, offset=u.vin[1].arg))

  def uop_alu(self, u):
    assert u.vin[0].dtype is not None
    operands = [self.r[x] for x in u.vin]
    lab = self.ssa(u, "alu")
    if needs_upcast := dtype == dtypes.half and u.arg not in self.supports_half:
      dtype = dtypes.float32
      out_lab, lab = lab, self.ssa(None, "alu_cast", self.types[dtype])
      for i, op in enumerate(operands):
        operands[i] = self.ssa(None, "alu_cast", self.types[dtype])
        self.kk(*self.render_cast(operands[i], op, dtype, dtypes.half)) # type: ignore
    if u.arg == BinaryOps.CMPLT or u.arg == BinaryOps.CMPEQ:
      # pass in the other dtype here
      self.kk(self.asm_for_op[u.arg](lab, *operands, u.vin[0].dtype, self.types[u.vin[0].dtype]))
    else:
      self.kk(self.asm_for_op[u.arg](lab, *operands, dtype, self.types[dtype]))
    if needs_upcast:
      self.kk(*self.render_cast(out_lab, lab, dtypes.half, dtype))

  def uop_define_acc(self, u):
    if u.dtype.count > 1:
      self.r[u] = [self.ssa(None, 'acc', self.types[u.dtype.scalar()]) for _ in range(u.dtype.count)]
      for uu in self.r[u]: self.kk(f"mov.b{self.types[u.dtype.scalar()][1:]} {uu}, {self.const(u.arg, u.dtype.scalar())};")
    else: self.kk(f"mov.b{self.types[u.dtype][1:]} {self.ssa(u, 'acc')}, {self.const(u.arg, u.dtype)};")

  def uop_special(self, u):
    assert u.arg[1][0] != "i", "idx not supported"
    self.kk(f"mov.u32 %{u.arg[1]}, {(self.gid if u.arg[1][0] == 'g' else self.lid)[u.arg[0]]};")
    self.r[u] = "%" + u.arg[1]
    self.kernel = [f".reg .u32 %{u.arg[1]};"] + self.kernel

  def uop_const(self, u):
    if u.dtype.count > 1: self.r[u] = [self.const(u.arg, u.dtype.scalar(), mov=True) for _ in range(u.dtype.count)]
    else: self.r[u] = self.const(u.arg, u.dtype, mov=True)

  def uop_gep(self, u): self.r[u] = self.r[u.vin[0]][u.arg]

  def uop_load(self, u):
          assert u.vin[1].dtype is not None
          if u.dtype.count > 1:
            self.r[u] = [self.ssa(None, 'val', self.types[u.dtype.scalar()]) for _ in range(u.dtype.count)]
            if(len(u.vin)>3):
              for v in self.r[u]: self.kk(f"mov.{self.mem_type(u.dtype.scalar())} {v}, {render_val(0, u.dtype.scalar())};")
            self.kk((f"@{self.r[u.vin[2]]}"if len(u.vin) > 3 else "")
              + f" ld{u.arg}.v{u.dtype.count}.{self.mem_type(u.dtype.scalar())} {{{', '.join(self.r[u])}}}, [{self.r[u.vin[0]]}+{u.vin[1].arg}];")
          else:
            self.kk(*self.render_load(self.r[u.vin[0]], self.ssa(u, 'val'), u.dtype, gate=self.r[u.vin[2]] if len(u.vin) > 3 else None,
                                alt=self.r[u.vin[3]] if len(u.vin) > 3 else None, ss=u.arg, offset=u.vin[1].arg))

  def uop_cast(self, u, bitcast=False):
    assert u.vin[0].dtype is not None
    if u.dtype.count>1: self.r[u] = [self.r[x] for x in u.vin] # type: ignore
    else: self.cast(self.r[u.vin[0]], u.dtype, u.vin[0].u.dtype, bitcast=u.uop is UOps.BITCAST, u=u)

  def uop_define_local(self, u):
    # TODO: we should sum these, and fetch 0xC000 from somewhere
    assert u.arg[1]*u.dtype.itemsize <= 0xC000, "too large local"
    self.kk(*self.render_local(self.ssa(u, 'local', self.types[dtypes.ulong]), u.arg[0], u.arg[1], u.dtype))

class PTXLanguage(AssemblyLanguage):
  kernel_prefix = """.version VERSION
.target TARGET
.address_size 64
.visible .entry"""
  barrier = "bar.sync\t0;"
  has_pred = True
  load_global = True
  label_prefix = "$"
  gid = [f'%ctaid.{chr(120+i)}' for i in range(3)]
  gdim = [f'%nctaid.{chr(120+i)}' for i in range(3)]
  lid = [f'%tid.{chr(120+i)}' for i in range(3)]
  asm_for_op = {
    UnaryOps.NEG: lambda d,a,dt,name: f"not.pred {d}, {a};" if name == "pred" else f"neg.{name} {d}, {a};",
    UnaryOps.EXP2: lambda d,a,dt,name: f"ex2.approx.{name} {d}, {a};", UnaryOps.LOG2: lambda d,a,dt,name: f"lg2.approx.{name} {d}, {a};",
    UnaryOps.SIN: lambda d,a,dt,name: f"sin.approx.{name} {d}, {a};", UnaryOps.SQRT: lambda d,a,dt,name: f"sqrt.approx.{name} {d}, {a};",
    BinaryOps.ADD: lambda d,a,b,dt,name: f"{'or' if name == 'pred' else 'add'}.{name} {d}, {a}, {b};",
    BinaryOps.SUB: lambda d,a,b,dt,name: f"sub.{name} {d}, {a}, {b};",
    BinaryOps.MUL: lambda d,a,b,dt,name: ('and' if dt == dtypes.bool else 'mul') + f"{'.lo' if dtypes.is_int(dt) else ''}.{name} {d}, {a}, {b};",
    BinaryOps.XOR: lambda d,a,b,dt,name: f"xor.pred {d}, {a}, {b};" if name == "pred" else f"xor.b{name[1:]} {d}, {a}, {b};",
    BinaryOps.DIV: lambda d,a,b,dt,name: f"div{'.approx' if dtypes.is_float(dt) else ''}.{name} {d}, {a}, {b};",
    BinaryOps.MAX: lambda d,a,b,dt,name: f"max.{name} {d}, {a}, {b};", BinaryOps.MOD: lambda d,a,b,dt,name: f"rem.{name} {d}, {a}, {b};",
    BinaryOps.CMPLT: lambda d,a,b,dt,name: f"setp.lt.{name} {d}, {a}, {b};",
    BinaryOps.CMPEQ: lambda d,a,b,dt,name: f"setp.eq.{name} {d}, {a}, {b};",
    TernaryOps.WHERE: lambda d,a,b,c,dt,name:
      f"@{a} mov.{name} {d}, {b};\n@!{a} mov.{name} {d}, {c};" if name == "pred" else f"selp.{'b16' if name == 'f16' else name} {d}, {b}, {c}, {a};"
  }
  supports_half = [UnaryOps.NEG, UnaryOps.EXP2, BinaryOps.ADD, BinaryOps.SUB, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPLT, TernaryOps.WHERE]
  # HACK: Use s16 and u16 for int8 and uint8 buffers. This can be wrong in cast.
  types = { dtypes.int8: "s16", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
            dtypes.uint8: "u16", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
            dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "pred" }

  const_requires_mov = [dtypes.half, dtypes.bool]

  def render_const(self, x:Union[float,int,bool], dtype, mov=None) -> Union[List[str], str]:
    val = render_val(x, dtype)
    if dtype == dtypes.bool: return [f"setp.ne.s16 {mov}, {val}, 0;"]
    return [f"mov.b{self.types[dtype][1:]} {mov}, {val};"] if mov else val

  def render_local(self, dest, name, size, dtype) -> List[str]:
    return [f".shared .align 4 .b8 {name}[{size*dtype.itemsize}];", f"mov.u64 {dest}, {name}[0];"]

  def render_loop(self, u) -> List[str]: return [f"mov.u32 {self.ssa(u, 'ridx')}, {self.r[u.vin[0]]};", f"{self.ssa_label(u, 'loop')}:"]

  def render_bra(self, b1, pred=None, b2=None) -> List[str]: return [f"@{pred} bra {b1};", f"@!{pred} bra {b2};"] if pred else [f"bra {b1};"]

  def mem_type(self, dtype): return 's8' if dtype.itemsize == 1 else 'b16' if dtype == dtypes.float16 else self.types[dtype]

  def render_load(self, loc, dest, dtype, gate=None, alt=None, ss="", offset=0) -> List[str]:
    assert dtype is not dtypes.bool
    if gate: return [f"@{gate} ld{ss}.{self.mem_type(dtype)} {dest}, [{loc}+{offset}];", f"@!{gate} mov.b{self.types[dtype][1:]} {dest}, {alt};"]
    else: return [f"ld{ss}.{self.mem_type(dtype)} {dest}, [{loc}+{offset}];"]

  def render_store(self, loc, val, dtype, gate=None, ss="", offset=0) -> List[str]:
    if dtype == dtypes.bool: return [f".reg .s16 {val}_cast;", *self.render_cast(f"{val}_cast", val, dtypes.int16, dtype),
                                     (f"@{gate} " if gate else "") + f"st{ss}.{self.mem_type(dtype)} [{loc}+{offset}], {val}_cast;"]
    return [(f"@{gate} " if gate else "") + f"st{ss}.{self.mem_type(dtype)} [{loc}+{offset}], {val};"]

  def render_cast(self, d:str, a:str, dtype:DType, atype:DType, bitcast=False, pred=False) -> List[str]:
    if bitcast: return [f"mov.b{self.types[dtype][1:]} {d}, {a};"]
    if atype == dtypes.bool: return[f"selp.b{self.types[dtype][1:]} {d}, {render_val(1, dtype)}, {render_val(0, dtype)}, {a};"]
    if dtype == dtypes.bool: return [f"setp.ne.b{self.types[atype][1:]} {d}, {a}, {self.render_const(0, atype)};"]
    rnd = ('.rzi' if dtypes.is_int(dtype) and dtypes.is_float(atype) else
           '.rn' if dtypes.is_float(dtype) and (dtype.itemsize < atype.itemsize or dtypes.is_int(atype) or atype == dtypes.bool) else '')
    return [f"cvt{rnd}.{self.types[dtype]}.{self.types[atype]} {d}, {a};"]

  def render_kernel(self, kernel, function_name, bufs, regs) -> str:
    kernel = [f".reg .{reg.split('_')[-2]} %{reg}<{cnt}>;" for reg,cnt in regs] + kernel + ["ret;"]
    def fmt(line): return line if line[0]=="$" else "\t" + line.replace(" ", "\t" if len(line.split(" ")[0]) > 7 else "\t\t", 1)
    return (f"{self.kernel_prefix} {function_name}(\n\t" +
            ',\n\t'.join([f".param .{'u64' if dtype.__class__ == PtrDType else self.types[dtype]} {name}" for name,dtype in bufs]) + "\n)\n{\n" +
            '\n'.join([fmt(line) for op in kernel for line in op.splitlines()]) +
            "\n}")

PTXRenderer = functools.partial(uops_to_asm, PTXLanguage())
