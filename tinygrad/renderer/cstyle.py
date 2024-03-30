from typing import Dict, List, Optional, NamedTuple, Tuple, Union, DefaultDict, cast, Literal, Callable
import math, functools
from collections import defaultdict, Counter
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.helpers import strip_parens, getenv, prod
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType
from tinygrad.codegen.uops import UOpGraph
from tinygrad.renderer.base import BaseLanguage


class CStyleLanguage(BaseLanguage):
  buffer_prefix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = "const int"
  barrier: str = ""
  end_token: Optional[str] = "}"
  code_for_workitem: Dict[Union[Literal["g"], Literal["l"], Literal["i"]], Callable] = {}
  global_max: List[int] = []
  local_max: List[int] = []
  extra_args: List[str] = []
  float4: Optional[str] = None
  uses_vload: bool = False
  uses_ptr_arithmetic: bool = False
  type_map: Dict[DType, str] = {}
  code_for_op: Dict = {
    UnaryOps.NEG: lambda x,dtype: f"(!{x})" if dtype is dtypes.bool else f"(-{x})", UnaryOps.SQRT: lambda x,dtype: f"sqrt({x})",
    UnaryOps.EXP2: lambda x,dtype: f"exp2({x})", UnaryOps.LOG2: lambda x,dtype: f"log2({x})", UnaryOps.SIN: lambda x,dtype: f"sin({x})",
    BinaryOps.ADD: lambda a,b,dtype: f"({a}+{b})", BinaryOps.SUB: lambda a,b,dtype: f"({a}-{b})", BinaryOps.MUL: lambda a,b,dtype: f"({a}*{b})",
    BinaryOps.DIV: lambda a,b,dtype: f"({a}/{b})", BinaryOps.MAX: lambda a,b,dtype: f"max({a},{b})", BinaryOps.MOD: lambda a,b,dtype: f"({a}%{b})",
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
    else: val = f"{float(x)}f" if dtypes.is_float(var_dtype) else f"{int(x)}" if dtypes.is_int(var_dtype) else f"{bool(x)}".lower()
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
      out_val = f"*(({self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix}{self.render_dtype(buf_dtype)}{output_dtype.count}*)({buf_name}+{idx}))"  # noqa: E501
    else:
      out_val = f"*({buf_name}+{idx})" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}]"
    return self.render_cast([out_val], output_dtype) if output_dtype != buf_dtype else out_val

  def get_kernel_modifier(self, uops:UOpGraph) -> str: return ""
  def render_kernel(self, function_name:str, uops:UOpGraph, prefix=None) -> str:
    tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,(dtype,_) in self.bufs) else ""  # noqa: E501
    buftypes = [(name,f"{'write_only' if mutable else 'read_only'} image2d_t" if dtype.name.startswith('image') else
                ("" if mutable else "const ")+self.buffer_prefix+self.render_dtype(dtype)+"*"+self.buffer_suffix if isinstance(dtype, PtrDType) else
                self.arg_int_prefix if dtype == dtypes.int else None) for name,(dtype,mutable) in self.bufs]
    prg = ''.join([f"{self.kernel_prefix}void {self.get_kernel_modifier(uops)}{function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(self.kernel), "\n}"])
    return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  # returns a str statement that does the store
  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx:str, idx_dtype:DType, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert var_dtype == dtypes.float.vec(4), f"images must be float4, getting {var_dtype}"
      return f"write_imagef({buf_name}, {idx}, {var_name});"
    if self.uses_vload and buf_dtype.scalar() == dtypes.float16 and var_dtype.scalar() != dtypes.float16:
      return f"vstore_half{'' if var_dtype.count == 1 else str(var_dtype.count)}({var_name}, 0, {buf_name}+{idx});"
    if var_dtype.count > 1:
      prefix = self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix
      return f"*(({prefix}{self.render_dtype(buf_dtype)}{var_dtype.count}*)({buf_name}+{idx})) = {var_name};"
    return f"*({buf_name}+{idx}) = {var_name};" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}] = {var_name};"

  def render_local(self, name:str, dtype:DType, size:int): return self.smem_align + self.smem_prefix + f"{self.render_dtype(dtype)} {name}[{size}];"
  def render_dtype(self, var_dtype:DType) -> str: return self.type_map.get(var_dtype, var_dtype.name)
  def render_loop(self, u): return f"for (int {(expr := self.ssa(u,'ridx'))} = {self.r[u.vin[0]]}; {expr} < {self.r[u.vin[1]]}; {expr}++) {{"
  def uop_special(self, u):
    self.kk(f"int {u.arg[1]} = {self.code_for_workitem[u.arg[1][0]](u.arg[0])}; /* {u.arg[2]} */")
    self.r[u] = u.arg[1]

  def uop_alu(self, u):
    # remove parens if ALU types are the same. TODO: can do more here
    if u.arg in {BinaryOps.ADD,BinaryOps.MUL,BinaryOps.XOR}: operands = [strip_parens(self.r[v]) if v.arg == u.arg else self.r[v]for v in u.vin]
    else: operands = [self.r[v] for v in u.vin]
    val = self.code_for_op[u.arg](*operands, u.dtype)
    assert self.child_count[u] != 0, f"childless ALU op found {u}"
    # TODO: fix index rendering issue. fix clang nested max macro issue
    if self.child_count[u] <= 1 and u.arg != BinaryOps.MAX and not getenv("EXPAND_SSA"): self.r[u] = val
    else: self.kk(f"{self.render_dtype(u.dtype)} {self.ssa(u,'alu')} = {val};")

  def uop_load(self, u):
    val = self.render_load(u.dtype, self.r[u.vin[0]], u.vin[0].dtype, strip_parens(self.r[u.vin[1]]), u.vin[0].uop is UOps.DEFINE_LOCAL)
    # NOTE: this relies on the load not happening if it's in the unselected branch
    if len(u.vin) > 3: val = self.code_for_op[TernaryOps.WHERE](self.r[u.vin[2]], val, self.r[u.vin[3]], u.dtype)
    self.kk(f"{self.render_dtype(u.dtype)} {self.ssa(u,'val')} = {val};")

  def uop_phi(self, u):
    self.kk(f"{self.r[u.vin[0]]} = {self.r[u.vin[1]]};")
    self.r[u] = self.r[u.vin[0]]

  def uop_cast(self, u, bitcast=False):
    if bitcast:
      assert len(u.vin) == 1
      precast = self.ssa(None,'precast')
      self.kk(f"{self.render_dtype(cast(DType, u.vin[0].dtype))} {precast} = {r[u.vin[0]]};")
      val = self.render_cast([precast], u.dtype, bitcast=True)
    else:
      val = self.render_cast([self.r[x] for x in u.vin], u.dtype, bitcast=False)
    if self.child_count[u] <= 1: self.r[u] = val
    else: self.kk(f"{self.render_dtype(u.dtype)} {self.ssa(u,'cast')} = {val};")

  def uop_wmma(self, u): self.kk(f"{self.render_dtype(u.dtype)} {self.ssa(u, 'wmma')} = {u.arg}({self.r[u.vin[0]]}, {self.r[u.vin[1]]}, {self.r[u.vin[2]]});")
  def uop_define_acc(self, u): self.kk(f"{self.render_dtype(u.dtype)} {self.ssa(u,'acc')} = {self.render_const(u.arg, u.dtype)};")
  def uop_const(self, u):  self.r[u] = self.render_const(u.arg, u.dtype) if u.arg >= 0 else f"({self.render_const(u.arg, u.dtype)})"
  def uop_gep(self, u):
    assert u.vin[0].dtype is not None
    from_ssa = u.vin[0].uop in {UOps.LOAD, UOps.WMMA, UOps.DEFINE_ACC}
    self.r[u] = (self.r[u.vin[0]] if from_ssa else f"{(self.r[u.vin[0]])}") + (f"[{u.arg}]" if u.vin[0].dtype.count > 4 else f".{'xyzw'[u.arg]}")

class OpenCLLanguage(CStyleLanguage):
  kernel_prefix = "__kernel "
  buffer_prefix = "__global "
  smem_align = "__attribute__ ((aligned (16))) "
  smem_prefix = "__local "
  barrier = "barrier(CLK_LOCAL_MEM_FENCE);"
  float4 = "(float4)"
  code_for_workitem = {"g": lambda x: f"get_group_id({x})", "l": lambda x: f"get_local_id({x})", "i": lambda x: f"get_global_id({x})"}
  uses_vload = True
  type_map = { dtypes.uint8: "uchar", dtypes.uint32: "uint", dtypes.uint16: "ushort", dtypes.uint64: "ulong" }
  def render_cast(self, x, var_dtype, bitcast=False) -> str:
    return f"as_{self.render_dtype(var_dtype)}({x[0]})" if bitcast else super().render_cast(x, var_dtype)

  def render_kernel(self, function_name, uops, prefix=None) -> str:
    if any(uop.dtype == dtypes.half for uop in uops): prefix = ["#pragma OPENCL EXTENSION cl_khr_fp16 : enable"]
    return super().render_kernel(function_name, uops, prefix)

class MetalLanguage(CStyleLanguage):
  kernel_prefix = "kernel "
  buffer_prefix = "device "
  smem_prefix = "threadgroup "
  arg_int_prefix = "constant int&"
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
  float4 = "float4"
  uses_ptr_arithmetic = True
  code_for_workitem = {"g": lambda x: f"gid.{chr(120+x)}", "l": lambda x: f"lid.{chr(120+x)}"}
  extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']
  type_map = {dtypes.bfloat16: "bfloat"}
  code_for_op = {**CStyleLanguage().code_for_op,
    BinaryOps.MAX: lambda a,b,dtype: f"(bfloat)max((float){a},(float){b})" if dtype == dtypes.bfloat16 else f"max({a},{b})",
    UnaryOps.SQRT: lambda x,dtype: f"(bfloat)sqrt({x})" if dtype == dtypes.bfloat16 else f"sqrt({x})",
    UnaryOps.EXP2: lambda x,dtype: f"(bfloat)exp2({x})" if dtype == dtypes.bfloat16 else f"exp2({x})",
    UnaryOps.LOG2: lambda x,dtype: f"(bfloat)log2({x})" if dtype == dtypes.bfloat16 else f"log2({x})",
    UnaryOps.SIN: lambda x,dtype: f"(bfloat)sin({x})" if dtype == dtypes.bfloat16 else f"sin({x})",}

  def render_cast(self, x: List[str], var_dtype: DType, bitcast=False) -> str:
    return f"as_type<{self.render_dtype(var_dtype)}>({x[0]})" if bitcast else super().render_cast(x, var_dtype)

  def render_kernel(self, function_name, uops, prefix=None):
    prefix = ["#include <metal_stdlib>","using namespace metal;"]
    if any(uop.uop == UOps.WMMA for uop in uops): prefix.append("""template<typename T, typename S, typename U> U __metal_wmma(T m, T n, U o) {
    S a,b,c; a.thread_elements()[0] = m.x; a.thread_elements()[1] = m.y; b.thread_elements()[0] = n.x; b.thread_elements()[1] = n.y;
    c.thread_elements()[0] = o.x; c.thread_elements()[1] = o.y; simdgroup_multiply_accumulate(c, a, b, c);
    return U(c.thread_elements()[0], c.thread_elements()[1]);\n}""")
    return super().render_kernel(function_name, uops, prefix)

code_for_op_half = {BinaryOps.MAX: lambda a,b,dtype: f"__hmax({a},{b})" if dtype in (dtypes.half, dtypes.bfloat16) else f"max({a},{b})",
                    UnaryOps.SQRT: lambda x,dtype: f"hsqrt({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sqrt({x})",
                    UnaryOps.SIN: lambda x,dtype: f"hsin({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sin({x})",
                    UnaryOps.LOG2: lambda x,dtype: f"hlog2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"log2({x})",
                    UnaryOps.EXP2: lambda x,dtype: f"hexp2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"exp2({x})",}

class CUDALanguage(CStyleLanguage):
  kernel_prefix = "extern \"C\" __global__ "
  smem_prefix = "__shared__ "
  smem_prefix_for_cast = False
  barrier = "__syncthreads();"
  float4 = "make_float4"
  code_for_workitem = {"g": lambda x: f"blockIdx.{chr(120+x)}", "l": lambda x: f"threadIdx.{chr(120+x)}",
                       "i": lambda x: f"(blockIdx.{chr(120+x)}*blockDim.{chr(120+x)}+threadIdx.{chr(120+x)})"}
  code_for_op = {**CStyleLanguage().code_for_op, **code_for_op_half}
  type_map = {dtypes.bfloat16: "nv_bfloat16"}

  def render_kernel(self, function_name, uops, prefix=None):
    prefix = ["#define INFINITY (__int_as_float(0x7f800000))","#define NAN (__int_as_float(0x7fffffff))"]
    if any(uop.dtype == dtypes.half for uop in uops):
      prefix += ["#include <cuda_fp16.h>", "struct half4 { half x, y, z, w; };", "struct half8 { half x, y, z, w, a, b, c, d; };",
      "__device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x, y, z, w}; return r; }",
      "__device__ half8 make_half8(half x, half y, half z, half w, half a, half b, half c, half d) { half8 r={x, y, z, w, a, b, c, d}; return r; }",
      """__device__ float4 __cuda_mma_m16n8k16_f16_f32(half8 a, half4 b, float4 c) { int *a_pk = (int *) (&a), *b_pk = (int *) (&b);
  asm( "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
    : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w) : "r"(a_pk[0]), "r"(a_pk[1]), "r"(a_pk[2]),  "r"(a_pk[3]), "r"(b_pk[0]), "r"(b_pk[1]) );
  return c;}""",]

    if any(uop.dtype == dtypes.bfloat16 for uop in uops): prefix.append("#include <cuda_bf16.h>")
    return super().render_kernel(function_name, uops, prefix=prefix)

code_for_op_hip = { UnaryOps.SQRT: lambda x,dtype: f"__ocml_sqrt_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                    UnaryOps.SIN: lambda x,dtype: f"__ocml_sin_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                    UnaryOps.LOG2: lambda x,dtype: f"__ocml_log2_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                    UnaryOps.EXP2: lambda x,dtype: f"__ocml_exp2_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                    # TODO: MAX with int uses fmax_f32?
                    BinaryOps.MAX: lambda a,b,dtype: f"__ocml_fmax_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32) }({a},{b})",}

def _make_hip_code_for_op():
  def wrapper(key, func):
    def cast_bf16(*args):
      if args[-1] == dtypes.bfloat16:
        operands = tuple(f"(float)({arg})" for arg in (args[1:-1] if key is TernaryOps.WHERE else args[:-1]))
        return f"(hip_bfloat16)({func(*(((args[0],) if key is TernaryOps.WHERE else ()) + operands), dtypes.float)})"
      return func(*args)
    return cast_bf16
  return { k:wrapper(k,v) for k,v in {**CStyleLanguage().code_for_op, **code_for_op_hip}.items() }

def _make_hip_dtype(base_type, name, cnt):
  nms = "xyzwabcdefghijkl"[:cnt]
  return f"typedef {base_type} {name}{cnt} __attribute__((ext_vector_type({cnt})));\n" + \
         f"static inline __attribute__((device)) {name}{cnt} make_{name}{cnt}(" + ', '.join([f"{base_type} {x}" for x in nms]) + \
         ") { return {" + ', '.join(nms) + "}; }"

class HIPLanguage(CStyleLanguage):
  kernel_prefix = """
  #define half _Float16

  extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_id(unsigned int);
  extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_group_id(unsigned int);
  extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_size(unsigned int);

  extern "C" {
  __attribute__((device)) __attribute__((const)) float __ocml_fmax_f32(float, float);
  __attribute__((device)) __attribute__((pure)) float __ocml_exp2_f32(float);
  __attribute__((device)) __attribute__((pure)) float __ocml_log2_f32(float);
  __attribute__((device)) float __ocml_sin_f32(float);
  __attribute__((device)) __attribute__((const)) float __ocml_sqrt_f32(float);
  __attribute__((device)) __attribute__((const)) double __ocml_fmax_f64(double, double);
  __attribute__((device)) __attribute__((pure)) double __ocml_exp2_f64(double);
  __attribute__((device)) __attribute__((pure)) double __ocml_log2_f64(double);
  __attribute__((device)) double __ocml_sin_f64(double);
  __attribute__((device)) __attribute__((const)) double __ocml_sqrt_f64(double);
  __attribute__((device)) __attribute__((const)) _Float16 __ocml_fmax_f16(_Float16, _Float16);
  __attribute__((device)) __attribute__((pure)) _Float16 __ocml_exp2_f16(_Float16);
  __attribute__((device)) __attribute__((pure)) _Float16 __ocml_log2_f16(_Float16);
  __attribute__((device)) _Float16 __ocml_sin_f16(_Float16);
  __attribute__((device)) __attribute__((const)) _Float16 __ocml_sqrt_f16(_Float16);
    }\n""" + '\n'.join([_make_hip_dtype(*x) for x in [
                     ("_Float16", "half", 2), ("_Float16", "half", 4), ("_Float16", "half", 8), ("_Float16", "half", 16),
                     ("float", "float", 8)]]) + """
  static __attribute__((device)) half8 __hip_wmma_f16_f16(half16 a, half16 b, half8 c) {
    half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { c_frag[n*2] = c[n]; }
    c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, false);
    for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;
  }\nextern "C" __attribute__((global))"""
  code_for_workitem = {"g": lambda x: f"__ockl_get_group_id({x})", "l": lambda x: f"__ockl_get_local_id({x})",
                       "i": lambda x: f"(__ockl_get_group_id({x})*__ockl_get_local_size({x})+__ockl_get_local_id({x}))"}
  code_for_op = _make_hip_code_for_op()
  smem_prefix = "__attribute__((shared))"
  barrier = '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");' + '__builtin_amdgcn_s_barrier();' + \
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");'
  float4 = "make_float4"
  uses_ptr_arithmetic = False  # NOTE: this fixes TestLinearizerOverflowAlt
  type_map = {dtypes.bfloat16: "hip_bfloat16"}

  def render_kernel(self, function_name, uops, prefix=None) -> str:
    prefix = ["#define INFINITY (__builtin_inff())\n#define NAN (__builtin_nanf(\"\"))",
              "typedef long unsigned int size_t;"]
    if any(uop.dtype == dtypes.bfloat16 for uop in uops): prefix.append("""
struct hip_bfloat16 {
  unsigned short data;
  __attribute__((device)) hip_bfloat16(float val) {
    union { float fp32; unsigned int u32; } u = {val};
    if (~u.u32 & 0x7f800000) { u.u32 += 0x7fff + ((u.u32 >> 16) & 1); } else if (u.u32 & 0xffff) { u.u32 |= 0x10000; }
    data = (u.u32 >> 16);
  }
  __attribute__((device)) operator float() const {
    unsigned int uval = data << 16;
    return *reinterpret_cast<float*>(&uval);
  }
};
static __attribute__((device)) bool operator<(hip_bfloat16 a, hip_bfloat16 b) { return ((float)a) < ((float)b); }
static __attribute__((device)) bool operator==(hip_bfloat16 a, hip_bfloat16 b) { return ((float)a) == ((float)b); }
""")
    prefix.append('\n'.join(_make_hip_dtype(*x) for x in [("float", "float", 2), ("float", "float", 4),
                                                             ("signed int", "int", 4), ("signed int", "int", 2)]))
    return super().render_kernel(function_name, uops, prefix)

  def get_kernel_modifier(self, uops:UOpGraph) -> str:
    requiredMaxThreadsPerBlock = prod(u.arg[2] for u in uops if u.uop == UOps.SPECIAL and u.arg[1][0] == "l")
    # https://clang.llvm.org/docs/AttributeReference.html#amdgpu-flat-work-group-size
    # NOTE: this makes hlb_cifar10 twice as fast, there may be more gains in tweaking these parameters
    return f"__attribute__((amdgpu_flat_work_group_size(1, {requiredMaxThreadsPerBlock})))"
