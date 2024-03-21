# An example to compile a small Tensorflow model to extremely portable C code

import os, sys
os.environ["CLANG"] = '1'

import numpy as np
import subprocess
from extra.onnx import get_run_onnx
from tinygrad.tensor import Tensor
from extra.export_model import export_model_clang, compile_net, jit_model
import onnx
import onnxruntime as ort


class TinyOnnx:
  def __init__(self, onnx_model):
    self.xname = onnx_model.graph.input[0].name
    self.yname = onnx_model.graph.output[0].name
    self.run_onnx = get_run_onnx(onnx_model)

  def forward(self, x):
    return self.run_onnx({self.xname: x}, debug=False)[self.yname]


def onnx_get_dimensions(onnx_tensor):
  # TODO: Is this actually the only way to get this?
  tensor_data_type = onnx_tensor.type.tensor_type.elem_type
  data_type_str = onnx.TensorProto.DataType.Name(tensor_data_type)

  shape = {"dims": [], "data_type": data_type_str}
  for dim in onnx_tensor.type.tensor_type.shape.dim:
    if dim.dim_value:
      shape["dims"].append(dim.dim_value)
  return shape


def onnx_get_shapes(onnx_model):
  inputs_shape = []
  for input_tensor in onnx_model.graph.input:
    input_shape = onnx_get_dimensions(input_tensor)
    inputs_shape.append(input_shape)
  outputs_shape = []
  for output_tensor in onnx_model.graph.output:
    output_shape = onnx_get_dimensions(output_tensor)
    outputs_shape.append(output_shape)
  return inputs_shape, outputs_shape


def onnx_validate(inputs_shape, outputs_shape):
  if len(inputs_shape)!= 1:
    raise Exception("Only one input is supported")
  if len(outputs_shape)!= 1:
    raise Exception("Only one output is supported")
  if len(inputs_shape[0]["dims"]) != 2 or inputs_shape[0]["dims"][0] != 1:
    raise Exception("Input shape must be [1, N]")
  if len(outputs_shape[0]["dims"]) != 2 or outputs_shape[0]["dims"][0] != 1:
    raise Exception("Output shape must be [1, N]")
  if inputs_shape[0]["data_type"] != "FLOAT":
    raise Exception("Input must be FLOAT (float16)")
  if outputs_shape[0]["data_type"] != "FLOAT":
    raise Exception("Output must be FLOAT (float16)")


def onnx_test(onnx_model, the_input):
  session = ort.InferenceSession(onnx_model.SerializeToString())
  input_name = session.get_inputs()[0].name
  input_data = the_input.numpy()
  output = session.run(None, {input_name: input_data})
  return output[0][0]


def tiny_model_render(tiny_model, the_input):
  run, special_names = jit_model(tiny_model, the_input)
  functions, statements, bufs, bufs_to_save = compile_net(run, special_names)
  # Render to clang
  c_code = export_model_clang(functions, statements, bufs, {}, ["input0"], ["output0"]) # hard coded 1D arrays for input, output
  return c_code, bufs_to_save


def tiny_model_run(tiny_model, the_input):
  run, _ = jit_model(tiny_model, the_input)
  output = run(the_input)
  return output[0].numpy()[0]


def clang_compile(c_code, bufs_to_save, input_shapes, output_shapes, save_files):
  input_len = input_shapes[0]["dims"][1]
  output_len = output_shapes[0]["dims"][1]
  cprog = ["#include <string.h>", "#include <stdio.h>", "#include <stdlib.h>"]
  cprog.append(c_code)

  # weights
  cprog.append("void initialize(float *weights) {")
  weights = bytes()
  for name,cl in bufs_to_save.items():
    cprog.append(f"memcpy({name}, weights + {len(weights)//output_len}, {len(cl._buf)*output_len});")
    weights += bytes(cl._buf)
  cprog.append("}")

  # write the weights to disk
  with open("/tmp/clang_weights", "wb") as f:
    f.write(weights)

  output_print = ["printf(\""]
  for _ in range(output_len-1):
    output_print.append("%f ")
  output_print.append("%f\\n\", ")
  for i in range(output_len-1):
    output_print.append(f"outputs[{i}], ")
  output_print.append(f"outputs[{output_len-1}]);")

  # test program
  cprog.append(f"""int main(int argc, char *argv[]) {{
    // read in the weights from disk
    FILE *f = fopen("/tmp/clang_weights", "rb");
    float *weights = (float *)malloc({len(weights)});
    fread(weights, 1, {len(weights)}, f);
    fclose(f);

    // init the net
    initialize(weights);

    // test run
    float input[{input_len}];
    float outputs[{output_len}];
    for (int i = 0; i < {input_len}; i++) scanf("%f", &input[i]);
    net(input, outputs);
    {"".join(output_print)}
  }}""")

  # ready the program
  prg = '\n'.join(cprog)

  if save_files:
    with open("clang_model.c", "w") as f:
      f.write(c_code)
    with open("clang_model_test.c", "w") as f:
        f.write(prg)

  # add test weights
  subprocess.check_output(['clang', '-O2', '-lm', '-fPIC', '-x', 'c', '-', '-o', "/tmp/clang_test"], input=prg.encode('utf-8'))


def clang_test(the_input):
  c_input = ' '.join(["%f" % x for x in the_input[0].numpy()])+"\n"
  c_output = [float(x) for x in subprocess.check_output(["/tmp/clang_test"], input=c_input.encode('utf-8')).decode('utf-8').strip().split(" ")]
  return c_output


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model", type=str, default="models/model.onnx", help="Path to onnx model file")
  parser.add_argument("-o", "--output", action="store_true", help="Output the generated C code")
  args = parser.parse_args()

  # load onnx model
  onnx_model = onnx.load(args.model)
  input_shapes, output_shapes = onnx_get_shapes(onnx_model)
  onnx_validate(input_shapes, output_shapes)

  # generate random input
  the_input = Tensor.randn(input_shapes[0]["dims"])

  # run onnx model
  onnx_output = onnx_test(onnx_model, the_input)
  print("onnx:   ", onnx_output, file=sys.stderr)

  # run tinygrad model
  tiny_model = TinyOnnx(onnx_model)
  tiny_output = tiny_model_run(tiny_model, the_input)
  print("tiny:   ", tiny_output, file=sys.stderr)
  np.testing.assert_allclose(onnx_output, tiny_output, atol=1e-5, rtol=1e-5)

  # render to clang
  c_code, bufs_to_save = tiny_model_render(tiny_model, the_input)

  # compile and test clang model
  clang_compile(c_code, bufs_to_save, input_shapes, output_shapes, args.output)
  clang_output = clang_test(the_input)
  print("clang:   ", clang_output, file=sys.stderr)
  np.testing.assert_allclose(onnx_output, clang_output, atol=1e-5, rtol=1e-5)
  np.testing.assert_allclose(tiny_output, clang_output, atol=1e-5, rtol=1e-5)

  