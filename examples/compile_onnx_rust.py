# An example to compile a small Tensorflow model to extremely portable C code

import os, sys
#os.environ["CLANG"] = '1'

import numpy as np
import subprocess
from extra.onnx import get_run_onnx
from tinygrad.tensor import Tensor
#from extra.export_model import export_model_clang, compile_net, jit_model
from extra.export_model import export_model_rust, compile_net, jit_model
import onnx
import onnxruntime as ort


class TinyOnnx:
  def __init__(self, onnx_model):
    self.xname = onnx_model.graph.input[0].name
    self.yname = onnx_model.graph.output[0].name
    self.run_onnx = get_run_onnx(onnx_model)
    print(f"run_onnx: {self.run_onnx}")

  def forward(self, x):
    print(f"x = {x}")
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
    raise Exception("Input must be FLOAT (float32)")
  if outputs_shape[0]["data_type"] != "FLOAT":
    raise Exception("Output must be FLOAT (float32)")


def onnx_test(onnx_model, np_input):
  s = onnx_model.SerializeToString()
  session = ort.InferenceSession(s)
  input_name = session.get_inputs()[0].name
  output = session.run(None, {input_name: np_input})
  return output[0][0]


def tiny_model_render(tiny_model, the_input):
  print("tiny_model_render jit_model")
  run, special_names = jit_model(tiny_model, the_input)
  print("tiny_model_render compile_net")
  functions, statements, bufs, bufs_to_save = compile_net(run, special_names)
  # Render to clang
  print("tiny_model_render export_model_rust")
  rs_code = export_model_rust(functions, statements, bufs, {}, ["input0"], ["output0"]) # hard coded 1D arrays for input, output
  print("tiny_model_render done")
  return rs_code, bufs_to_save


def tiny_model_run(tiny_model, the_input):
  run, _ = jit_model(tiny_model, the_input)
  output = run(the_input)
  return output[0].numpy()[0]


def rust_compile(rs_code, bufs_to_save, input_shapes, output_shapes, save_files):
  input_len = input_shapes[0]["dims"][1]
  output_len = output_shapes[0]["dims"][1]
  rsprog = ["use std::fs::File;","use std::io::{self, Read};",""]
  rsprog += [rs_code,""]

  # weights
  rsprog += ["impl Net {","  pub fn initialize(&mut self, weights: &[f32]) {"]
  weights = bytes()
  for name,cl in bufs_to_save.items():
    rsprog.append(f"    self.{name}.copy_from_slice(&weights[{len(weights)//4}..{len(weights)//4+cl.size*output_len}]);")
    weights += bytes(cl._buf)
  rsprog += ["  }","}",""]

  # write the weights to disk
  with open("/tmp/rust_weights", "wb") as f:
    f.write(weights)

  # test program
  rsprog += ["fn main() -> io::Result<()> {","  // Initialize network","  let mut net = Net::new();","",""]
  rsprog += ["  // Read weights from a file","  let mut f = File::open(\"/tmp/rust_weights\")?;","  let mut weights_bytes = Vec::new();","  f.read_to_end(&mut weights_bytes)?;",""]
  rsprog += ["  // Convert bytes to f32","  let mut weights = [0.0f32; 16384];","  // Now map the weights_bytes into weights"]
  rsprog += ["  for i in 0..(weights_bytes.len()/4) {","    weights[i] = f32::from_le_bytes([weights_bytes[i*4], weights_bytes[i*4+1], weights_bytes[i*4+2], weights_bytes[i*4+3]]);","  }",""]
  rsprog += ["  // Initialize the network with weights","  net.initialize(&weights);",""]
  rsprog += ["  // Get inputs","  for i in 0..input.len() {","    io::stdin().read_line(&mut line).unwrap();","    input[i] = line.trim().parse::<f32>().unwrap();","  }",""]
  rsprog += ["  // Run the network","  net.run(&input, &mut output);",""]
  rsprog += ["  // Print the output","  for i in 0..output.len() {","    print!(\"{}\", output[i]);","  }"]
  rsprog += ["  Ok(())","}"]

  # ready the program
  print(rsprog)
  prg = '\n'.join(rsprog)

  if save_files:
    with open("rust_model.rs", "w") as f:
      f.write(rs_code)
    with open("rust_model_test.rs", "w") as f:
        f.write(prg)

  # Compile the source
  #      rustc -O --crate-type=cdylib - -o 
  subprocess.check_output(['rustc', '-O', '-', '-o', "/tmp/rust_test"], input=prg.encode('utf-8'))


def rust_test(the_input):
  c_input = ' '.join(["%f" % x for x in the_input[0].numpy()])+"\n"
  c_output = [float(x) for x in subprocess.check_output(["/tmp/rust_test"], input=c_input.encode('utf-8')).decode('utf-8').strip().split(" ")]
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

  # generate random input for onnx (np) and for tinygrad (Tensor)
  np.random.seed(123)
  np_input = np.random.randn(*input_shapes[0]["dims"]).astype(np.float32)
  the_input = Tensor(np_input)
  #the_input = Tensor.randn(input_shapes[0]["dims"])

  # run onnx model
  onnx_output = onnx_test(onnx_model, np_input)
  print("onnx:   ", onnx_output, file=sys.stderr)

  # run tinygrad model
  tiny_model = TinyOnnx(onnx_model)
  # tiny_output = tiny_model_run(tiny_model, the_input)
  # print("tiny:   ", tiny_output, file=sys.stderr)
  # np.testing.assert_allclose(onnx_output, tiny_output, atol=1e-5, rtol=1e-5)

  # render to clang
  rust_code, bufs_to_save = tiny_model_render(tiny_model, the_input)
  #print("rust_code: ", rust_code, file=sys.stderr)

  # compile and test clang model
  rust_compile(rust_code, bufs_to_save, input_shapes, output_shapes, args.output)
  rust_output = rust_test(the_input)
  print("rust:   ", rust_output, file=sys.stderr)
  np.testing.assert_allclose(onnx_output, rust_output, atol=1e-5, rtol=1e-5)
  np.testing.assert_allclose(tiny_output, rust_output, atol=1e-5, rtol=1e-5)
