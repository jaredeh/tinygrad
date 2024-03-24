# An example to compile an ONNX model into code and test it

import os, sys
import numpy as np
import subprocess
from extra.onnx import get_run_onnx
from tinygrad import dtypes
from tinygrad.tensor import Tensor
from tinygrad.helpers import DEBUG
from tinygrad.device import Device
from extra.export_model import export_model, compile_net, jit_model
import onnx
import onnxruntime as ort
from onnx.helper import tensor_dtype_to_np_dtype


def onnx_get_dimensions(onnx_tensor):
  tensor_data_type = onnx_tensor.type.tensor_type.elem_type
  data_type_str = tensor_dtype_to_np_dtype(tensor_data_type)
  shape = {"dims": [], "dtype": data_type_str}
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
  if len(inputs_shape)!= 1:
    raise Exception("Only one input is supported")
  if len(outputs_shape)!= 1:
    raise Exception("Only one output is supported")
  if len(inputs_shape[0]["dims"]) != 2 or inputs_shape[0]["dims"][0] != 1:
    raise Exception("Input shape assumed to be [1, N]")
  if len(outputs_shape[0]["dims"]) != 2 or outputs_shape[0]["dims"][0] != 1:
    raise Exception("Output shape assumed to be [1, N]")
  return inputs_shape, outputs_shape


def onnx_test(onnx_model, np_input):
  s = onnx_model.SerializeToString()
  session = ort.InferenceSession(s)
  input_name = session.get_inputs()[0].name
  output = session.run(None, {input_name: np_input})
  return output[0][0]


def onnx_load_model(model_path):
  if model_path is None:
    print("Create dummy onnx model")
    import torch
    class DummyModel(torch.nn.Module):
      def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 4)
      def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    torch_model = DummyModel()
    dummy_input = torch.randn(1, 10)
    model_path = "/tmp/dummy_model.onnx"
    torch.onnx.export(torch_model, dummy_input, model_path)
  elif not os.path.exists(model_path):
    raise Exception(f"Model file {model_path} not found")
  return onnx.load(model_path)


# converts ONNX model to Tinygrad compatible model
class TinyOnnx:
  def __init__(self, onnx_model):
    self.xname = onnx_model.graph.input[0].name
    self.yname = onnx_model.graph.output[0].name
    self.run_onnx = get_run_onnx(onnx_model)

  def forward(self, x):
    return self.run_onnx({self.xname: x}, debug=False)[self.yname]


def tiny_model_run(tiny_model, the_input):
  run, _ = jit_model(tiny_model, the_input)
  output = run(the_input)
  return output[0].numpy()[0]


def clang_compile(c_code, bufs, bufs_to_save, inputs, outputs, save_files):
  dtype_map = {dtypes.float: ("float",4)}
  input_name = list(inputs.keys())[0]
  output_name = list(outputs.keys())[0]
  input_type = dtype_map[bufs[input_name][1]]
  output_type = dtype_map[bufs[output_name][1]]
  input_len = int(inputs[input_name]//input_type[1])
  output_len = int(outputs[output_name]//output_type[1])
  weights_type = input_type

  cprog = ["#include <string.h>", "#include <stdio.h>", "#include <stdlib.h>"]
  cprog += [c_code, ""]

  # weights
  cprog += [f"void initialize({weights_type[0]} *weights) {{"]
  weights = bytes()
  for name,cl in bufs_to_save.items():
    cprog.append(f"  memcpy({name}, weights + {len(weights)//weights_type[1]}, {len(cl._buf)});")
    weights += bytes(cl._buf)
  cprog += ["}", ""]

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
  output_print = ''.join(output_print)

  # test program
  cprog += [f"int main(int argc, char *argv[]) {{",
    "  // read in the weights from disk",
    "  FILE *f = fopen(\"/tmp/clang_weights\", \"rb\");",
    f"  {weights_type[0]} *weights = ({weights_type[0]} *)malloc({len(weights)});",
    f"  fread(weights, 1, {len(weights)}, f);",
    "  fclose(f);", "","  // init the net","  initialize(weights);",""
    "  // test run",f"  {input_type[0]} input[{input_len}];"
    f"  {output_type[0]} outputs[{output_len}];",
    f"  for (int i = 0; i < {input_len}; i++) scanf(\"%f\", &input[i]);"
    f"  net(input, outputs);", ""
    f"  {output_print}", "}"]

  # ready the program
  prg = '\n'.join(cprog)

  if save_files:
    with open("clang_model_test.c", "w") as f:
        f.write(prg)

  # add test weights
  subprocess.check_output(['clang', '-O2', '-lm', '-fPIC', '-x', 'c', '-', '-o', "/tmp/compile_onnx_test"], input=prg.encode('utf-8'))


def rust_compile(rs_code, bufs, bufs_to_save, inputs, outputs, save_files):
  dtype_map = {dtypes.float: ("f32",4)}
  input_name = list(inputs.keys())[0]
  output_name = list(outputs.keys())[0]
  input_type = dtype_map[bufs[input_name][1]]
  output_type = dtype_map[bufs[output_name][1]]
  input_len = int(inputs[input_name]//input_type[1])
  output_len = int(outputs[output_name]//output_type[1])
  weights_type = input_type

  rsprog = ["use std::fs::File;","use std::io::{self, Read};",""]
  rsprog += [rs_code,""]

  # code to initialize weights
  rsprog += ["impl Net {",f"  pub fn initialize(&mut self, weights: &[{input_type[0]}]) {{"]
  weights = bytes()
  for name,cl in bufs_to_save.items():
    rsprog.append(f"    self.{name}.copy_from_slice(&weights[{len(weights)//weights_type[1]}..{len(weights)//weights_type[1]+cl.size}]);")
    weights += bytes(cl._buf)
  rsprog += ["  }","}",""]

  # write the weights to disk
  with open("/tmp/rust_weights", "wb") as f:
    f.write(weights)

  # test program
  rsprog += ["fn main() -> io::Result<()> {","  // Initialize network","  let mut net = Net::new();","","",
    f"  // Create an input buffer of {input_len} {input_type[0]}s",f"  let mut input = [0.0; {input_len}];",f"  let mut output = [0.0; {output_len}];","  let mut line = String::new();","",
    "  // Read weights from a file","  let mut f = File::open(\"/tmp/rust_weights\")?;","  let mut weights_bytes = Vec::new();","  f.read_to_end(&mut weights_bytes)?;","",
    f"  // Convert bytes to {weights_type[0]}",f"  let mut weights: Vec<{weights_type[0]}> = Vec::with_capacity(weights_bytes.len() / {weights_type[1]});","  // Now map the weights_bytes into weights",
    f"  for i in 0..(weights_bytes.len()/{weights_type[1]}) {{",f"    weights.push({weights_type[0]}::from_le_bytes([{','.join(['weights_bytes[i*4+'+str(i)+']' for i in range(weights_type[1])])}]));","  }","",
    "  // Initialize the network with weights","  net.initialize(&weights);","",
    "  // Get inputs","  for i in 0..input.len() {","    io::stdin().read_line(&mut line).unwrap();","    input[i] = line.trim().parse::<f32>().unwrap();","    line.clear();","  }","",
    "  // Run the network","  net.run(&input, &mut output);","",
    "  // Print the output","  let outputstr = output.iter().map(|item| item.to_string()).collect::<Vec<_>>().join(\" \");","  print!(\"{}\", outputstr);","",
    "  Ok(())","}"]

  # ready the program
  prg = '\n'.join(rsprog)

  if save_files:
    with open("rust_model_test.rs", "w") as f:
        f.write(prg)

  # Compile the source
  rustc_cmd = ['rustc']
  if int(os.environ.get('DEBUG',0)) < 2:
    rustc_cmd += ['-Aunused_parens','-Aunused_mut']
  rustc_cmd += ['-O', '-', '-o', "/tmp/compile_onnx_test"]
  subprocess.check_output(rustc_cmd, input=prg.encode('utf-8'))


def compile_src(tiny_model, device, the_input, save_files, save_weights):
  src_code, inputs, outputs, _ = export_model(tiny_model, device, the_input, save_weights=save_weights)
  run, special_names = jit_model(tiny_model, the_input)
  _, _, bufs, bufs_to_save = compile_net(run, special_names)
  if device == "rust":
    rust_compile(src_code, bufs, bufs_to_save, inputs, outputs, save_files)
  elif device == "clang":
    clang_compile(src_code, bufs, bufs_to_save, inputs, outputs, save_files)
  else:
    raise Exception(f"Unknown device {device}")


def compiled_test(the_input):
  c_input = '\n'.join(["%f" % x for x in the_input[0].numpy()])+"\n"
  c_output = [float(x) for x in subprocess.check_output(["/tmp/compile_onnx_test"], input=c_input.encode('utf-8')).decode('utf-8').strip().split(" ")]
  return c_output


if __name__ == "__main__":
  import argparse
  SUPPORTED_LANGUAGES = ["rust", "clang"]

  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model", type=str, help="Path to onnx model file")
  parser.add_argument("--notest", action="store_true", help="Don't test the generated code")
  parser.add_argument("-s","--save", action="store_true", help="Save the generated src code")
  parser.add_argument("-w", "--weights", action="store_true", help="Encode weights in the generated src code")
  parser.add_argument("-l", "--language", type=str, default="clang", help=f"Device to compile for, one of {SUPPORTED_LANGUAGES}")
  args = parser.parse_args()

  # Set up the device/language settings
  if args.language not in SUPPORTED_LANGUAGES:
    raise Exception(f"Example only supports '{SUPPORTED_LANGUAGES}' not {args.language}")
  if not args.save and args.notest:
    raise Exception("Can't do --notest without --save, nothing to do")
  os.environ[args.language.upper()] = "1"
  Device.DEFAULT = args.language.upper()
  print(f"Compiling for {args.language}", file=sys.stderr)

  # load models
  onnx_model = onnx_load_model(args.model)
  tiny_model = TinyOnnx(onnx_model)

  # generate random input for onnx (np) and for tinygrad (Tensor)
  input_shapes, output_shapes = onnx_get_shapes(onnx_model)
  np.random.seed(123)
  np_input = np.random.randn(*input_shapes[0]["dims"]).astype(input_shapes[0]["dtype"])
  the_input = Tensor(np_input)

  if not args.notest:
    # run onnx model as the control
    onnx_output = onnx_test(onnx_model, np_input)
    print(f"onnx:     {onnx_output}", file=sys.stderr)

    # run tinygrad model
    tiny_output = tiny_model_run(tiny_model, the_input)
    print(f"tiny:     {tiny_output}", file=sys.stderr)
    np.testing.assert_allclose(onnx_output, tiny_output, atol=1e-5, rtol=1e-5)

    # compile and run the generated code
    compile_src(tiny_model, args.language, the_input, args.save, args.weights)
    compiled_output = compiled_test(the_input)
    print(f"compiled: {compiled_output}", file=sys.stderr)
    np.testing.assert_allclose(onnx_output, compiled_output, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(tiny_output, compiled_output, atol=1e-5, rtol=1e-5)
    print("Passed all tests")

  if args.save:
    print("Saving src code", file=sys.stderr)
    src_code = export_model(tiny_model, args.language, the_input, save_weights=args.weights)
    srcfilename = "model.rs" if args.language == "rust" else "model.c"
    with open(srcfilename, "w") as f:
      f.write(src_code[0])
    if DEBUG > 1: print("src_code: ", src_code, file=sys.stderr)
