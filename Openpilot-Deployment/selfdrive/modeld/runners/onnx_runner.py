#!/usr/bin/env python3

import os
from re import A
import sys
import select
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import onnxruntime as ort # pylint: disable=import-error

pipein =  -1
pipeout = -1

def read(sz):
  dd = []
  gt = 0
  global pipein 
  read_pollers = select.poll()
  read_pollers.register(pipein, select.POLLIN)
  while gt < sz * 4:
    err = read_pollers.poll(1000)
    assert(len(err) > 0)
    st = os.read(pipein, sz * 4 - gt)
    assert(len(st) > 0)
    dd.append(st)
    gt += len(st)
  return np.frombuffer(b''.join(dd), dtype=np.float32)

def write(d):
  global pipeout 
  os.write(pipeout, d.tobytes())

def run_loop(m):
  ishapes = [[1]+ii.shape[1:] for ii in m.get_inputs()]

  keys = [x.name for x in m.get_inputs()]
  print("ready to run onnx model", keys, ishapes, file=sys.stderr)
  while 1:
    inputs = []
    for shp in ishapes:
      ts = np.product(shp)
      #print("reshaping %s with offset %d" % (str(shp), offset), file=sys.stderr)
      inputs.append(read(ts).reshape(shp))
    ret = m.run(None, dict(zip(keys, inputs)))
    #print(ret, file=sys.stderr)
    for r in ret:
      write(r)


if __name__ == "__main__":
  # print("parameter : ", sys.argv[2],sys.argv[3])
  pipein = int(sys.argv[2])
  pipeout = int(sys.argv[3])

  print("Onnx available providers: ", ort.get_available_providers(), file=sys.stderr)
  options = ort.SessionOptions()
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  if 'OpenVINOExecutionProvider' in ort.get_available_providers() and 'ONNXCPU' not in os.environ:
    provider = 'OpenVINOExecutionProvider'
  elif 'CUDAExecutionProvider' in ort.get_available_providers() and 'ONNXCPU' not in os.environ:
    options.intra_op_num_threads = 2
    provider = 'CUDAExecutionProvider'
  else:
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 8
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    provider = 'CPUExecutionProvider'

  print("Onnx selected provider: ", [provider], file=sys.stderr)
  ort_session = ort.InferenceSession(sys.argv[1], options, providers=[provider])
  print("Onnx using ", ort_session.get_providers(), file=sys.stderr)
  run_loop(ort_session)
