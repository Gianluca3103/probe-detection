"""Fixed-shape TensorRT runtime for the selected YOLO11-N probe detector.

The engine is expected to contain preprocessing, YOLO11-N, confidence
filtering, top-1 NMS, and coordinate restoration. Its output is a packed
``[count, detections...]`` tensor where every row has
``[x1, y1, x2, y2, confidence, class_id]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

#Stores BBOX parameters
@dataclass(frozen=True, slots=True)
class ProbeDetection:

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


class ProbeDetector:
    #Checks if CUDA is available
    def __init__(self, engine_path: str | Path, device: str = "cuda:0") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("ProbeDetector requires an NVIDIA CUDA device")

        import tensorrt as trt
        #Checks if the tensorRT input file has been created
        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
        #Registers tensorRT plugin globally
        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), "")
        #Converts device into a torch object
        self._device = torch.device(device)
        #Creates a logger
        logger = trt.Logger(trt.Logger.ERROR)
        # Creates tensorRT runtime object
        self._runtime = trt.Runtime(logger)

        #Check if TensorRT can parse through input
        self._engine = self._runtime.deserialize_cuda_engine(
            self.engine_path.read_bytes()
        )
        if self._engine is None:
            raise RuntimeError(f"Could not deserialize engine: {self.engine_path}")

        #Initializes useful objects
        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream(device=self._device)
        self._buffers: dict[str, torch.Tensor] = {}
        input_names: list[str] = []
        output_names: list[str] = []
        dtype_map = {
            trt.DataType.UINT8: torch.uint8,
            trt.DataType.HALF: torch.float16,
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool,
        }
        #Goes through every tensor and allocates GPU memory
        for index in range(self._engine.num_io_tensors):
            #Stores name of the string in the position number of the tensor
            name = self._engine.get_tensor_name(index)
            #Given tensor name, store tensor shape
            shape = tuple(int(value) for value in self._engine.get_tensor_shape(name))
            #Extract type of variable
            dtype = dtype_map[self._engine.get_tensor_dtype(name)]
            #Given shape and type allocate the appropriate amount of memory
            tensor = torch.empty(shape, dtype=dtype, device=self._device)
            #Keeps reference on the tensor
            self._buffers[name] = tensor
            #Tells TensorRT which memory address to use for input/output
            self._context.set_tensor_address(name, tensor.data_ptr())
            #Classify if tensor is input/output
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)

        #Does some validation on the inputs and outputs for proper execution
        if len(input_names) != 1 or len(output_names) != 1:
            raise ValueError("Deployment engine must have exactly one input and output")
        self._input_name = input_names[0]
        self._output_name = output_names[0]
        input_tensor = self._buffers[self._input_name]
        output_tensor = self._buffers[self._output_name]
        if input_tensor.dtype != torch.uint8 or input_tensor.ndim != 4:
            raise ValueError("Expected a uint8 NHWC engine input")
        if input_tensor.shape[0] != 1 or input_tensor.shape[-1] != 3:
            raise ValueError("Expected engine input shape [1, height, width, 3]")
        if output_tensor.ndim != 3 or output_tensor.shape[-1] != 6:
            raise ValueError("Expected packed engine output shape [1, rows, 6]")
        if output_tensor.shape[1] != 2:
            raise ValueError("Expected a top-1 engine with one header and one detection")

        #Allocates the memory for the tensors in the RAM
        self._input_host = torch.empty_like(input_tensor, device="cpu", pin_memory=True)
        self._output_host = torch.empty_like(output_tensor, device="cpu", pin_memory=True)
        #Initializes CUDA Graph
        self._graph: torch.cuda.CUDAGraph | None = None

    #this just removes the batch size of the tensor
    #used as a reference of the frame size for future validations
    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self._input_host.shape[1:])

    #converting tensors from pytorch to numpy
    @property
    def input_buffer(self) -> np.ndarray:
        return self._input_host[0].numpy()
    #Setting up the warmup stage
    def prepare(self) -> None:
        #If CUDA Graph interface exist pretend this doesn't exist
        if self._graph is not None:
            return
        #Running the warmup stage
        for _ in range(3):
            self._enqueue()
        #Initialize Graph object
        graph = torch.cuda.CUDAGraph()
        #Enable capture mode for cuda graph
        with torch.cuda.graph(graph, stream=self._stream):
            #Recorded step 1: copy from the pinned CPU buffer into the GPU input tensor
            self._buffers[self._input_name].copy_(self._input_host, non_blocking=True)
            #Recorded step 2: run the TensorRT Performs the TensorRT check once during capture
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed during CUDA Graph capture")
            #Recorded step 3: copy the result from the GPU output back into the CPU output buffer
            self._output_host.copy_(
                self._buffers[self._output_name], non_blocking=True
            )
        #Stores graph capture
        self._graph = graph

    #Runs the Inference
    def detect_buffer(self) -> ProbeDetection | None:
        #Ensures warmup + graph capture has been prepared
        self.prepare()

        assert self._graph is not None
        #Reexecutes the graph captured
        with torch.cuda.stream(self._stream):
            self._graph.replay()
        # Blocks CPU execution until the entire graph operation is finished
        self._stream.synchronize()


        count = int(self._output_host[0, 0, 0])
        if count == 0: #No detection
            return None
        if count != 1: #Impossible
            raise RuntimeError(f"Top-1 engine returned invalid detection count: {count}")
        #so this stores the parameters given by the output when count==1
        x1, y1, x2, y2, confidence, class_id = (
            float(value) for value in self._output_host[0, 1]
        )
        #Validation checks
        if round(class_id) != 0:
            raise RuntimeError(f"Unexpected class id from single-class engine: {class_id}")
        if not (0.0 <= x1 < x2 <= self.frame_shape[1]):
            raise RuntimeError(f"Invalid output x coordinates: {(x1, x2)}")
        if not (0.0 <= y1 < y2 <= self.frame_shape[0]):
            raise RuntimeError(f"Invalid output y coordinates: {(y1, y2)}")
        if not 0.0 <= confidence <= 1.0:
            raise RuntimeError(f"Invalid output confidence: {confidence}")
        return ProbeDetection(x1, y1, x2, y2, confidence)
    #Runs inference to prime the GPU
    def _enqueue(self) -> None:
        #line temporarily directs the block's GPU operations onto a dedicated stream
        with torch.cuda.stream(self._stream), torch.inference_mode():
            #Move the data into the appropriate buffers
            self._buffers[self._input_name].copy_(self._input_host, non_blocking=True)
            #Sanity check
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            #Doing same but output
            self._output_host.copy_(
                self._buffers[self._output_name], non_blocking=True
            )
        self._stream.synchronize()
