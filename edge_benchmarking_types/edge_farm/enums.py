from enum import Enum


class ModelFormat(str, Enum):
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
