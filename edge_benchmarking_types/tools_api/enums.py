from enum import Enum


class OnnxModelInputType(str, Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    UINT8 = "uint8"
    STRING = "string"
    BOOL = "bool"


class OnnxExportMode(str, Enum):
    EVAL = "EVAL"
    PRESERVE = "PRESERVE"
    TRAINING = "TRAINING"


class OnnxOperatorExportType(str, Enum):
    ONNX = "ONNX"
    ONNX_FALLTHROUGH = "ONNX_FALLTHROUGH"
    ONNX_ATEN = "ONNX_ATEN"
    ONNX_ATEN_FALLBACK = "ONNX_ATEN_FALLBACK"


class ModelFormat(str, Enum):
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
