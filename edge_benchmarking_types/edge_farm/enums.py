from enum import Enum


class InferenceClientType(str, Enum):
    TRITON_DENSENET = "TritonDenseNetClient"
    TRITON_YOLO = "TritonYoloClient"
