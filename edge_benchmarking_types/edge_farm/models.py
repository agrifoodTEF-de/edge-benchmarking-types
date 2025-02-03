from typing import Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict
from edge_benchmarking_types.edge_farm.enums import InferenceClientType


class BenchmarkData(BaseModel):
    bucket_name: str
    dataset: List[str]
    labels: Optional[str] = Field(default=None)
    model: str
    model_metadata: Optional[str] = Field(default=None)
    model_repository: str

    model_config = ConfigDict(protected_namespaces=())


class EdgeDeviceConfig(BaseModel):
    protocol: str = Field(default="http")
    host: str
    port: Optional[int] = Field(default=None)


class InferenceClientConfig(BaseModel):
    protocol: str = Field(default="http")
    host: str


class TritonInferenceClientConfig(InferenceClientConfig):
    model_name: Optional[str] = Field(default=None)
    model_version: str = Field(default="1")
    batch_size: int = Field(default=1)

    model_config = ConfigDict(protected_namespaces=())


class TritonDenseNetClientConfig(TritonInferenceClientConfig):
    num_classes: int = Field(default=0)
    scaling: Optional[str] = Field(default=None)


class TritonYoloClientConfig(TritonInferenceClientConfig):
    num_classes: int = Field(default=0)
    scaling: Optional[str] = Field(default=None)
    confidence_thres: float = Field(default=0.2, ge=0, le=1)
    iou_thres: float = Field(default=0.2, ge=0, le=1)
    input_width: int
    input_height: int


class BenchmarkConfig(BaseModel):
    edge_device: EdgeDeviceConfig
    inference_client: Union[TritonInferenceClientConfig]
    inference_client_type: InferenceClientType
