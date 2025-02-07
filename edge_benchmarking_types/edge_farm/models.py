from typing import Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict


class BenchmarkModel(BaseModel):
    name: str
    repository: str
    metadata: Optional[str] = Field(default=None)
    labels: Optional[str] = Field(default=None)


class BenchmarkData(BaseModel):
    bucket_name: str
    model: BenchmarkModel
    dataset: List[str]


class EdgeDevice(BaseModel):
    protocol: str = Field(default="http")
    host: str
    port: Optional[int] = Field(default=None)


class InferenceClient(BaseModel):
    protocol: str = Field(default="http")
    host: str


class TritonInferenceClient(InferenceClient):
    model_name: Optional[str] = Field(default=None)
    model_version: str = Field(default="1")
    batch_size: int = Field(default=1)

    model_config = ConfigDict(protected_namespaces=())


class TritonDenseNetClient(TritonInferenceClient):
    num_classes: int = Field(default=0)
    scaling: Optional[str] = Field(default=None)


class TritonYoloClient(TritonInferenceClient):
    num_classes: int = Field(default=0)
    scaling: Optional[str] = Field(default=None)
    confidence_thres: float = Field(default=0.2, ge=0, le=1)
    iou_thres: float = Field(default=0.2, ge=0, le=1)
    input_width: int
    input_height: int


class BenchmarkConfig(BaseModel):
    edge_device: EdgeDevice
    inference_client: Union[TritonDenseNetClient, TritonYoloClient]
