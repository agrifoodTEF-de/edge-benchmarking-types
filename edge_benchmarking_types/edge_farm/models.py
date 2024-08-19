from typing import Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict


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


class InferenceClientConfig(BaseModel):
    protocol: str = Field(default="http")
    host: str


class TritonInferenceClientConfig(InferenceClientConfig):
    model_name: Optional[str] = Field(default=None)
    model_version: str = Field(default="1")
    num_classes: int = Field(default=0)
    batch_size: int = Field(default=1)
    scaling: Optional[str] = Field(default=None)

    model_config = ConfigDict(protected_namespaces=())


class BenchmarkConfig(BaseModel):
    edge_device: EdgeDeviceConfig
    inference_client: Union[TritonInferenceClientConfig]
