from typing import Optional, List
from pydantic import BaseModel, ConfigDict
from edge_benchmarking_types.edge_device.enums import InferenceServerType


class BenchmarkData(BaseModel):
    bucket_name: str
    dataset: List[str]
    labels: str
    model: str
    model_metadata: str
    model_repository: str

    model_config = ConfigDict(protected_namespaces=())


class EdgeDeviceConfig(BaseModel):
    protocol: str = "http"
    host: str


class InferenceClientConfig(BaseModel):
    protocol: str = "http"
    host: str
    model_name: Optional[str]
    model_version: str = "1"
    num_classes: int = 0
    batch_size: int = 1
    scaling: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())


class BenchmarkConfig(BaseModel):
    inference_server_type: InferenceServerType
    edge_device: EdgeDeviceConfig
    inference_client: InferenceClientConfig
