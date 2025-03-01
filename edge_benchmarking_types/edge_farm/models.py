from typing import Optional, List, Union, Any, Dict
from pydantic import BaseModel, Field, ConfigDict, field_validator


class Latency(BaseModel):
    average: float
    percentiles: Dict[int, float]


class PerformanceResult(BaseModel):
    total_time: float
    sample_count: int
    samples_per_second: float
    latency: Latency


class InferPerformance(BaseModel):
    preprocess: PerformanceResult
    inference: PerformanceResult
    postprocess: PerformanceResult


class BenchmarkInferResult(BaseModel):
    performance: InferPerformance
    results: Dict[str, Any]


class DatasetSample(BaseModel):
    filename: str
    content_type: Optional[str] = Field(default=None)
    data: Any


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
    num_workers: int = Field(default=1)
    samples_per_second: Optional[float] = Field(default=None)

    @field_validator("num_workers")
    @classmethod
    def check_num_workers_nonzero(cls, v: int) -> int:
        if v == 0:
            raise ValueError("Field num_workers cannot be zero.")
        return v

    @field_validator("samples_per_second")
    @classmethod
    def check_samples_per_second_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("Field samples_per_second has to be positive.")
        return v


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
    cpu_only: bool = Field(default=False)
