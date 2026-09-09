from datetime import datetime
from typing import Optional, List, Literal, Union, Any, Dict
from edge_benchmarking_types.patterns import HOSTNAME_REGEX
from edge_benchmarking_types.edge_farm.enums import (
    OptimizationFactor,
    LatencyPercentile,
)
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    model_validator,
)


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
    # Time spent fetching/loading each batch's samples from the data provider
    # (e.g. a SeaweedFS S3 GET) before preprocessing. Optional/additive so old
    # result JSON and un-bumped consumers keep deserializing.
    load: Optional[PerformanceResult] = Field(default=None)
    warmup: Optional[float] = Field(default=None)
    # Wall-clock boundaries (manager clock — the inference client runs on the
    # Edge-Farm manager, not the device). Optional/additive so old result JSON
    # and un-bumped consumers keep deserializing.
    run_started_at: Optional[datetime] = Field(default=None)
    run_finished_at: Optional[datetime] = Field(default=None)
    first_inference_started_at: Optional[datetime] = Field(default=None)
    first_inference_finished_at: Optional[datetime] = Field(default=None)
    warmup_started_at: Optional[datetime] = Field(default=None)
    warmup_finished_at: Optional[datetime] = Field(default=None)


class BenchmarkInferResult(BaseModel):
    performance: InferPerformance
    results: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = Field(default=None)


class DatasetSample(BaseModel):
    filename: str
    content_type: Optional[str] = Field(default=None)
    data: bytes


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
    host: str = Field(..., pattern=HOSTNAME_REGEX, max_length=253)
    port: Optional[int] = Field(default=None, ge=1, le=65535)


class S3DataProvider(BaseModel):
    bucket_name: str
    prefix: Optional[str] = Field(default=None)


class LocalDataProvider(BaseModel):
    path: str


class InferenceClient(BaseModel):
    protocol: str = Field(default="http")
    host: str
    port: Optional[int] = Field(default=None, ge=1, le=65535)
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
    warm_up: bool = Field(default=False)
    model_config = ConfigDict(protected_namespaces=())


class TritonDenseNetClient(TritonInferenceClient):
    client_type: Literal["TritonDenseNetClient"] = "TritonDenseNetClient"
    num_classes: int = Field(default=0)
    scaling: Optional[str] = Field(default=None)


class TritonYoloClient(TritonInferenceClient):
    client_type: Literal["TritonYoloClient"] = "TritonYoloClient"
    num_classes: int = Field(default=0)
    scaling: Optional[str] = Field(default=None)
    confidence_thres: float = Field(default=0.2, ge=0, le=1)
    iou_thres: float = Field(default=0.2, ge=0, le=1)
    input_width: int
    input_height: int


class TritonBirdNetClient(TritonInferenceClient):
    """BirdNET (v3.0) acoustic classifier.

    The model takes raw float32 waveform of shape ``(batch, sample_rate *
    segment_seconds)`` -- the mel-spectrogram is baked into the ONNX graph -- and
    returns per-species logits plus embeddings. All waveform preparation
    (decode, mono-mix, resample, segment, pad) happens client-side; the defaults
    below reproduce the reference ``birdnet`` pipeline for v3.0.

    Note there is no ``num_classes``: BirdNET's top-k is applied client-side
    after the flat sigmoid, not by Triton's ``class_count``.
    """

    client_type: Literal["TritonBirdNetClient"] = "TritonBirdNetClient"
    top_k: int = Field(default=5, ge=1)
    confidence_thres: float = Field(default=0.1, ge=0, le=1)
    sample_rate: int = Field(default=32_000, gt=0)
    segment_seconds: float = Field(default=3.0, gt=0)
    overlap_seconds: float = Field(default=0.0, ge=0)
    apply_sigmoid: bool = Field(default=True)
    sigmoid_sensitivity: float = Field(default=1.0)
    bandpass_fmin: Optional[int] = Field(default=None, ge=0)
    bandpass_fmax: Optional[int] = Field(default=None, ge=0)
    # Requested upper bound on rows in a single Triton request. One audio file
    # expands to many segments, so an unbounded batch can reach hundreds of MB:
    # 1000 segments x 96000 samples x 4 bytes is ~384 MB.
    #
    # This is an upper bound, not the effective value: the client clamps it to
    # the model's max_batch_size at run time, because Triton *rejects* an
    # oversized request rather than splitting it. Edge devices start Triton with
    # --backend-config=default-max-batch-size=32, so an auto-completed model
    # caps out at 32 segments (96s of audio) unless a config.pbtxt raises it.
    max_segments_per_request: int = Field(default=256, ge=1)

    @model_validator(mode="after")
    def check_overlap_below_segment(self) -> "TritonBirdNetClient":
        if self.overlap_seconds >= self.segment_seconds:
            raise ValueError(
                "Field overlap_seconds has to be smaller than segment_seconds."
            )
        return self

    @model_validator(mode="after")
    def check_bandpass_range(self) -> "TritonBirdNetClient":
        fmin, fmax = self.bandpass_fmin, self.bandpass_fmax
        if fmin is not None and fmax is not None and fmin >= fmax:
            raise ValueError(
                "Field bandpass_fmin has to be smaller than bandpass_fmax."
            )
        return self


# Tag -> model, used by the back-compat validator below. Order matters only for
# readability; dispatch is by the discriminating field, not by position.
AnyTritonInferenceClient = Union[
    TritonDenseNetClient,
    TritonYoloClient,
    TritonBirdNetClient,
]


class BenchmarkConfig(BaseModel):
    edge_device: EdgeDevice
    inference_client: AnyTritonInferenceClient = Field(discriminator="client_type")
    cpu_only: bool = Field(default=False)

    @model_validator(mode="before")
    @classmethod
    def infer_client_type(cls, data: Any) -> Any:
        """Accept payloads from callers that predate ``client_type``.

        Without a tag, a discriminated union rejects the payload outright
        (``union_tag_not_found``). Producers pinned to an older version of this
        package -- notably the Agri-Gaia frontend, which builds this dict by
        hand -- would then have every DenseNet/YOLO job fail with a 422. Infer
        the tag from the fields that are actually present instead.

        Without a discriminator these unions mis-resolve silently: pydantic's
        smart union validates a minimal BirdNET payload as a DenseNet client and
        drops every BirdNET parameter without raising.
        """
        if not isinstance(data, dict):
            return data

        client = data.get("inference_client")
        if not isinstance(client, dict) or "client_type" in client:
            return data

        if "input_width" in client or "input_height" in client:
            client_type = "TritonYoloClient"
        elif any(
            key in client
            for key in ("sample_rate", "segment_seconds", "top_k", "overlap_seconds")
        ):
            client_type = "TritonBirdNetClient"
        else:
            client_type = "TritonDenseNetClient"

        # Copy rather than mutate: `data` may be a caller-owned dict.
        return {**data, "inference_client": {**client, "client_type": client_type}}


class DeviceCatalogEntry(BaseModel):
    """Non-derivable, admin-maintained metadata for a class of edge device.

    Keyed by the device's GPU model string (matched as a substring of
    ``DeviceInfo.gpu[*].model``) or pinned per hostname. ``tier_rank`` orders
    devices from smallest/cheapest (low) to largest/most capable (high) and is
    used as a cost proxy / tie-breaker. ``cost_eur`` is the acquisition price in
    euros. Costs are editable defaults — confirm against actual procurement.
    """

    gpu_model: str
    tier_rank: int
    cost_eur: Optional[float] = Field(default=None, ge=0)
    power_envelope_watts: Optional[float] = Field(default=None, ge=0)


class DeviceCandidateResult(BaseModel):
    """Outcome for a single candidate device in a recommendation run."""

    hostname: str
    benchmark_job_id: Optional[str] = Field(default=None)
    latency_ms: Optional[float] = Field(default=None)
    energy_joules: Optional[float] = Field(default=None)
    accuracy: Optional[float] = Field(default=None)
    cost_eur: Optional[float] = Field(default=None)
    tier_rank: Optional[int] = Field(default=None)
    meets_constraint: bool = Field(default=False)
    excluded_reason: Optional[str] = Field(default=None)


class DeviceRecommendation(BaseModel):
    """Result of an auto-search: the winning device plus the ranked candidates.

    ``winner_hostname`` is ``None`` when no candidate satisfied the latency
    constraint; ``candidates`` always lists every candidate (including excluded
    ones, each carrying an ``excluded_reason``) so callers can explain the
    outcome.
    """

    factor: OptimizationFactor
    latency_metric: LatencyPercentile
    latency_threshold_ms: float
    min_accuracy: Optional[float] = Field(default=None)
    accuracy_metric: str = Field(default="accuracy")
    winner_hostname: Optional[str] = Field(default=None)
    candidates: List[DeviceCandidateResult] = Field(default_factory=list)
