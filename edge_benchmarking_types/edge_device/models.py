from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_serializer
from edge_benchmarking_types.edge_device.enums import JobStatus


class BenchmarkJob(BaseModel):
    id: str
    benchmark_results: Dict[str, List]
    inference_results: Optional[Dict[str, List[Any]]]
    status: JobStatus


class InferenceServerStatus(BaseModel):
    ready: bool


class PlatformInfo(BaseModel):
    system: str
    architecture: Tuple[str, str]
    release: str
    version: str
    machine: str
    libvc_ver: Tuple[str, str]
    python_version: str
    python_implementation: str
    python_compiler: str


class VirtualMemory(BaseModel):
    total: int
    available: int
    percent: float
    used: int
    free: int
    active: int
    inactive: int
    buffers: int
    cached: int
    shared: int
    slab: int


class SwapMemory(BaseModel):
    total: int
    used: int
    free: int
    percent: float
    sin: int
    sout: int


class MemoryInfo(BaseModel):
    virtual_memory: VirtualMemory
    swap_memory: SwapMemory


class CpuInfo(BaseModel):
    arch_string_raw: str
    vendor_id_raw: str
    brand_raw: str
    hz_advertised_friendly: str
    hz_actual_friendly: str
    hz_advertised: List[int]
    hz_actual: List[int]
    arch: str
    bits: int
    count: int
    l1_data_cache_size: int
    l1_instruction_cache_size: int
    l2_cache_size: int
    l3_cache_size: int
    model: int
    flags: List[str]


class DiskUsage(BaseModel):
    path: str
    total: int
    used: int
    free: int
    percent: float


class DiskIoCounters(BaseModel):
    read_count: int
    write_count: int
    read_bytes: int
    write_bytes: int
    read_time: int
    write_time: int
    read_merged_count: int
    write_merged_count: int
    busy_time: int


class DiskInfo(BaseModel):
    usage: DiskUsage
    io_counters: DiskIoCounters


class NetworkInterface(BaseModel):
    duplex: str
    speed: int
    mtu: int
    flags: str


class NetworkInfo(BaseModel):
    interfaces: Dict[str, NetworkInterface]


class SensorInfo(BaseModel):
    fan: Dict[str, List[Dict]]
    temperatures: Dict[str, List[Dict]]


class GPUStatus(BaseModel):
    railgate: bool
    tpc_pg_mask: bool
    three_d_scaling: bool
    load: float


class GPUFreq(BaseModel):
    governor: str
    cur: int
    max: int
    min: int
    GPC: List[int]


class GPU(BaseModel):
    model: str
    type: str
    status: GPUStatus
    freq: GPUFreq
    power_control: Optional[str]


class DeviceHeader(BaseModel):
    ip: str
    name: str
    hostname: str
    heartbeat_interval: int
    timestamp: datetime = Field(default_factory=datetime.now)
    online: Optional[bool] = Field(default=True)

    @field_serializer("timestamp")
    def serialize_timestamp(self, timestamp: datetime) -> str:
        return timestamp.isoformat()


class DeviceInfo(BaseModel):
    platform: PlatformInfo
    memory: MemoryInfo
    cpu: CpuInfo
    disk: DiskInfo
    network: NetworkInfo
    sensor: SensorInfo
    gpu: Optional[List[GPU]] = Field(default=None)


class Device(BaseModel):
    header: DeviceHeader
    info: DeviceInfo
