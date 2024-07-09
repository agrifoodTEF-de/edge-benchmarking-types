import psutil
import cpuinfo
import platform

from datetime import datetime
from pydantic import BaseModel, field_serializer
from typing import Optional, List, Dict, Any, ClassVar, Tuple
from edge_benchmarking_types.edge_device.enums import JobStatus


class BenchmarkJob(BaseModel):
    id: str
    benchmark_results: Dict[str, List]
    inference_results: Optional[Dict[str, List[Any]]]
    status: JobStatus


class InferenceServerStatus(BaseModel):
    ready: bool


class PlatformInfo(BaseModel):
    system: str = platform.system()
    architecture: Tuple[str, str] = platform.architecture()
    release: str = platform.release()
    version: str = platform.version()
    machine: str = platform.machine()
    libvc_ver: Tuple[str, str] = platform.libc_ver()
    python_version: str = platform.python_version()
    python_implementation: str = platform.python_implementation()
    python_compiler: str = platform.python_compiler()


class VirtualMemory(BaseModel):
    _vm: ClassVar = psutil.virtual_memory()

    total: int = _vm.total
    available: int = _vm.available
    percent: float = _vm.percent
    used: int = _vm.used
    free: int = _vm.free
    active: int = _vm.active
    inactive: int = _vm.inactive
    buffers: int = _vm.buffers
    cached: int = _vm.cached
    shared: int = _vm.shared
    slab: int = _vm.slab


class SwapMemory(BaseModel):
    _sm: ClassVar = psutil.swap_memory()
    total: int = _sm.total
    used: int = _sm.used
    free: int = _sm.free
    percent: float = _sm.percent
    sin: int = _sm.sin
    sout: int = _sm.sout


class MemoryInfo(BaseModel):
    virtual_memory: VirtualMemory = VirtualMemory()
    swap_memory: SwapMemory = SwapMemory()


class CpuInfo(BaseModel):
    _cpu: ClassVar = cpuinfo.get_cpu_info()

    arch_string_raw: str = _cpu["arch_string_raw"]
    vendor_id_raw: str = _cpu["vendor_id_raw"]
    brand_raw: str = _cpu["brand_raw"]
    hz_advertised_friendly: str = _cpu["hz_advertised_friendly"]
    hz_actual_friendly: str = _cpu["hz_actual_friendly"]
    hz_advertised: List[int] = _cpu["hz_advertised"]
    hz_actual: List[int] = _cpu["hz_actual"]
    arch: str = _cpu["arch"]
    bits: int = _cpu["bits"]
    count: int = _cpu["count"]
    l1_data_cache_size: int = _cpu["l1_data_cache_size"]
    l1_instruction_cache_size: int = _cpu["l1_instruction_cache_size"]
    l2_cache_size: int = _cpu["l2_cache_size"]
    l3_cache_size: int = _cpu["l3_cache_size"]
    model: int = _cpu["model"]
    flags: List[str] = _cpu["flags"]


class DiskUsage(BaseModel):
    _path: ClassVar[str] = "/"
    _du: ClassVar = psutil.disk_usage(path=_path)
    path: str = _path
    total: int = _du.total
    used: int = _du.used
    free: int = _du.free
    percent: float = _du.percent


class DiskIoCounters(BaseModel):
    _dioc: ClassVar = psutil.disk_io_counters()
    read_count: int = _dioc.read_count
    write_count: int = _dioc.write_count
    read_bytes: int = _dioc.read_bytes
    write_bytes: int = _dioc.write_bytes
    read_time: int = _dioc.read_time
    write_time: int = _dioc.write_time
    read_merged_count: int = _dioc.read_merged_count
    write_merged_count: int = _dioc.write_merged_count
    busy_time: int = _dioc.busy_time


class DiskInfo(BaseModel):
    usage: DiskUsage = DiskUsage()
    io_counters: DiskIoCounters = DiskIoCounters()


class NetworkInterface(BaseModel):
    duplex: str
    speed: int
    mtu: int
    flags: str


class NetworkInfo(BaseModel):
    interfaces: Dict[str, NetworkInterface] = {
        name: NetworkInterface(**iface._asdict())
        for name, iface in psutil.net_if_stats().items()
        if iface.isup and "loopback" not in iface.flags
    }


class SensorInfo(BaseModel):
    # TODO: Needed?
    _bat: ClassVar[Any] = psutil.sensors_battery()

    fan: Dict[str, List[Dict]] = {
        fan: [speed._asdict() for speed in speeds]
        for fan, speeds in psutil.sensors_fans().items()
    }
    temperatures: Dict[str, List[Dict]] = {
        component: [temp._asdict() for temp in temps]
        for component, temps in psutil.sensors_temperatures().items()
    }


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
    timestamp: datetime = datetime.now()
    online: Optional[bool] = None

    @field_serializer("timestamp")
    def serialize_timestamp(self, timestamp: datetime) -> str:
        return timestamp.isoformat()


class DeviceInfo(BaseModel):
    platform: PlatformInfo = PlatformInfo()
    memory: MemoryInfo = MemoryInfo()
    cpu: CpuInfo = CpuInfo()
    disk: DiskInfo = DiskInfo()
    network: NetworkInfo = NetworkInfo()
    sensor: SensorInfo = SensorInfo()
    gpu: Optional[List[GPU]] = None


class Device(BaseModel):
    header: DeviceHeader
    info: DeviceInfo = DeviceInfo()
