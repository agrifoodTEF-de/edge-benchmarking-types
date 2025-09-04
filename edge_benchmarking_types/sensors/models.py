from typing import Union, Optional
from pydantic import BaseModel, Field, ConfigDict

from edge_benchmarking_types.sensors.enums import (
    SensorType,
    WebcamImageFormat,
    OakImageResolution,
)


class OakClientConfig(BaseModel):
    rgb_resolution: OakImageResolution
    rgb_queue_size: int = Field(default=1, ge=1)
    warmup: int = Field(default=3, ge=1)


class WebcamClientConfig(BaseModel):
    port: int = Field(ge=1, le=65535)
    timeout: int = Field(default=3)
    img_format: WebcamImageFormat = Field(default=WebcamImageFormat.RAW)


class SensorConfig(BaseModel):
    client_config: Union[OakClientConfig, WebcamClientConfig]
    max_sample_size: int = Field(ge=1)


class SensorInfo(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    type: SensorType
    name: str
    manufacturer: str
    model: str
    serial: str
    hostname: str
    ip: str
    online: Optional[bool] = Field(default=False)
