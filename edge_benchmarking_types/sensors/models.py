from typing import Union
from pydantic import BaseModel, Field

from edge_benchmarking_types.sensors.enums import (
    OakImageResolution,
    WebcamImageFormat,
)


class SensorClient(BaseModel):
    ip: str


class OakClient(SensorClient):
    rgb_resolution: OakImageResolution
    rgb_queue_size: int = Field(default=1, ge=1)
    warmup: int = Field(default=3, ge=1)


class WebcamClient(SensorClient):
    port: int
    timeout: int = Field(default=3)
    img_format: WebcamImageFormat = Field(default=WebcamImageFormat.RAW)


class Sensor(BaseModel):
    client: Union[OakClient, WebcamClient]
    max_sample_size: int = Field(ge=1)
