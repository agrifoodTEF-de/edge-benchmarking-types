from typing import Optional, List
from pydantic import BaseModel, Field

from edge_benchmarking_types.tools_api.enums import (
    OnnxModelInputType,
    OnnxExportMode,
    OnnxOperatorExportType,
)


class PytorchToOnnxConversionConfig(BaseModel):
    input_names: Optional[List[str]] = Field(default=None)
    input_shapes: List[List[int]]
    input_types: List[OnnxModelInputType]
    output_names: Optional[List[str]] = Field(default=None)
    verbose: bool = Field(default=False)
    optimize: bool = Field(default=False)
    training: OnnxExportMode = Field(default=OnnxExportMode.EVAL)
    operator_export_type: OnnxOperatorExportType = Field(
        default=OnnxOperatorExportType.ONNX
    )
    opset_version: int = Field(gt=0, default=14)
