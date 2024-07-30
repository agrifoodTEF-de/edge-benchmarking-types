from enum import Enum


class JobStatus(str, Enum):
    CREATING = "creating"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class InferenceServerType(str, Enum):
    TRITON = "triton"


class ContainerAction(str, Enum):
    REMOVE = "remove"
    CREATE = "create"
