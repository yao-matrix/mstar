"""Compatibility imports for the accelerator graph configuration API."""

from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig as CudaGraphConfig,
)
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfigType as CudaGraphConfigType,
)
from mstar.engine.accelerator_graph_config import (
    BasicBatchedAcceleratorGraphConfig as BasicBatchedCudaGraphConfig,
)
from mstar.engine.accelerator_graph_config import (
    FlashInferPackedCudaGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCaptureShape,
    PiecewiseConfigType,
    PiecewisePackedConfig,
    distribute_tokens,
)
from mstar.engine.accelerator_graph_config import (
    PiecewiseAcceleratorGraphConfig as PiecewiseCudaGraphConfig,
)

__all__ = [
    "BasicBatchedCudaGraphConfig",
    "CudaGraphConfig",
    "CudaGraphConfigType",
    "FlashInferPackedCudaGraphConfig",
    "PiecewiseBatchedConfig",
    "PiecewiseCaptureShape",
    "PiecewiseConfigType",
    "PiecewiseCudaGraphConfig",
    "PiecewisePackedConfig",
    "distribute_tokens",
]
