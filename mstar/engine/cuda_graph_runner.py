"""Compatibility imports for the accelerator graph runner API."""

from mstar.engine.accelerator_graph_config import BasicBatchedAcceleratorGraphConfig as BasicBatchedCudaGraphConfig
from mstar.engine.accelerator_graph_runner import (
    AcceleratorGraphData as CudaGraphData,
)
from mstar.engine.accelerator_graph_runner import (
    AcceleratorGraphKey as CudaGraphKey,
)
from mstar.engine.accelerator_graph_runner import (
    AcceleratorGraphRunner as CudaGraphRunner,
)
from mstar.engine.accelerator_graph_runner import (
    AcceleratorGraphSlot as CudaGraphSlot,
)
from mstar.engine.accelerator_graph_runner import (
    PiecewiseAcceleratorGraphRunner as PiecewiseCudaGraphRunner,
)
from mstar.engine.accelerator_graph_runner import (
    StatelessAcceleratorGraphRunner as StatelessCudaGraphRunner,
)
from mstar.engine.accelerator_graph_runner import (
    build_piecewise_runners,
)

__all__ = [
    "BasicBatchedCudaGraphConfig",
    "CudaGraphData",
    "CudaGraphKey",
    "CudaGraphRunner",
    "CudaGraphSlot",
    "PiecewiseCudaGraphRunner",
    "StatelessCudaGraphRunner",
    "build_piecewise_runners",
]
