"""Flight dynamics modeling package (Transformer-LSTM hybrid)"""

from .core.model_architecture import FlightModelBuilder
from .core.data_processing import FlightDataProcessor
from .utils.visualization import ResultVisualizer

__version__ = "1.0.0"
__all__ = [
    'FlightModelBuilder',
    'FlightDataProcessor',
    'ResultVisualizer'
]

# 包初始化日志
if __debug__:
    print(f"Initialized {__name__} v{__version__}")