"""
Initialize the src package.
"""

from .agents import DDPGAgent, QRDDPGAgent
from .benchmark_strategies import BacktestBenchmark, BenchmarkStrategies
from .data_processor import DataProcessor
from .environment import PortfolioEnv
from .utils import calculate_portfolio_metrics, normalize_weights

__version__ = "1.0.0"
__author__ = "Abrar Ahmed"

__all__ = [
    "DataProcessor",
    "PortfolioEnv",
    "DDPGAgent",
    "QRDDPGAgent",
    "BenchmarkStrategies",
    "BacktestBenchmark",
    "calculate_portfolio_metrics",
    "normalize_weights",
]
