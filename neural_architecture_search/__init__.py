"""
Neural Architecture Search (NAS) package for wireless signal classification

This package provides automated neural architecture discovery for wireless signal
classification models using evolutionary algorithms and multi-objective optimization.
"""

__version__ = "1.0.0"
__author__ = "Wireless Signal Classification Team"

from .nas_optimization import WirelessSignalNAS, run_nas_demo

__all__ = [
    'WirelessSignalNAS',
    'run_nas_demo'
]
