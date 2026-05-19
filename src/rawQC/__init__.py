"""
rawQC - Identification-Free Quality Control Metrics for Mass Spectrometry

This package provides comprehensive quality control metrics for mass spectrometry
data without requiring peptide/protein identification. It processes mzML files and
generates mzQC-compliant reports with optional visualizations.

Main Components:
    - calculate_metrics: Core function for computing QC metrics from mzML files
    - main: CLI entry point for command-line usage
    
Example Usage:
    >>> from rawQC import calculate_metrics
    >>> json_output = calculate_metrics(["sample1.mzML", "sample2.mzML"])
    
    Or via command line:
    $ python -m rawQC --demo --download-demo
"""

__version__ = "0.1.0"
__author__ = "Timo Sachsenberg"

# Import main public API
from .calculate_metrics import (
    calculate_metrics,
    main,
)

# Define public API
__all__ = [
    "calculate_metrics",
    "main",
    "__version__",
]

