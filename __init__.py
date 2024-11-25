"""
Bookshelf Scanner
A computer vision system for detecting and extracting text from book spines.
"""

__version__ = '0.1.0'

from BookSegmenter      import BookSegmenter
from TextExtractor      import TextExtractor
from ParameterOptimizer import ParameterOptimizer

__all__ = [
    'BookSegmenter',
    'TextExtractor', 
    'ParameterOptimizer'
]