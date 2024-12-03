"""
Bookshelf Scanner
A computer vision system for detecting and extracting text from book spines.
"""

__version__ = '0.1.0'

from bookshelf_scanner.core.utils               import Utils
from bookshelf_scanner.core.module_logger       import ModuleLogger
from bookshelf_scanner.core.book_segmenter      import BookSegmenter
from bookshelf_scanner.core.text_extractor      import TextExtractor
from bookshelf_scanner.core.parameter_optimizer import ParameterOptimizer

__all__ = [
    'Utils',
    'ModuleLogger',
    'BookSegmenter',
    'TextExtractor', 
    'ParameterOptimizer'
]