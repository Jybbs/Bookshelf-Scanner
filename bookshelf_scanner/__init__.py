"""
Bookshelf Scanner
A computer vision system for detecting and extracting text from book spines.
"""

__version__ = '0.1.0'

from bookshelf_scanner.core.utils            import Utils
from bookshelf_scanner.core.module_logger    import ModuleLogger
from bookshelf_scanner.core.book_segmenter   import BookSegmenter
from bookshelf_scanner.core.text_extractor   import TextExtractor
from bookshelf_scanner.core.config_optimizer import ConfigOptimizer
from bookshelf_scanner.core.fuzzy_matcher    import FuzzyMatcher

__all__ = [
    'Utils',
    'ModuleLogger',
    'BookSegmenter',
    'TextExtractor', 
    'ConfigOptimizer',
    'FuzzyMatcher'
]