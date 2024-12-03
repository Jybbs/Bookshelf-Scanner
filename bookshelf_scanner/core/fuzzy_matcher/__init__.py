"""
Text matching between OCR results and master book database.
"""

from .matcher import FuzzyMatcher

__all__ = ['TextMatcher']