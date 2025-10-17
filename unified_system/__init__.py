# -*- coding: utf-8 -*-
"""
Zunifikowany System Analizy WIG30 v2.0
Kompleksowy system do analizy fundamentalnej, technicznej i future scoring
"""

__version__ = "2.0.0"
__author__ = "WIG30 Radar Team"

from .core.unified_analyzer import UnifiedWIG30Analyzer
from .config.settings import config

__all__ = ['UnifiedWIG30Analyzer', 'config']