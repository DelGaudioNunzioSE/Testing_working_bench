"""Binoculars - Zero-Shot Detection of Machine-Generated Text

This implementation is adapted from the original repository:
https://github.com/ahans30/Binoculars

License: BSD-3-Clause License
Copyright (c) 2024, ahans30 (original implementation)

Reference:
    Hans, A., Schwarzschild, A., Cherepanova, V., Kazemi, H., Saha, A., 
    Goldblum, M., Geiping, J., & Goldstein, T. (2024). 
    "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text"
    arXiv:2401.12070
"""

from .detector import Binoculars

__all__ = ["Binoculars"]
