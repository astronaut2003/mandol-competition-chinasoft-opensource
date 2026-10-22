"""Compatibility import; implementation lives in the shared mandol-substrate package."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module('mandol_substrate.gates')
