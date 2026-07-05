"""Shared pytest configuration.

This adds the src directory to the import path so the test modules can import the simulation
engine, cell model, algorithms, and profile generators directly by name.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
