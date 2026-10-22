"""
Pytest configuration and environment setup for SatQuery AI Backend.
Ensures repository root is always present on sys.path.
"""
import os
import sys

repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
