"""
conftest.py
-----------
Pytest configuration. Ensures the project root is on sys.path so
test modules can import from src/ and api/ without installing the package.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
