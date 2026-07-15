"""
Configuration file for doctests with PyTest.
"""

import pytest

from thztools import reset_option


# Reset options before each test
@pytest.fixture(autouse=True)
def global_reset() -> None:
    reset_option()
