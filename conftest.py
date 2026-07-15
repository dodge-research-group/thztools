import platform

import numpy as np
import pytest


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def pytest_configure(config: pytest.Config) -> None:
    if is_apple_silicon() and np.__version__ < "2.3":
        warning_filter = [
            "ignore:divide by zero encountered in matmul:RuntimeWarning",
            "ignore:overflow encountered in matmul:RuntimeWarning",
            "ignore:invalid value encountered in matmul:RuntimeWarning",
        ]
        for warn in warning_filter:
            config.addinivalue_line("filterwarnings", warn)
