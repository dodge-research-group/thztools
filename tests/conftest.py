import subprocess

import numpy
import pytest


def is_silicon_m4() -> bool:
    try:
        return (
            "M4"
            in subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            )
            .decode()
            .strip()
        )
    except Exception:
        return False


def pytest_configure(config: pytest.Config) -> None:
    if is_silicon_m4() and numpy.__version__ < "2.3":
        print(
            "M4 Chip detected with NumPy",
            numpy.__version__,
            ": filtering run-time warnings due to NumPy Issue #28687",
        )
        warning_filter = [
            "ignore:divide by zero encountered in matmul:RuntimeWarning",
            "ignore:overflow encountered in matmul:RuntimeWarning",
            "ignore:invalid value encountered in matmul:RuntimeWarning",
        ]
        for warn in warning_filter:
            config.addinivalue_line("filterwarnings", warn)
