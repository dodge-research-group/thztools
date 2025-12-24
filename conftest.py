import subprocess

import numpy as np
import pytest


def is_silicon_m4() -> bool:
    try:
        return (
            "M4"
            in subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
            )
            .decode()
            .strip()
        )
    except FileNotFoundError:
        return False


def pytest_configure(config: pytest.Config) -> None:
    if is_silicon_m4() and np.__version__ < "2.3":
        warning_filter = [
            "ignore:divide by zero encountered in matmul:RuntimeWarning",
            "ignore:overflow encountered in matmul:RuntimeWarning",
            "ignore:invalid value encountered in matmul:RuntimeWarning",
        ]
        for warn in warning_filter:
            config.addinivalue_line("filterwarnings", warn)
