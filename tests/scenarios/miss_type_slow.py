"""Keep table diagnostics running while testing the independent front tree."""

import runpy
import time
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name("miss_type.py")))
import cards.miss_type as miss_type

original_table = miss_type._missingness_table


def slow_table(*args, **kwargs):
    time.sleep(10)
    return original_table(*args, **kwargs)


miss_type._missingness_table = slow_table
app = scenario["app"]
