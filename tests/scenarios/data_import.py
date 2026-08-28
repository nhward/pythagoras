"""Test-owned Shiny application for the data-import browser tests."""

import os
import sys
from pathlib import Path

import seaborn
import ucimlrepo

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# Building the card UI enumerates these catalogues. Keep browser tests local,
# deterministic, and independent of the catalogue services.
seaborn.get_dataset_names = lambda: ["tips", "iris"]


def list_uci():
    print("Iris 53")
    print("Wine Quality 186")


ucimlrepo.list_available_datasets = list_uci

from cards.data_import import instance

this = instance()
app = this.application()
