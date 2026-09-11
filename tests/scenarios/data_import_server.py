"""Data-import browser fixture forced into remote-server mode."""

import os
import sys
from pathlib import Path

import seaborn
import ucimlrepo

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

seaborn.get_dataset_names = lambda: ["tips", "iris"]


def list_uci():
    print("Iris 53")
    print("Wine Quality 186")


ucimlrepo.list_available_datasets = list_uci

import cards.data_import as data_import

data_import.Module.runtime_mode = classmethod(lambda cls, session: "server")

this = data_import.instance()
app = this.application()
