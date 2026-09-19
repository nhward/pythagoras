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

from cards import data_import
import pandas as pd
from sklearn.datasets import load_iris

data_import.openml_catalogue = lambda page: pd.DataFrame([
    {"id": 61, "name": "iris", "version": 1, "rows": 150, "columns": 5}
])
def download_fixture(dataset_id):
    if str(dataset_id) != "61":
        raise ValueError("Fixture download failed")
    return load_iris(as_frame=True).frame

data_import.download_openml = download_fixture
from cards.data_import import instance

this = instance()
this.restore_configuration_state({
    "inputs": {"Navset": "OpenML", "OpenMLDataset": "61", "OName": "restored flowers", "OpenMLPage": 1},
    "last_committed_tab": "OpenML",
})
app = this.application()
