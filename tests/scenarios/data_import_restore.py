"""Test-owned app for restoring a committed data-import card state."""

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
ucimlrepo.list_available_datasets = lambda: None

from cards.data_import import instance

this = instance()
this.restore_configuration_state({
    "inputs": {
        "Navset": "Web based",
        "ServerFile": None,
        "LocalFilePath": "",
        "FName": "file draft",
        "Dataset": "sklearn::iris",
        "DName": "restored iris",
        "Url": "https://example.test/uncommitted.csv",
        "UName": "web draft",
        "UciDataset": None,
        "IName": "",
        "Separator": ",",
        "Sheet": 1,
    },
    "last_committed_tab": "Dataset based",
})
app = this.application()
