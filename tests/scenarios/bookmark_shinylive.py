"""Full application fixture forced into Shinylive runtime mode."""

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# Bookmark startup must not depend on remote dataset catalogues.
import seaborn
import ucimlrepo
from module import Module

seaborn.get_dataset_names = lambda: ["tips", "iris"]
ucimlrepo.list_available_datasets = lambda: None

Module.IS_SHINYLIVE = True
Module.runtime_mode = classmethod(lambda cls, session: "shinylive")

from app import app  # noqa: F401
