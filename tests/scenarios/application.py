"""Full application with deterministic dataset catalogues for browser tests."""
import os
import sys
from pathlib import Path

import seaborn
import ucimlrepo

APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# Rendering data_import builds both catalogues, including on hidden tabs.
# App/workflow tests use local files or sklearn datasets, not these services.
seaborn.get_dataset_names = lambda: ['tips', 'iris']
ucimlrepo.list_available_datasets = lambda: None

from app import app  # noqa: F401
