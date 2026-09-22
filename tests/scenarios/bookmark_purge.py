"""Isolated local bookmark card; the test supplies a temporary bookmark folder."""
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))

from cards.sys_bookmark import instance
from module import Module

Module.runtime_mode = classmethod(lambda cls, session: 'local')
this = instance()
app = this.application()
