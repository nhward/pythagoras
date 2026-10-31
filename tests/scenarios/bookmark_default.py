"""A reactive committed-name provider independent of bookmark data flow."""
import os
import sys
from pathlib import Path
APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))
from cards.sys_bookmark import instance
from module import Module
from shiny import reactive, ui

Module.runtime_mode = classmethod(lambda cls, session: 'server')
committed_name = reactive.Value('Zebra data')
this = instance(committed_data_name_provider=committed_name)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.input_select(
    'TestCommittedName', label='Committed import (test)',
    choices=['Zebra data', 'Accounts', 'Unavailable'], selected='Zebra data'))


def server(input, output, session):
    result = original_server(input, output, session)
    @reactive.effect
    def change_import():
        name = input.TestCommittedName()
        if name:
            committed_name.set(name)
    return result


this.server = server
app = this.application()
