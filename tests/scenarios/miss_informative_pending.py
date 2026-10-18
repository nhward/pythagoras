"""A forest task kept running to distinguish diagnostic and export dependencies."""
import asyncio
from pathlib import Path
import runpy

scenario = runpy.run_path(str(Path(__file__).with_name("miss_informative.py")))
this = scenario["this"]
decorate = this.extended_task


def pending_task(function):
    async def wait(*args, **kwargs):
        await asyncio.Event().wait()
    return decorate(wait)


this.extended_task = pending_task
this.restore_configuration_state({"inputs": {"Shadow": ["x"], "InformativeOnly": False}})
app = this.application()
