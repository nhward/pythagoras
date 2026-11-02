"""Cancellable, isolated native fits; cancelling stops CPU work, not just await."""
import asyncio
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path


async def run_selection(source, method, options, progress=None, *, diagnostic=False):
    from code_recording import collect_job_code
    process = None
    with tempfile.TemporaryDirectory(prefix='pythagoras-selection-') as directory:
        root = Path(directory)
        request, response, status = (root / n for n in ('request.pkl', 'response.pkl', 'progress.json'))
        request.write_bytes(pickle.dumps((source, method, options, diagnostic)))
        try:
            environment = dict(os.environ, OMP_NUM_THREADS=str(options['n_jobs']),
                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            # No shell, no inherited UI/session objects, and bounded stderr.
            with (root / 'stderr.txt').open('wb') as errors:
                process = await asyncio.create_subprocess_exec(sys.executable,
                    str(Path(__file__).with_name('feature_selection_worker.py')),
                    str(request), str(response), str(status), env=environment,
                    stdout=asyncio.subprocess.DEVNULL, stderr=errors)
                last = None
                while process.returncode is None:
                    try:
                        await asyncio.wait_for(process.wait(), timeout=.2)
                    except asyncio.TimeoutError:
                        pass
                    if status.exists() and progress is not None:
                        value = json.loads(status.read_text())
                        if value != last:
                            await progress(value)
                            last = value
            if process.returncode or not response.exists():
                detail = (root / 'stderr.txt').read_text()[-2000:]
                raise RuntimeError(f'Feature-selection worker failed: {detail}')
            return collect_job_code([pickle.loads(response.read_bytes())])[0]
        finally:
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
