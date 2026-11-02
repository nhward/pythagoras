"""Private subprocess entry point; inputs are application-owned temporary files."""
import json
import pickle
import sys
from pathlib import Path


def main():
    from code_recording import recorded_job
    from selection_analysis import analyze_selection, selection_stability
    source, method, options, diagnostic = pickle.loads(Path(sys.argv[1]).read_bytes())
    progress_path = Path(sys.argv[3])
    def progress(message):
        # Atomic replacement lets the parent read without partial JSON.
        pending = progress_path.with_suffix('.pending')
        pending.write_text(json.dumps(message))
        pending.replace(progress_path)
    result = recorded_job(selection_stability if diagnostic else analyze_selection, source, method, options, progress)
    Path(sys.argv[2]).write_bytes(pickle.dumps(result))


if __name__ == '__main__':
    main()
