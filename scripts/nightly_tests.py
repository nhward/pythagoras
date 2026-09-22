"""Run the full suite and email failures through the Mac's Apple Mail account."""
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / '.make' / 'nightly-tests'
RECIPIENT = 'nhward60@gmail.com'
MAIL_SCRIPT = '''on run argv
    set recipientAddress to item 1 of argv
    set subjectText to item 2 of argv
    set bodyText to item 3 of argv
    set logFile to POSIX file (item 4 of argv)
    tell application "Mail"
        set reportMessage to make new outgoing message with properties {subject:subjectText, content:bodyText, visible:false}
        tell reportMessage
            make new to recipient at end of to recipients with properties {address:recipientAddress}
            make new attachment with properties {file name:logFile} at after the last paragraph
        end tell
        send reportMessage
    end tell
end run
'''


def email_failure(log, code):
    body = (f'Pythagoras nightly tests failed with exit code {code}.\n'
            f'Checkout: {ROOT}\nLog: {log}\n\n'
            'The complete pytest output is attached.\n\n' +
            log.read_text(errors='replace')[-12000:])
    subprocess.run(
        ['/usr/bin/osascript', '-', RECIPIENT,
         f'Pythagoras nightly tests FAILED — {log.stem}', body, str(log)],
        input=MAIL_SCRIPT, text=True, check=True, timeout=120,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check setup without running tests or sending email')
    args = parser.parse_args()
    python = ROOT / '.venv/bin/python'
    if args.check:
        assert python.is_file(), f'Missing interpreter: {python}'
        assert Path('/usr/bin/osascript').is_file(), 'AppleScript is required'
        print(f'Ready: {python} -m pytest tests; failure recipient: {RECIPIENT}')
        print('Apple Mail delivery and macOS Automation permission require a real failure notification.')
        return 0
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / 'run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('A nightly test run is already active; skipping.')
            return 0
        stamp = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S_%z')
        log = REPORTS / f'{stamp}.log'
        report = REPORTS / f'{stamp}.xml'
        command = [str(python), '-m', 'pytest', 'tests', '--tb=short', f'--junitxml={report}']
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'app'), PYTHONUNBUFFERED='1')
        print(f'Running full suite. Output: {log}', flush=True)
        with log.open('w') as output:
            output.write(f'Started: {stamp}\nCheckout: {ROOT}\nCommand: {command!r}\n\n')
            output.flush()
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=4 * 60 * 60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    output.write('\nNightly suite exceeded the four-hour limit.\n')
                    code = 124
            except OSError as error:
                output.write(f'\nCould not start pytest: {error}\n')
                code = 127
            output.write(f'\nExit code: {code}\n')
        if code:
            try:
                email_failure(log, code)
                print(f'Failure report submitted to Apple Mail for {RECIPIENT}.')
            except (OSError, subprocess.SubprocessError) as error:
                with log.open('a') as output:
                    output.write(f'\nEmail notification failed: {error}\n')
                print(f'EMAIL FAILED: {error}. Test results remain at {log}', file=sys.stderr)
        else:
            print('All tests passed; no email sent.')
        return code if code >= 0 else 1


if __name__ == '__main__':
    sys.exit(main())
