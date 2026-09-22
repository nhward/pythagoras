# Nightly tests in Positron

The **Pythagoras: nightly test suite** task runs all tests (including browser tests)
at 02:00 in the Mac's local timezone. Cron Tasks must be enabled, this workspace
must be open in Positron, and the Mac must be awake at that time. The extension
cannot wake the Mac or run while Positron is closed. `caffeinate -i` prevents idle
sleep during a run; it does not wake the computer beforehand.

- View/run manually: Command Palette → **Tasks: Run Task** → **Pythagoras: nightly test suite**.
- Edit the command: `.vscode/tasks.json`.
- Edit/disable the schedule: `.vscode/settings.json`, `cronTasks.tasks` (`0 2 * * *`).
- Check scheduler registration: **View → Output → Cron Tasks**. Reload the Positron window if necessary.
- Read results: `.make/nightly-tests/`, timestamped `.log` and JUnit `.xml` files.
  These reports are ignored by Git and retained until you remove them.

Failures, collection errors, and the four-hour timeout trigger an email to
**nhward60@gmail.com** using Apple Mail's default sending account. Successful runs
send no email. Failed runs remain failed even if email delivery fails; notification
errors are appended to the log. Reports contain test output and local file paths.

Apple Mail must have a working sending account. macOS may ask permission for
Positron (or its Python/terminal process) to control Mail on the first notification.
Allow that under **System Settings → Privacy & Security → Automation** when prompted.
Delivery has not been verified by a live email. A report accepted by Apple Mail may
remain in its outbox while offline. Run manually while present before relying on
unattended notifications; a successful test run will not exercise email permissions.

Check runner prerequisites without running tests or sending mail:

```sh
.venv/bin/python scripts/nightly_tests.py --check
```
