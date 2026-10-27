# Investigating idle browser activity

Open the app with `?diagnostics=1` appended to its URL (use `&diagnostics=1`
if it already has a query string). Alternatively run
`PythagorasDiagnostics.start()` in the browser console. Sampling is disabled
by default. It counts Shiny events and DOM mutations without recording values
or sending diagnostic messages to the server. Mutation observation has overhead;
use this for investigation, then stop it.

Reproduce the high CPU usage, leave it running for at least 15 seconds, and run
`PythagorasDiagnostics.download()` in Safari's Web Inspector console. If useful,
stop the server and capture another 15 seconds before downloading. The capture
retains up to 120 five-second samples. `PythagorasDiagnostics.snapshot()` inspects
the capture; `PythagorasDiagnostics.stop()` stops collection but retains it.
Connection state remains null until a connection event is observed.

High repeated `DOM:<id>:<attribute>` counts identify mutation hotspots. Repeated
Shiny input/output names suggest a reactive feedback loop. Continuing mutations
after disconnection indicate browser activity independent of server computation.
The animation list is a snapshot at export time. A quiet capture does not exclude
CPU-heavy JavaScript, canvas/WebGL rendering, or browser layout/compositing work.
For those cases record a short Safari Web Inspector Timelines capture while CPU
is high; its JavaScript stacks and rendering activity are needed to locate the
source. Sampling elapsed time measures timer delay, not CPU usage.

For server-side extended-task cancellation details, start the usual server with
`PYTHAGORAS_DIAGNOSTICS=1` in its environment. The `pythagoras.tasks` logger records
the function, instance, generation, running/queued state, and cancellation reason.
This traces the app's extended tasks, not Shiny's internal invalidation timers.

In the installed Shiny implementation, `invalidate_later` cancels its timer when
the context invalidates or the session ends. It catches cancellation during sleep,
but prints and re-raises cancellation while waiting for the reactive lock. Thus
that traceback alone does not show an application task failing or a CPU loop.
No library exception suppression or scheduling changes are made by diagnostics.
