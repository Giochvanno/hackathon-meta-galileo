# Security policy

## Supported versions

The latest release and `main` get fixes.

## Reporting a vulnerability

Please report privately: *Security → Report a vulnerability* on this repository (GitHub private
advisory). Do not open a public issue for a vulnerability, and never paste an API key anywhere,
including in a report. You should get an answer within a few days.

## What is in scope

* **The web app** (`galileo/web.py`), when someone hosts it with a Meta Model API key. It spends
  the host's credits, so it limits live runs per visitor (by the address the hosting proxy saw,
  which a visitor cannot fake), caps the queue at 4 jobs, refuses oversized uploads before
  reading them, checks a daily budget before every paid call, serves files only from fixed
  folders (path traversal is refused), and deletes visitors' videos after the measurement.
  Ways around any of these are in scope.
* **Secrets.** The key is read from the environment or a local `.env`, which is git-ignored and
  never baked into the Docker image; CI fails if a secret-like string is committed.

The public site on GitHub Pages is a static copy with no server and no key: it cannot spend
anyone's credits.

## Out of scope

Rate limits or outages of the Meta Model API itself; problems that need a key the attacker
already controls.
