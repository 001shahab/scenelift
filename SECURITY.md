# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Write to
**shb@3sholding.com** with:

- what the problem is and where it is in the code,
- the steps to reproduce it,
- what an attacker could do with it.

You will get an acknowledgement, and then a fix or a plan. Please give us
reasonable time to release a fix before you publish details.

## Supported versions

Security fixes are made on the latest release.

## What to know before you deploy SceneLift Studio

SceneLift Studio (`python app.py`) is a local tool for one user.

- It binds to `127.0.0.1` by default, so it is reachable from your own computer only.
- **It has no authentication.** Anyone who can reach the port can read, edit and
  delete the scenes in the library folder and upload files up to the size limit.
  If you start it with `--host 0.0.0.0`, put it behind a reverse proxy that adds
  authentication and TLS, or keep it on a trusted network.
- Uploaded files are decoded with OpenCV and Pillow. Keep both up to date, and
  do not accept uploads from people you do not trust on a machine that matters.
- Scene ids are restricted to letters, digits, `-` and `_`, so a request cannot
  address a path outside the library folder.
- The optional model backends load third-party weights with PyTorch. Download
  checkpoints only from the official sources named in
  [docs/BACKENDS.md](docs/BACKENDS.md).
