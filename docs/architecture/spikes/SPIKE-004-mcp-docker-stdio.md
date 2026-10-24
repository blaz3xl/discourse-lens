# SPIKE-004 — Docker MCP stdio

Docker Engine 29.7.2 was tested after explicit permission. A namespaced,
temporary `discourselens-spike-004:local` image was built from `python:3.14-slim`,
ran as UID 10001 with read-only root filesystem and `/tmp` tmpfs, accepted two
line-delimited JSON-RPC requests and emitted two parseable responses on stdout.
Logs/errors are directed to stderr. No ports, host networking, external network,
container name or Docker socket were used. The image was removed afterwards.

This validates the generic Docker+stdio operating contract only; it does not
validate any named MCP host or the production MCP SDK/server.
