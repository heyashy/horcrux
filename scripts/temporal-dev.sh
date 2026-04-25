#!/usr/bin/env bash
# Temporal dev server — single binary, in-memory event history.
#   Frontend (SDK):  localhost:7233
#   UI:              http://localhost:8233
#
# Restart wipes history. Don't restart this between worker restarts during the
# Phase 4 crash test — only the WORKER gets killed; the server stays up.

set -euo pipefail

exec temporal server start-dev \
    --port 7233 \
    --ui-port 8233 \
    --namespace default
