#!/usr/bin/env bash
# Helper to run on the MacBook M2 Max BEFORE deploying Hermès.
#
# 1) Install Ollama (one-time):
#       brew install ollama
# 2) Pull the recommended models:
#       ollama pull qwen3-coder-next      # primary brain (80B-A3B MoE, ~40 GB, 4-bit)
#       ollama pull qwen3.6:27b           # secondary (dense, fast classifier)
# 3) Run this script.
#
# It binds Ollama to the Tailscale interface so ONLY tailnet peers can hit it,
# and tags this node so ACLs can scope access to tag:hermes.

set -euo pipefail

if ! command -v tailscale >/dev/null; then
  echo "Tailscale not found — install from https://tailscale.com/download/mac"; exit 1
fi
if ! command -v ollama >/dev/null; then
  echo "Ollama not found — install with: brew install ollama"; exit 1
fi

TS_HOSTNAME=${TS_HOSTNAME:-hermes-mac}
sudo tailscale up --hostname="$TS_HOSTNAME" --advertise-tags=tag:mac-llm --ssh

TS_IP=$(tailscale ip -4)
echo "Tailscale IP : $TS_IP  (hostname: $TS_HOSTNAME)"

export OLLAMA_HOST="${TS_IP}:11434"
export OLLAMA_KEEP_ALIVE="24h"
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_NUM_PARALLEL=2

echo "Starting Ollama on $OLLAMA_HOST ..."
exec ollama serve
