#!/usr/bin/env bash
set -euo pipefail

PROXY_VARS=(http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY no_proxy)
for var in "${PROXY_VARS[@]}"; do
  if [[ -n "${!var-}" ]]; then
    unset "$var"
  fi
done

PDF_PATH="data/documents/sample_0.pdf"
if [[ $# -gt 0 ]]; then
  PDF_PATH="$1"
  shift
fi

uv run python main.py "$PDF_PATH" "$@"
