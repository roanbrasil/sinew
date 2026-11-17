#!/usr/bin/env bash
# Opens (or comments on) one issue per failing nightly job, so a broken
# provider is reported once instead of every night.
set -euo pipefail
job="$1"
title="Nightly live tests failing: $job"
run_url="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
existing=$(gh issue list --state open --search "\"$title\" in:title" --json number --jq '.[0].number // empty')
if [ -n "$existing" ]; then
  gh issue comment "$existing" --body "Still failing: $run_url"
else
  gh issue create --title "$title" --label bug --label testing \
    --body "The nightly \`Live\` workflow failed for **$job**: $run_url

Recorded payloads may have drifted from the real API, a model may have changed, or a service is down. Reproduce locally with the commands in \`.github/workflows/live.yml\`."
fi
