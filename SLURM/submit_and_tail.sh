#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <job-file>"
  echo "Example: $0 train_qwen2.5-3b_test.job"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

JOB_FILE="$(basename "$1")"
if [ ! -f "$JOB_FILE" ]; then
  echo "Job file not found in $SCRIPT_DIR: $JOB_FILE"
  exit 1
fi

mkdir -p logs

SUBMIT_OUTPUT="$(sbatch "$JOB_FILE")"
echo "$SUBMIT_OUTPUT"

JOB_ID="$(awk '/Submitted batch job/ {print $4}' <<< "$SUBMIT_OUTPUT")"
if [ -z "$JOB_ID" ]; then
  echo "Could not read job id from sbatch output."
  exit 1
fi

LOG_FILE="logs/job_${JOB_ID}.out"
echo "Waiting for $LOG_FILE"
echo "Check queue with: squeue -j $JOB_ID"
echo "Stop watching with Ctrl+C; this will not cancel the job."

while [ ! -f "$LOG_FILE" ]; do
  if command -v squeue >/dev/null 2>&1; then
    QUEUE_STATE="$(squeue -h -j "$JOB_ID" -o "%T %R" 2>/dev/null || true)"
    if [ -n "$QUEUE_STATE" ]; then
      echo "Still waiting: $QUEUE_STATE"
    else
      echo "Job $JOB_ID is no longer in squeue, but $LOG_FILE was not created."
      if command -v sacct >/dev/null 2>&1; then
        sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,ExitCode,Reason%40 || true
      fi
      exit 1
    fi
  else
    echo "Log file not created yet."
  fi
  sleep 10
done

echo "Log file created: $LOG_FILE"
tail -f "$LOG_FILE"
