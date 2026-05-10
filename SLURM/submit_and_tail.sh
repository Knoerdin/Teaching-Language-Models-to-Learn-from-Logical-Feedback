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

tail -F "$LOG_FILE"
