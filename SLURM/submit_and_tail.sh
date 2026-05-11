#!/bin/bash

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 [sbatch-options...] <job-file>"
  echo "Example: $0 train_qwen2.5-3b_test.job"
  echo "Example: $0 --gpus=2 --mem=240G train_qwen3.5-9b.job"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

JOB_FILE="$(basename "${@: -1}")"
SBATCH_ARGS=()
if [ "$#" -gt 1 ]; then
  SBATCH_ARGS=("${@:1:$#-1}")
fi

if [ ! -f "$JOB_FILE" ]; then
  echo "Job file not found in $SCRIPT_DIR: $JOB_FILE"
  exit 1
fi

sanitize_path_part() {
  tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9._-]+/_/g; s/^_+//; s/_+$//'
}

read_sbatch_value() {
  local key="$1"
  local value=""

  value="$(sed -nE "s/^#SBATCH[[:space:]]+--${key}=([^[:space:]]+).*/\\1/p" "$JOB_FILE" | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return
  fi

  sed -nE "s/^#SBATCH[[:space:]]+--${key}[[:space:]]+([^[:space:]]+).*/\\1/p" "$JOB_FILE" | head -n 1
}

override_value() {
  local key="$1"
  local value="$2"
  local index=0

  while [ "$index" -lt "${#SBATCH_ARGS[@]}" ]; do
    local arg="${SBATCH_ARGS[$index]}"
    if [[ "$arg" == "--${key}="* ]]; then
      value="${arg#*=}"
    elif [[ "$arg" == "--${key}" ]] && [ "$((index + 1))" -lt "${#SBATCH_ARGS[@]}" ]; then
      value="${SBATCH_ARGS[$((index + 1))]}"
    fi
    index=$((index + 1))
  done

  printf '%s\n' "$value"
}

JOB_NAME="$(override_value "job-name" "$(read_sbatch_value "job-name")")"
if [ -z "$JOB_NAME" ]; then
  JOB_NAME="${JOB_FILE%.job}"
fi

GPUS="$(override_value "gpus" "$(read_sbatch_value "gpus")")"
CPUS="$(override_value "cpus-per-task" "$(read_sbatch_value "cpus-per-task")")"
MEMORY="$(override_value "mem" "$(read_sbatch_value "mem")")"

JOB_LABEL="$(printf '%s' "$JOB_NAME" | sanitize_path_part)"
PARAM_LABEL="gpus${GPUS:-unknown}_cpus${CPUS:-unknown}_mem${MEMORY:-unknown}"
PARAM_LABEL="$(printf '%s' "$PARAM_LABEL" | sanitize_path_part)"
LOG_DIR="logs/$JOB_LABEL/$PARAM_LABEL"
mkdir -p "$LOG_DIR"

SUBMIT_OUTPUT="$(sbatch "${SBATCH_ARGS[@]}" --output="$LOG_DIR/job_%j.out" --error="$LOG_DIR/job_%j.out" "$JOB_FILE")"
echo "$SUBMIT_OUTPUT"

JOB_ID="$(awk '/Submitted batch job/ {print $4}' <<< "$SUBMIT_OUTPUT")"
if [ -z "$JOB_ID" ]; then
  echo "Could not read job id from sbatch output."
  exit 1
fi

LOG_FILE="$LOG_DIR/job_${JOB_ID}.out"
echo "Logs grouped under: $LOG_DIR"
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
