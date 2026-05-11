#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${1:-$SCRIPT_DIR/logs}"

sanitize_path_part() {
  tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9._-]+/_/g; s/^_+//; s/_+$//'
}

first_match() {
  local pattern="$1"
  local file="$2"
  sed -nE "$pattern" "$file" | head -n 1
}

metadata_source_for() {
  local file="$1"
  local job_id
  local paired

  if grep -q "Starting .* [0-9][0-9]* on " "$file" 2>/dev/null; then
    printf '%s\n' "$file"
    return
  fi

  job_id="$(basename "$file" | sed -nE 's/^job_([0-9]+)\..*$/\1/p')"
  paired="$LOG_ROOT/job_${job_id}.out"
  if [ -n "$job_id" ] && [ -f "$paired" ]; then
    printf '%s\n' "$paired"
    return
  fi
  if [ -n "$job_id" ]; then
    paired="$(find "$LOG_ROOT" -type f -name "job_${job_id}.out" | head -n 1)"
    if [ -n "$paired" ]; then
      printf '%s\n' "$paired"
      return
    fi
  fi

  printf '%s\n' "$file"
}

gpu_count_from_cuda_visible_devices() {
  local value="$1"
  if [ -z "$value" ] || [ "$value" = "unset" ] || [ "$value" = "NoDevFiles" ]; then
    return
  fi
  awk -F',' '{print NF}' <<< "$value"
}

label_for_file() {
  local file="$1"
  local source_file
  local job_name
  local gpus
  local cuda_devices
  local steps
  local param_label

  source_file="$(metadata_source_for "$file")"
  job_name="$(first_match 's/.*Starting ([^[:space:]]+) [0-9][0-9]* on .*/\1/p' "$source_file")"
  if [ -z "$job_name" ]; then
    job_name="unknown_job"
  fi

  gpus="$(first_match 's/^Torch processes:[[:space:]]*([0-9]+).*/\1/p' "$source_file")"
  if [ -z "$gpus" ]; then
    cuda_devices="$(first_match 's/^CUDA_VISIBLE_DEVICES=(.*)$/\1/p' "$source_file")"
    gpus="$(gpu_count_from_cuda_visible_devices "$cuda_devices" || true)"
  fi

  steps="$(first_match 's/^[[:space:]]*steps\/batch\/gen:[[:space:]]*([0-9]+)\/([0-9]+)\/([0-9]+).*/steps\1_batch\2_gen\3/p' "$source_file")"

  param_label="gpus${gpus:-unknown}"
  if [ -n "$steps" ]; then
    param_label="${param_label}_${steps}"
  fi

  printf '%s/%s\n' \
    "$(printf '%s' "$job_name" | sanitize_path_part)" \
    "$(printf '%s' "$param_label" | sanitize_path_part)"
}

if [ ! -d "$LOG_ROOT" ]; then
  echo "Log directory not found: $LOG_ROOT"
  exit 1
fi

shopt -s nullglob
for log_file in "$LOG_ROOT"/job_*.out "$LOG_ROOT"/job_*.err; do
  [ -f "$log_file" ] || continue

  relative_group="$(label_for_file "$log_file")"
  destination_dir="$LOG_ROOT/$relative_group"
  destination="$destination_dir/$(basename "$log_file")"

  mkdir -p "$destination_dir"
  if [ "$log_file" = "$destination" ]; then
    continue
  fi

  if [ -e "$destination" ]; then
    echo "Keeping existing destination, leaving source in place: $destination"
    continue
  fi

  mv "$log_file" "$destination"
  echo "$log_file -> $destination"
done
