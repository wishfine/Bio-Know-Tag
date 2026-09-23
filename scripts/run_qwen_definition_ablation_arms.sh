#!/usr/bin/env bash
# Run four repeated-ablation arms in ABBA order against one shared Qwen pool.
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 smoke|full RUN_ROOT MODEL ENDPOINT [ENDPOINT ...]" >&2
  exit 2
fi

mode=$1
run_root=$2
model=$3
shift 3
endpoints=("$@")
sample_root=${SAMPLE_ROOT:-runtime/20260921-184755-definition-ablation}
workers=${QWEN_ABLATION_TOTAL_WORKERS:-240}
per_endpoint_limit=${QWEN_ABLATION_PER_ENDPOINT_LIMIT:-30}

if [[ $mode != smoke && $mode != full ]]; then
  echo "Mode must be smoke or full" >&2
  exit 2
fi

for file in name_only_tasks.jsonl name_plus_definition_tasks.jsonl name_only_labels.jsonl name_plus_definition_labels.jsonl paired_tasks.jsonl; do
  test -s "$sample_root/$file" || { echo "Missing $sample_root/$file" >&2; exit 1; }
done

if (( workers < 1 || per_endpoint_limit < 1 )); then
  echo "Worker and per-endpoint limits must be positive" >&2
  exit 2
fi

for endpoint in "${endpoints[@]}"; do
  models_url="${endpoint%/chat/completions}/models"
  curl -fsS --connect-timeout 3 --max-time 10 "$models_url" >/dev/null || {
    echo "Qwen service unavailable: $models_url" >&2
    exit 1
  }
done

mkdir -p "$run_root/$mode"
{
  printf 'model=%s\nmode=%s\nworkers=%s\nper_endpoint_limit=%s\n' \
    "$model" "$mode" "$workers" "$per_endpoint_limit"
  printf 'endpoint=%s\n' "${endpoints[@]}"
} > "$run_root/$mode/launch-parameters.txt"

for arm in name_1 definition_1 definition_2 name_2; do
  case "$arm" in
    name_*) tasks=name_only_tasks.jsonl; labels=name_only_labels.jsonl ;;
    definition_*) tasks=name_plus_definition_tasks.jsonl; labels=name_plus_definition_labels.jsonl ;;
  esac
  arm_dir="$run_root/$mode/$arm"
  mkdir -p "$arm_dir"
  if [[ $mode == full && -s $arm_dir/pid ]]; then
    read -r old_pid < "$arm_dir/pid"
    if kill -0 "$old_pid" 2>/dev/null; then
      echo "Already running $arm: PID $old_pid" >&2
      exit 1
    fi
  fi

  args=(
    scripts/run_definition_coverage_batches.py
    --tasks "$sample_root/$tasks"
    --labels "$sample_root/$labels"
    --run-dir "$arm_dir"
    --model "$model"
    --max-batch-size 40
    --char-budget 55000
    --max-tokens 2048
    --timeout 600
    --retries 3
    --retry-delay 1
    --request-interval 0
    --max-in-flight-per-endpoint "$per_endpoint_limit"
  )
  for endpoint in "${endpoints[@]}"; do
    args+=(--endpoint "$endpoint")
  done
  if [[ $mode == smoke ]]; then
    env PYTHONPATH=src python -u "${args[@]}" --workers 1 --limit 30 > "$arm_dir/run.log" 2>&1
    echo "$arm smoke: $(python -c 'import json,sys; r=json.load(open(sys.argv[1])); print("processed={} error={}".format(r["processed"],r["error"]))' "$arm_dir/report.json")"
  else
    env PYTHONPATH=src python -u "${args[@]}" --workers "$workers" > "$arm_dir/nohup.log" 2>&1 &
    printf '%s\n' "$!" > "$arm_dir/pid"
    echo "$arm full: PID $!; log=$arm_dir/nohup.log"
    wait "$!"
    echo "$arm complete"
  fi
done
