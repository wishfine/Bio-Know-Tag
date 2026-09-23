#!/usr/bin/env bash
# Run four repeated-ablation arms concurrently, each on two exclusive endpoints.
set -euo pipefail

if [[ $# -ne 11 ]]; then
  echo "Usage: $0 smoke|full RUN_ROOT MODEL ENDPOINT_9304 ... ENDPOINT_9311" >&2
  exit 2
fi

mode=$1
run_root=$2
model=$3
shift 3
endpoints=("$@")
sample_root=${SAMPLE_ROOT:-runtime/20260921-184755-definition-ablation}
workers=${QWEN_ABLATION_WORKERS_PER_ARM:-60}
per_endpoint_limit=${QWEN_ABLATION_PER_ENDPOINT_LIMIT:-30}

if [[ $mode != smoke && $mode != full ]]; then
  echo "Mode must be smoke or full" >&2
  exit 2
fi

for file in name_only_tasks.jsonl name_plus_definition_tasks.jsonl name_only_labels.jsonl name_plus_definition_labels.jsonl paired_tasks.jsonl; do
  test -s "$sample_root/$file" || { echo "Missing $sample_root/$file" >&2; exit 1; }
done

if (( workers < 1 || per_endpoint_limit < 1 || workers > 2 * per_endpoint_limit )); then
  echo "Worker and per-endpoint limits must be positive" >&2
  exit 2
fi
if [[ $(printf '%s\n' "${endpoints[@]}" | sort -u | wc -l | tr -d ' ') -ne 8 ]]; then
  echo "All eight endpoints must be distinct" >&2
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
  printf 'model=%s\nmode=%s\nworkers_per_arm=%s\nper_endpoint_limit=%s\n' \
    "$model" "$mode" "$workers" "$per_endpoint_limit"
  for index in 0 1 2 3; do
    arm=(name_1 definition_1 name_2 definition_2)
    printf '%s endpoint=%s endpoint=%s\n' \
      "${arm[$index]}" "${endpoints[$((2 * index))]}" "${endpoints[$((2 * index + 1))]}"
  done
} > "$run_root/$mode/launch-parameters.txt"

arms=(name_1 definition_1 name_2 definition_2)
for arm in "${arms[@]}"; do
  if [[ $mode == full && -s $run_root/$mode/$arm/pid ]]; then
    read -r old_pid < "$run_root/$mode/$arm/pid"
    if kill -0 "$old_pid" 2>/dev/null; then
      echo "Already running $arm: PID $old_pid" >&2
      exit 1
    fi
  fi
done

pids=()
for index in 0 1 2 3; do
  arm=${arms[$index]}
  case "$arm" in
    name_*) tasks=name_only_tasks.jsonl; labels=name_only_labels.jsonl ;;
    definition_*) tasks=name_plus_definition_tasks.jsonl; labels=name_plus_definition_labels.jsonl ;;
  esac
  arm_dir="$run_root/$mode/$arm"
  mkdir -p "$arm_dir"
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
  args+=(--endpoint "${endpoints[$((2 * index))]}")
  args+=(--endpoint "${endpoints[$((2 * index + 1))]}")
  if [[ $mode == smoke ]]; then
    env PYTHONPATH=src python -u "${args[@]}" --workers 2 --limit 30 > "$arm_dir/run.log" 2>&1 &
  else
    env PYTHONPATH=src python -u "${args[@]}" --workers "$workers" > "$arm_dir/nohup.log" 2>&1 &
  fi
  printf '%s\n' "$!" > "$arm_dir/pid"
  pids+=("$!")
  echo "$arm started: PID $!; endpoints=${endpoints[$((2 * index))]},${endpoints[$((2 * index + 1))]}"
done

status=0
for index in 0 1 2 3; do
  if ! wait "${pids[$index]}"; then
    echo "${arms[$index]} failed; see $run_root/$mode/${arms[$index]}/run.log or nohup.log" >&2
    status=1
  elif [[ $mode == smoke ]]; then
    arm_dir="$run_root/$mode/${arms[$index]}"
    echo "${arms[$index]} smoke: $(python -c 'import json,sys; r=json.load(open(sys.argv[1])); print("processed={} error={}".format(r["processed"],r["error"]))' "$arm_dir/report.json")"
  else
    echo "${arms[$index]} complete"
  fi
done
exit "$status"
