#!/usr/bin/env bash

# Corrected B4 generation launcher. SERVER ONLY.
#
# Configured inference routes:
#   Llama 3.1 70B -> OpenRouter -> CoreWeave (bf16)
#   Qwen 2.5 72B  -> Hugging Face Inference Providers -> Novita (bf16)
#   GPT-5-mini     -> OpenAI

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_generation.sh [options]

Options:
  --dry-run          Build prompts without API calls; temporary outputs are deleted.
  --pilot            One eligible PR: six Llama contexts and P5/P6 for Qwen/GPT.
  --limit N          Set both issue and Llama-control limits (pilot/testing alias).
  --issue-limit N    P5/P6 PRs per model (default: 273).
  --llama-control-limit N
                     Llama P1-P4 PRs (default: 308).
  --workers N        API workers per generator (default: 5).
  --run-id ID        Versioned output identifier (default: b4-corrected-v3).
  --output-root DIR  Parent output directory.
  -h, --help         Show this help.

Required environment variables for a real run:
  OPENROUTER_API_KEY  Llama via OpenRouter/CoreWeave
  HF_TOKEN            Qwen via Hugging Face/Novita
  OPENAI_API_KEY      GPT-5-mini via OpenAI

Optional:
  GENERATION_PYTHON   Absolute Python interpreter; otherwise `pyenv exec python`.
EOF
}

b4_issue_limit=273
b4_llama_control_limit=308
b4_llama_p4_file_token_budget=90000
b4_llama_truncated_p4_completion_token_cap=4096
b4_workers=5
b4_run_id="b4-corrected-v3"
b4_dry_run=0
b4_output_root="phase3/datasets/temperature_1"

while (($#)); do
  case "$1" in
    --dry-run)
      b4_dry_run=1
      shift
      ;;
    --pilot)
      b4_issue_limit=1
      b4_llama_control_limit=1
      shift
      ;;
    --limit)
      [[ $# -ge 2 ]] || { echo "--limit requires a value" >&2; exit 2; }
      b4_issue_limit="$2"
      b4_llama_control_limit="$2"
      shift 2
      ;;
    --issue-limit)
      [[ $# -ge 2 ]] || { echo "--issue-limit requires a value" >&2; exit 2; }
      b4_issue_limit="$2"
      shift 2
      ;;
    --llama-control-limit)
      [[ $# -ge 2 ]] || { echo "--llama-control-limit requires a value" >&2; exit 2; }
      b4_llama_control_limit="$2"
      shift 2
      ;;
    --workers)
      [[ $# -ge 2 ]] || { echo "--workers requires a value" >&2; exit 2; }
      b4_workers="$2"
      shift 2
      ;;
    --run-id)
      [[ $# -ge 2 ]] || { echo "--run-id requires a value" >&2; exit 2; }
      b4_run_id="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "--output-root requires a value" >&2; exit 2; }
      b4_output_root="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$b4_issue_limit" =~ ^[0-9]+$ ]] && ((b4_issue_limit >= 1 && b4_issue_limit <= 273)) || {
  echo "--issue-limit must be an integer from 1 to 273" >&2
  exit 2
}
[[ "$b4_llama_control_limit" =~ ^[0-9]+$ ]] && ((b4_llama_control_limit >= 1 && b4_llama_control_limit <= 308)) || {
  echo "--llama-control-limit must be an integer from 1 to 308" >&2
  exit 2
}
[[ "$b4_workers" =~ ^[0-9]+$ ]] && ((b4_workers >= 1 && b4_workers <= 32)) || {
  echo "--workers must be an integer from 1 to 32" >&2
  exit 2
}
[[ "$b4_run_id" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "--run-id may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
}

b4_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$b4_repo_root"

if [[ "$(hostname -s)" != "nokogiri" ]]; then
  echo "Refusing to run generation outside Nokogiri" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Refusing to run with tracked or staged repository changes" >&2
  exit 1
fi

b4_branch="$(git branch --show-current)"
[[ "$b4_branch" == "revision/co-author-review" ]] || {
  echo "Expected branch revision/co-author-review; found $b4_branch" >&2
  exit 1
}

b4_manifest="phase3/analysis/outputs/tier_b/b4-issue-source-audit-v1/regeneration_sources.jsonl"
b4_manifest_sha="bdeb1d70b03c46d9f54e8775bb816d71183d486a5e55d9f76f992ab7938b49dc"
[[ -f "$b4_manifest" ]] || { echo "Missing audited issue manifest: $b4_manifest" >&2; exit 1; }
b4_observed_manifest_sha="$(sha256sum "$b4_manifest" | awk '{print $1}')"
[[ "$b4_observed_manifest_sha" == "$b4_manifest_sha" ]] || {
  echo "Audited issue manifest hash mismatch" >&2
  exit 1
}

run_python() {
  if [[ -n "${GENERATION_PYTHON:-}" ]]; then
    "$GENERATION_PYTHON" "$@"
  else
    pyenv exec python "$@"
  fi
}

if [[ -z "${GENERATION_PYTHON:-}" ]]; then
  command -v pyenv >/dev/null 2>&1 || {
    echo "pyenv is unavailable; set GENERATION_PYTHON to an absolute interpreter" >&2
    exit 1
  }
elif [[ ! -x "$GENERATION_PYTHON" ]]; then
  echo "GENERATION_PYTHON is not executable: $GENERATION_PYTHON" >&2
  exit 1
fi

b4_temp_output=""
if ((b4_dry_run)); then
  b4_temp_output="$(mktemp -d /tmp/b4-corrected-generation-dry-run.XXXXXX)"
  b4_output_dir="$b4_temp_output"
  trap 'if [[ -n "${b4_temp_output:-}" && -d "$b4_temp_output" ]]; then rm -rf -- "$b4_temp_output"; fi' EXIT
else
  for b4_key_name in OPENROUTER_API_KEY HF_TOKEN OPENAI_API_KEY; do
    [[ -n "${!b4_key_name:-}" ]] || {
      echo "Required credential is unset: $b4_key_name" >&2
      exit 1
    }
  done
  export HF_TOKEN OPENAI_API_KEY
  b4_output_dir="$b4_output_root/$b4_run_id"
  if [[ -e "$b4_output_dir/completion.json" ]]; then
    echo "Run is already complete; refusing to modify $b4_output_dir" >&2
    exit 1
  fi
fi

mkdir -p "$b4_output_dir"
exec > >(tee -a "$b4_output_dir/launcher.log") 2>&1

b4_llama_out="$b4_output_dir/meta-llama_llama-3.1-70b-instruct_${b4_run_id}.csv"
b4_qwen_out="$b4_output_dir/Qwen_Qwen2.5-72B-Instruct_${b4_run_id}.csv"
b4_gpt_out="$b4_output_dir/chatgpt_results_${b4_run_id}.csv"
b4_dry_args=()
if ((b4_dry_run)); then
  b4_dry_args+=(--dry-run)
fi
b4_llama_control_scope_args=()
if ((b4_llama_control_limit < 308)); then
  [[ "$b4_llama_control_limit" -eq "$b4_issue_limit" ]] || {
    echo "Partial Llama controls must use the same PR count as issue contexts" >&2
    exit 2
  }
  b4_llama_control_scope_args+=(--eligible-only)
fi

echo "B4 corrected generation launcher"
echo "Host: $(hostname -s)"
echo "Branch: $b4_branch"
echo "Commit: $(git rev-parse HEAD)"
echo "Run ID: $b4_run_id"
echo "Issue-context PR limit: $b4_issue_limit"
echo "Llama P1-P4 PR limit: $b4_llama_control_limit"
echo "Llama P4 file token budget: $b4_llama_p4_file_token_budget"
echo "Truncated Llama P4 completion token cap: $b4_llama_truncated_p4_completion_token_cap"
echo "Workers per generator: $b4_workers"
echo "Dry run: $b4_dry_run"
echo "Manifest SHA-256: $b4_manifest_sha"
echo "Output directory: $b4_output_dir"

run_llama() {
  OPENAI_API_KEY="${OPENROUTER_API_KEY:-}" run_python \
    phase3/generation/generate_openai.py \
    --base-url "https://openrouter.ai/api/v1" \
    --model "meta-llama/llama-3.1-70b-instruct" \
    --builder llama \
    --router openrouter \
    --provider coreweave \
    --quantization bf16 \
    --temperature 1 \
    --seed 42 \
    --truncate-p4 \
    --max-context-tokens "$b4_llama_p4_file_token_budget" \
    --truncated-max-tokens "$b4_llama_truncated_p4_completion_token_cap" \
    --variations P-1 P-2 P-3 P-4 \
    --limit "$b4_llama_control_limit" \
    --workers "$b4_workers" \
    --issue-manifest "$b4_manifest" \
    --out "$b4_llama_out" \
    "${b4_llama_control_scope_args[@]}" \
    "${b4_dry_args[@]}"
  OPENAI_API_KEY="${OPENROUTER_API_KEY:-}" run_python \
    phase3/generation/generate_openai.py \
    --base-url "https://openrouter.ai/api/v1" \
    --model "meta-llama/llama-3.1-70b-instruct" \
    --builder llama \
    --router openrouter \
    --provider coreweave \
    --quantization bf16 \
    --temperature 1 \
    --seed 42 \
    --variations P-5 P-6 \
    --limit "$b4_issue_limit" \
    --workers "$b4_workers" \
    --issue-manifest "$b4_manifest" \
    --resume "$b4_llama_out" \
    "${b4_dry_args[@]}"
}

run_qwen() {
  run_python \
    phase3/generation/generate_openai.py \
    --base-url "https://router.huggingface.co/v1" \
    --model "Qwen/Qwen2.5-72B-Instruct" \
    --builder qwen \
    --router huggingface \
    --provider novita \
    --quantization bf16 \
    --temperature 1 \
    --seed 42 \
    --variations P-5 P-6 \
    --limit "$b4_issue_limit" \
    --workers "$b4_workers" \
    --issue-manifest "$b4_manifest" \
    --out "$b4_qwen_out" \
    "${b4_dry_args[@]}"
}

run_gpt() {
  run_python \
    phase3/generation/generate_chatgpt.py \
    --model "gpt-5-mini-2025-08-07" \
    --variation P-5 \
    --seed 42 \
    --limit "$b4_issue_limit" \
    --workers "$b4_workers" \
    --issue-manifest "$b4_manifest" \
    --resume "$b4_gpt_out" \
    "${b4_dry_args[@]}"
  run_python \
    phase3/generation/generate_chatgpt.py \
    --model "gpt-5-mini-2025-08-07" \
    --variation P-6 \
    --seed 42 \
    --limit "$b4_issue_limit" \
    --workers "$b4_workers" \
    --issue-manifest "$b4_manifest" \
    --resume "$b4_gpt_out" \
    "${b4_dry_args[@]}"
}

echo "Starting Llama through OpenRouter/CoreWeave"
run_llama >>"$b4_output_dir/llama.log" 2>&1 &
b4_llama_pid=$!
echo "Starting Qwen through Hugging Face/Novita"
run_qwen >>"$b4_output_dir/qwen.log" 2>&1 &
b4_qwen_pid=$!
echo "Starting GPT-5-mini through OpenAI"
run_gpt >>"$b4_output_dir/gpt.log" 2>&1 &
b4_gpt_pid=$!

b4_failed=0
for b4_job in \
  "llama:$b4_llama_pid" \
  "qwen:$b4_qwen_pid" \
  "gpt:$b4_gpt_pid"; do
  b4_job_name="${b4_job%%:*}"
  b4_job_pid="${b4_job##*:}"
  if wait "$b4_job_pid"; then
    echo "$b4_job_name process exited successfully"
  else
    echo "$b4_job_name process failed; inspect $b4_output_dir/$b4_job_name.log" >&2
    b4_failed=1
  fi
done

((b4_failed == 0)) || exit 1

if ((b4_dry_run)); then
  echo "Dry run passed; no API calls were made and temporary outputs will be deleted"
  exit 0
fi

run_python phase3/generation/verify_b4_generation_outputs.py \
  --manifest "$b4_manifest" \
  --sampled-prs phase3/datasets/selected_prs_sampled.csv \
  --expected-issue-prs "$b4_issue_limit" \
  --expected-llama-control-prs "$b4_llama_control_limit" \
  --llama "$b4_llama_out" \
  --qwen "$b4_qwen_out" \
  --gpt "$b4_gpt_out" \
  --report "$b4_output_dir/completion.json"

echo "Corrected generation and fail-closed verification completed"
