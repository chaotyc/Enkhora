#!/usr/bin/env bash
# test_cluster.sh - validate the enkhora prototype on a real multi-node (GPU) cluster.
#
# Runs, in order, continuing past failures (each stage is PASS / FAIL / SKIP):
#   0 inventory   GPU model/memory/driver, IB/IP hints, torch/CUDA, git SHA + code hash on every node
#   1 fabric      anchor<->donor round trips over the project's transport: ping, exec-sized payloads
#                 (14 KiB/token x 1,8,32,64), real exec calls, bundle fetch bandwidth; per link and per GPU model
#   2 decode      real-model distributed decode vs an all-local reference (bitwise on one GPU model,
#                 |logit diff| <= --audit-atol across models), tok/s, per-layer p50/p95/p99, batch sweep
#   3 chaos       during ONE generation: set_kappa->0, SIGKILL a primary on a remote node, SIGSTOP/SIGCONT,
#                 planned leave, kill the control plane, kill every donor -> output must stay exact
#   4 colocate    experiment 3 (bench/colocate.py) on one GPU of each distinct GPU model
#   5 collect     every role's log in results/cluster-<timestamp>/, summary.md + summary.json, cleanup check
#
# Layout: node 0 = control plane + anchor (GPU 0); a donor on every other GPU (node 0's remaining GPUs too,
# unless --no-anchor-node-donors). Anti-affinity job id = one per node by default (--jobs node|gpu|single).
#
# SECURITY: every process listens on plain, UNAUTHENTICATED TCP ports (control 7400, anchor 7401,
# donors 7410+k, agents 7470+i). Anyone who can reach them can send commands or read activations.
# Run only inside your own allocation / a trusted private network, never on internet-facing addresses.
#
# ---------------------------------------------------------------------------------------------------
# Slurm (auto-detected from SLURM_JOB_ID; roles start as `srun --nodes=1 --ntasks=1 -w <node> --overlap`)
#
#   batch:
#     sbatch -N 3 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:4 --mem=200G --time=02:00:00 \
#            [-p <partition>] [-A <account>] -o cluster-%j.out \
#            --wrap 'scripts/test_cluster.sh --model allenai/OLMoE-1B-7B-0924'
#   interactive:
#     salloc -N 2 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:4 --mem=200G --time=01:00:00
#     scripts/test_cluster.sh --model tiny                       # fast smoke test first (~2 min)
#     scripts/test_cluster.sh                                     # OLMoE-1B-7B (default model)
#     scripts/test_cluster.sh --model Qwen/Qwen3-30B-A3B --bundle-size 16 --master-dir /local/scratch/m
#   If steps see no GPUs, add --srun-args '--gres=gpu:4'. The repo and python must be on a shared
#   filesystem (same path on every node).
#
# SSH (plain Linux nodes, no scheduler): one host per line, first line = node 0 (anchor)
#     $ cat hosts.txt
#     ubuntu@10.0.0.11  slots=8              # slots = GPUs to use (default: all that nvidia-smi shows)
#     ubuntu@10.0.0.12  slots=8  addr=10.1.0.12
#     ubuntu@10.0.0.13                       # optional: addr=<IP peers should use>, job=<anti-affinity id>
#     scripts/test_cluster.sh --hosts hosts.txt --iface ib0              # repo+env already on every host
#     scripts/test_cluster.sh --hosts hosts.txt --sync                   # rsync repo + install .venv first
#   Needs passwordless ssh from this machine to every host, and node 0 must reach every node's ports.
#
# Local dry run (this machine pretends to be 3 nodes x 2 GPUs; CPU only; same launcher code path):
#     scripts/test_cluster.sh --launcher local --model tiny
#     scripts/test_cluster.sh --launcher local --model tiny-olmoe --local-nodes 2 --local-gpus 3
#
# Show the exact commands without launching anything (fake node list / hostfile, fake GPU counts):
#     scripts/test_cluster.sh --launcher slurm --nodelist 'gpu[01-03]' --fake-gpus 4 --print-plan
#     scripts/test_cluster.sh --launcher ssh --hosts hosts.txt --iface ib0 --print-plan
#
# Common options (all others: scripts/test_cluster.sh --help-all):
#   --python PATH | ENKHORA_PYTHON=PATH   interpreter to run (default: <repo>/.venv/bin/python, else python3)
#   --model tiny|tiny-olmoe|<HF id>       --bundle-size N    --dtype bfloat16    --master-dir DIR
#   --iface ib0   --bind-host 0.0.0.0   --control-port/--anchor-port/--donor-port/--agent-port
#   --stages 0,1,2,3,4,5   --batches 1,8,32,64   --max-new 32   --chaos-max-new 64   --audit-atol 0.1
#   --jobs node|gpu|single   --no-anchor-node-donors   --gpus-per-node N   --out DIR
#
# Exit code: 0 all stages PASS/SKIP, 3 a stage failed, 130 interrupted. Cleanup (kill every launched
# process on every node) runs on success, failure, Ctrl-C and SIGTERM; every process also has a hard
# TTL (--ttl) so nothing outlives the test even if this machine loses its connection.
# END USAGE
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ENKHORA_PYTHON:-}"
OUT=""
PLAN_ONLY=0
args=()

usage() { sed -n '2,/^# END USAGE/p' "${BASH_SOURCE[0]}" | grep -v '^# END USAGE' | sed 's/^# \{0,1\}//'; }

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --help-all) HELP_ALL=1; shift ;;
    --python) PY="$2"; shift 2 ;;
    --python=*) PY="${1#*=}"; shift ;;
    --out) OUT="$2"; shift 2 ;;
    --out=*) OUT="${1#*=}"; shift ;;
    --print-plan) PLAN_ONLY=1; args+=("$1"); shift ;;
    *) args+=("$1"); shift ;;
  esac
done

if [[ -z "$PY" ]]; then
  if [[ -x "$REPO/.venv/bin/python" ]]; then PY="$REPO/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if [[ -n "${HELP_ALL:-}" ]]; then exec "$PY" "$REPO/scripts/cluster.py" run --help; fi

if [[ -z "$OUT" ]]; then OUT="$REPO/results/cluster-$(date +%Y%m%d-%H%M%S)"; fi
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
args+=(--out "$OUT")

if ((PLAN_ONLY)); then exec "$PY" "$REPO/scripts/cluster.py" run "${args[@]}"; fi

mkdir -p "$OUT"
rm -f "$OUT/.cleanup_done"
on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ -e "$OUT/launch.json" && ! -e "$OUT/.cleanup_done" ]]; then
    echo "test_cluster.sh: orchestrator did not finish its cleanup; cleaning up now" >&2
    "$PY" "$REPO/scripts/cluster.py" cleanup "$OUT" || true
  fi
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT TERM HUP

set +e
"$PY" "$REPO/scripts/cluster.py" run "${args[@]}"
rc=$?
set -e
echo "results: $OUT/summary.md"
exit "$rc"
