# Interstitial v1 prototype

A working prototype of *Opportunistic MoE Inference: Architecture v0*
([design doc](Interstitial%20Inference%20Algorithm.md)). The dense trunk and all
per-request state stay on an anchor. Donor processes hold bundles of expert weights
and execute them as a pure function. Activations travel; weights never move on the
decode path.

It runs end to end on a laptop CPU, using a small random-weight Qwen3-MoE with the
real HF architecture. The same code targets Qwen3-30B-A3B or OLMoE on GPUs.

## What it demonstrates today (CPU, this repo)

| Claim from the design | How it is checked | Result |
|---|---|---|
| Split inference is exact | Distributed decode vs an all-local run with the same kernel contract | Logits **bitwise equal** |
| Donor loss costs latency, never correctness | SIGKILL, SIGSTOP (hung donor), host memory spike and control-plane death, all during one generation | Bitwise equal; hedges, circuit breaker and fallback logged |
| Every donor gone = slow, not wrong | Kill all donors mid-run | Bitwise equal (anchor floor) |
| κ shrink is a routing-table edit | `set_kappa` on the primary donor | Primaries swap, **0 bytes moved** |
| Planned drain | `leave` | Re-replication first, then exit; no fallback calls |
| Silent errors are caught | Donor that adds 1e-3 to outputs, audit rate 1.0 | Detected, quarantined, output still exact |
| Watchdog protects the host | Simulated reclaim spike test, 1000 trials | 0 host failures when the burst between polls fits inside the guard |
| Planner and predictor | Replay of a 14-day synthetic trace | Coverage, exposure, GiB re-replicated per hour; oracle vs predictor |

"Bitwise" works because of the numeric contract in `interstitial/experts.py`:

- The dispatcher sends a whole bundle to one replica per layer step, so each expert's matmul input is identical on every replica.
- Partials are reduced in a fixed order: expert id within a bundle, then bundle id at the anchor.
- `chunk_rows`, the kernel-length cap, is the same everywhere.

On GPUs this holds within one device type and software stack. Use `audit_atol > 0` across GPU generations.

## Quickstart

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
```

```bash
.venv/bin/python -m pytest -q
```

```bash
.venv/bin/python -m interstitial demo
```

```bash
.venv/bin/python -m interstitial sim --days 14
```

The test suite has 37 tests and runs in about 25 s. The e2e tests spawn real donor processes and SIGKILL or SIGSTOP them.

### Separate processes (one machine or many)

```bash
python -m interstitial control --port 7400 --expert-demand 1.0
python -m interstitial donor --id gpu-a --job 9001 --control CTRL:7400 --kappa 1.0 --mem-mib 8 --life-s 3600
python -m interstitial donor --id gpu-b --job 9002 --control CTRL:7400 --kappa 0.0 --mem-mib 8 --life-s 72000
python -m interstitial anchor --model tiny --master ./master --control CTRL:7400 --batch 8 --max-new 32
python -m interstitial ctl status --control CTRL:7400
python -m interstitial ctl set_kappa gpu-a 0.0 --control CTRL:7400
python -m interstitial ctl host_alloc gpu-b 6 --control CTRL:7400    # simulated host grows by 6 MiB
python -m interstitial ctl leave gpu-a --control CTRL:7400
```

On GPUs, change these flags:

- Donors: add `--device cuda:N --nvml N --guard-mib 12288 --advertise-host <ip>`.
- Anchor: use `--model Qwen/Qwen3-30B-A3B --dtype bfloat16 --device cuda:0 --bundle-size 16`.
- Export the master store once: `python -m interstitial export --model … --out ./master --bundle-size 16`.

## Layout

| Path | Design section | What it is |
|---|---|---|
| `interstitial/experts.py` | §3 | Bundle kernel, `build_route`, fixed-order merge: the numeric contract |
| `interstitial/store.py` | §4 | Master store (safetensors per bundle, sha256 manifest) |
| `interstitial/model/hf.py` | §4 | `RemoteMoE` swap-in for HF v5 Qwen3-MoE / OLMoE blocks; tiny test model; export |
| `interstitial/anchor/dispatcher.py` | §6, §7 | Per-layer scatter/gather, per-donor deadlines, hedging, circuit breaker, epochs between layers, audit |
| `interstitial/anchor/fallback.py` | §7 | Ladder rungs 3–5 (device cache → RAM → disk); `LocalExecutor` reference |
| `interstitial/anchor/anchor.py` | §4 | Anchor process: trunk, master-store server, control link |
| `interstitial/control/planner.py` | §5 | Pure greedy planner: memory, κ, job anti-affinity, admission, hysteresis, store-only sources |
| `interstitial/control/server.py` | §4, §7 | Registry, replans on events, routing epochs, drain-then-drop, planned leave, quarantine |
| `interstitial/donor/executor.py` | §4, §8 | Donor process: exec/fetch, throttled checksummed loads, status |
| `interstitial/donor/slabs.py`, `watchdog.py` | §8 ring 2 | Slab lifecycle; NVML or simulated probe; standbys-first eviction |
| `interstitial/donor/governor.py` | §8 | Token bucket (burst bounded), AIMD, on/off-window slowdown estimator |
| `interstitial/sim/` | §9, §11 exp. 1 | Trace schema and loader, synthetic trace, empirical survival, replay |
| `bench/colocate.py` | §11 exp. 3 | Host slowdown vs κ (GPU), plus idle-second-context cost |
| `bench/spike.py` | §11 exp. 4 | Reclaim spike test (GPU via NVML, or simulated) |
| `interstitial/cluster.py`, `demo.py` | §11 exp. 5 | One-machine cluster harness and chaos demo |

## Validation experiments: status

| # | Experiment | Status |
|---|---|---|
| 1 | Trace-driven simulator | **Runs.** Needs the Quest reconstruction converted to the CSV schema in `sim/trace.py`. The bundled synthetic trace is calibrated only to the retrospective's donor counts and is not evidence |
| 2 | Routing traces | Not started. Bundles are contiguous expert ranges until traces justify clustering |
| 3 | Co-location micro-benchmark | **Harness ready**, needs one GPU: `python bench/colocate.py --device cuda:0 --host-cmd '<host code>'` |
| 4 | Reclaim spike test | **Simulated: pass.** GPU mode ready (`--device cuda:0`), not run |
| 5 | Split inference with kills | **Passes on one machine over TCP** (tests + demo). Needs two GPU nodes for real per-layer RTT and default g |
| 6 | Fabric sweep | Not started (needs site access) |

Spike-test finding to carry into the design: the residual risk is wider than "one allocation larger than the guard". It is **host allocations that total more than the guard within one poll interval**. Going from a 20 ms to a 5 ms poll cut those failures by about half in simulation. Size the guard and poll interval from the measured *burst* distribution, not the step distribution.

## Deliberate v1 simplifications

- **Transport.** TCP via asyncio with a JSON-header frame format; there is no pickle on the wire. A UCX/RDMA transport would implement the same `RpcClient.request` interface. NCCL is avoided because losing one peer aborts the whole communicator.
- **Node agent.** The watchdog and governor run inside the donor process, not in a separate privileged per-node agent.
- **Slabs.** Each slab is a plain tensor allocation, followed by `empty_cache` on drop. Fixed-size chunked slabs are a v2 item.
- **Survival.** The live control plane takes survival from each donor's `--life-s` and `--limit-s`. The empirical predictor runs only in the simulator. The Slurm adapter (prolog/epilog hooks) is not built.
- **Prefill.** Prefill uses the same remote path. `DispatchConfig.prefill_local_tokens` keeps it on the anchor instead.
- **Batching.** Static batches through HF `generate`. No continuous batching, no gateway, one anchor.
- **Placement and routing.** A bundle without a compute-backed primary still lists its store replicas. Their budget refusals steer the dispatcher to the anchor after one round trip.
- **Model loading.** Loading a real model materializes its experts once before `patch_model` drops them, so the peak RAM is the full model.
- **Untested on hardware.** NVML probes, `--device cuda`, and the AIMD loop against real counters are written but untested here, because there is no GPU on this machine.
