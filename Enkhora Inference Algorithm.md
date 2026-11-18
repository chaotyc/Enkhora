# Enkhora — Opportunistic MoE Inference: Architecture v0 (from scratch)

Nov 13, 2026 · @tc

## Summary

Keep the model's dense trunk and every piece of per-request state on one or two dedicated anchor GPUs. Treat every other GPU as a stateless expert executor that holds a slice of expert weights in spare VRAM. During decode, activations travel to the experts; weights never move.

The design rests on one asymmetry. In a DeepSeek-V3-class model, routed experts are about 654B of 671B parameters (97%), yet each token uses 8 of 256 per layer ([technical report](https://arxiv.org/abs/2412.19437)). One expert is 44 MB at 8 bits. The activation it needs is 14 KB. Sending the question to the shelf is about 3,000 times cheaper than carrying the book to the desk.

The Quest data shows the same asymmetry on the supply side: 55.7% of paired GPU-hours had peak VRAM under 25%, but only 20.6% had mean compute under 25%. Spare memory is plentiful and spare compute is not. Experts are the part of the model that needs a lot of the first and little of the second.

| Decision | Choice | Reason |
| --- | --- | --- |
| What moves on the wire | Activations (14 KB per token per layer), not weights | 3,000x smaller; makes cross-node operation practical |
| Where state lives | KV cache, sampler and sequence state only on anchors | Donor loss can cost latency but can never corrupt a sequence |
| Source of truth for weights | Full master copy on anchor-node NVMe | Reclaim is a drop, never a save; no donor is required for correctness |
| Placement unit | A bundle of co-routed experts (about 1.4 GB), packed layer by layer | Fits fragmented headroom; keeps each layer on few donors |
| How a donor is described | Three numbers: memory budget, compute budget, survival curve | One planner covers busy hosts, quiet hosts and fully idle GPUs |
| Co-location mechanism | A separate process and CUDA context per donor GPU, time-sliced, never MPS | Keeps memory and fault domains apart from the host job |
| Binding constraint | Host compute interference, not memory | Quest memory covers the expert tier 1.3x at P10 and 2.1x at median |

**Biggest open number:** how much a host job slows per millisecond of expert kernel time injected into its GPU. Section 11 makes that the first experiment. Every throughput figure below is an estimate until it is measured.

## What the Quest data forces

Eight observations from the 30-day retrospective each remove a design option. The table reads left to right: what was measured, what it rules out, what the design does instead.

| Quest observation | What it rules out | Design consequence |
| --- | --- | --- |
| 55.7% of paired GPU-hours have peak VRAM under 25%; only 20.6% have mean compute under 25%; 13.0% have both | Treating a donor as a spare GPU | Donors mostly lend memory. Compute is budgeted per donor and spent where hosts are quiet |
| Median 36 qualifying GPUs at once, spread over about 16 nodes (about 2 per node) | Any design that pools memory only inside a node | The data path is cross-node from day one. Node-local hops are a bonus, not an assumption |
| 27 of the median 36 have at least 4 hours left, but that count uses future knowledge | A static placement | A lifetime predictor ranks donors. Roughly a quarter of candidates are short-lived at any instant |
| P10 pooled headroom is 830 GiB after the 12 GiB guard; the expert tier of a 671B model is 609 GiB at 8 bits | Full mirroring at all times | Replication is partial in bad hours. The fallback must be dedicated, not another donor |
| Peak VRAM is a sampled high-water mark | Sizing headroom from accounting data | Accounting only nominates donors. A live watchdog on the node decides and reclaims |
| Utilization is kernel-active time; zeros may be missing telemetry | Reading spare compute from jobstats | Each donor's compute budget is learned online, starting near zero |
| Three GPU and fabric generations (A100 40 GB on EDR, A100 80 GB on HDR, H100 on NDR) | One latency constant, one weight format | Round-trip time is measured per link. Weights are stored at 8 bits and cast to BF16 where FP8 kernels are missing |
| 131 configured GPUs, 106.8 allocated on average | Ignoring unallocated GPUs | An idle GPU is a donor with a full compute budget and a short, uncertain life. Same planner, no special case |

Two caveats carry through the whole design. The pooled-memory figures assume separate allocations can be combined, which is not yet proven. And the unallocated count is an accounting average, not a verified live availability rate.

## Core idea: an anchored trunk and a scavenged expert tier

Split the model along the line where memory and compute separate. The trunk is small, compute-heavy and stateful. The routed experts are huge, compute-light and stateless.

**Analogy.** Picture a library with one reading desk and books shelved in other departments' empty shelf space. A clerk stands at each shelf. The reader never carries books to the desk. She sends a slip to the clerk, who reads the right pages and sends back an answer. If a department reclaims its shelf, the clerk leaves and the slip goes to another shelf holding a copy, or to the basement archive.

| Part | Size in the 671B example | Character | Lives on |
| --- | --- | --- | --- |
| Trunk: embeddings, attention, norms, router gates, shared experts, the three dense layers | About 17B parameters, 17 GB at 8 bits | Compute-heavy; owns the KV cache | Anchor GPUs |
| Routed experts: 58 MoE layers x 256 experts | About 654B parameters, 609 GiB at 8 bits | Memory-heavy; a token touches 8 of 256 per layer | Donor GPUs, with a master copy on anchor NVMe |

### Expert execution is a pure function

For a token with hidden state h at layer l, a donor holding expert set D computes one partial sum and returns it:

```latex
y_D(h) = \sum_{e \,\in\, \mathrm{TopK}(h) \,\cap\, D} g_e(h)\, E_e(h)
```

The anchor adds the partial sums, the shared expert and the residual. The donor keeps nothing between calls. That makes every call idempotent, so the anchor may retry it, send it twice, or abandon it without any risk to the sequence.

### The placement unit is a bundle

DeepSeek-V3's router already groups experts. Each layer has 8 groups of 32, and a token draws its 8 experts from at most 4 groups ([config](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/b3480c8cccc78b782bb7623abb79cc4d104831ad/inference/configs/config_671B.json), [report](https://arxiv.org/abs/2412.19437)). A bundle is one layer's group: 32 experts, 1.41 GB at 8 bits. The model has 464 bundles, and a full layer is 8 bundles, or 11.3 GB.

A bundle fits the smallest donor slot in the screen (20 GiB spare less the 12 GiB guard). Models without routing groups get bundles from co-activation clustering instead.

### Every donor is three numbers

- **M, memory budget:** capacity, minus the host's live usage, minus the guard.
- **κ, compute budget:** the fraction of wall time our kernels may occupy that GPU.
- **S(t), survival curve:** the probability the donor is still there t minutes from now.

| Role | Typical GPU | κ | What it does |
| --- | --- | --- | --- |
| Anchor | Dedicated Slurm job owned by the service | 1 | Trunk, KV cache, gating, dispatch, expert cache, last-resort execution |
| Execute donor | Unallocated GPU held as a preemptible job, or a host with little kernel-active time | 0.1 to 1 | Holds primary bundles and runs them |
| Store donor | Busy host with spare VRAM | Near 0 | Holds standby replicas; serves failover and re-replication |

Roles are not assigned. They fall out of κ, and a donor drifts between execute and store as its host's behaviour changes.

### Do not count on a hot set

Published traces show expert use is sparse and skewed within one request but evens out across many requests ([MoE-Infinity](https://arxiv.org/html/2401.14361v2)). A multi-user service sees the aggregate. So the design assumes flat traffic: the compute a donor absorbs is proportional to the experts it executes. Per-request skew is used in one place only, the anchor's fallback cache.

## System architecture

Three planes, with one rule between them: the control plane is never on the per-token path. If it dies, decode continues on the last routing table and only placement freezes.

&#91;embedded content: architecture · anchor, donors, control plane\]

Solid arrows are the per-token path: hidden rows out, partial sums back. The dashed arrow is used only when a primary is lost or redundancy is being rebuilt. Arrows from the control plane carry placement, never tokens.

### Deployment view

&#91;embedded content: deployment view · roles of GPUs across three fabric islands\]

The anchor sits on the fastest island, and donors span all three GPU generations, so the data path crosses nodes from the start. The 4-GPU job on Quest 12 is one failure domain, so no bundle keeps both of its replicas there.

### Inside one donor GPU

&#91;embedded content: inside one donor GPU · VRAM and GPU time\]

Two separate loops in the node agent protect the host. The watchdog frees slabs when the host grows into the guard. The governor keeps expert kernels short and inside the κ budget.

### Anchor data plane (dedicated, one node, one or two GPUs)

- **Trunk engine.** Attention, KV cache, sampling. A fork or plugin of an existing serving engine.
- **Gate and dispatcher.** Computes each token's top-k experts, picks one replica per bundle, sends one message per donor per layer, and merges the partial sums.
- **Expert cache.** Anchor VRAM left after the trunk and KV cache. Holds bundles pulled in for fallback.
- **Master store.** The full expert tier on local NVMe, page-cached in host RAM. This is the only copy the system depends on.

### Node agent (one per donor node, privileged)

- **Slab manager.** Allocates donor VRAM in fixed chunks, loads bundles into them, frees them on command.
- **Executor.** One OS process and one CUDA context per donor GPU. Runs bundle kernels and returns partial sums. A crash takes down one executor, never the host job.
- **Memory watchdog and compute governor.** Local control loops that protect the host (section 8). They act without asking the control plane.

The agent runs on cores and RAM that Slurm reserves for the system, not on the job's allocation. Because the master copy lives on the anchor, a donor node needs about 1 GB of pinned RAM for message buffers, not hundreds.

### Control plane (service node, no GPU)

- **Slurm adapter.** Reads job starts, ends, time limits and GPU bindings from accounting and from prolog and epilog hooks.
- **Donor registry.** The live (M, κ, S) for every GPU, fed by the node agents.
- **Lifetime predictor.** Turns job metadata into a survival curve (section 9).
- **Placement planner.** Decides which bundle replicas go where (section 5).
- **Routing table.** A versioned map from bundle to an ordered replica list, pushed to anchors.

### Transport

RDMA over the cluster's InfiniBand, with one persistent connection per anchor and donor GPU. Version 0 lands messages in pinned host memory and copies to the GPU. GPUDirect RDMA is an optimization to add only if the hop budget in section 6 needs it.

## Expert placement algorithm

Placement is an incremental greedy assignment that runs on events, not a global solve. It answers three questions per bundle: how many copies, on which donors, and which copy executes.

### Inputs

- **Bundles:** size (1.41 GB each in the example) and measured heat. Heat defaults to uniform until traces say otherwise.
- **Donors:** M, κ, S(t), plus node, Slurm job ID, fabric generation and measured round-trip time to the anchor.

### Hard constraints

1. **Memory.** Bundles on a donor fit inside M.
2. **Compute.** Expected kernel time of the bundles a donor executes stays inside κ.
3. **Anti-affinity by job.** Two replicas of a bundle never sit inside the same Slurm job. A four-GPU job that ends takes all four GPUs at once, so the failure domain is the job, not the GPU.
4. **Zero replicas is legal.** The master copy always exists, so a bundle with no donor copy is slow, not lost.

### Priorities, in order

1. **Coverage:** every bundle has a primary on a donor with compute to spare.
2. **Redundancy where it is needed:** surplus memory buys standbys for the bundles whose primary is most likely to vanish.
3. **Locality:** each layer sits on few donors, close to the anchor.

### Steps

1. **Rank donors twice.** Once by execute-worthiness (high κ, low round-trip time), once by stability (expected remaining life).
2. **Place primaries layer by layer.** For each layer, take the g best execute donors with room and split the layer's 8 bundles across them. Deduct memory and compute.
3. **Place standbys.** Walk bundles from most fragile primary to least. Give each a second replica in a different job, on the most stable donor with room. Store donors absorb most of these.
4. **Spend leftovers.** In good hours, add third replicas on execute donors for load spreading and hedging.
5. **Publish** a new routing-table epoch.

**The spread factor g** trades speed against stragglers. At large batch, a layer's time is set by reading its weights. Splitting the layer across g donors runs those reads in parallel, but the anchor then waits for the slowest of g replies. Start at g = 2 and tune from measurement.

**Pair fragile with stable.** An unallocated GPU is the best executor and the least reliable donor, since the scheduler can hand it to a queued job at any minute. A busy host with a 40-hour job is the reverse. The planner deliberately pairs them: primary on the fast fragile donor, standby on the slow stable one.

### Events

| Event | Planner response | Bytes moved |
| --- | --- | --- |
| Donor joins | Fill it with standbys for the least-protected bundles; promote to primary only past a hysteresis margin | One donor's worth |
| Host wakes up and κ shrinks | Swap roles: the standby elsewhere becomes primary | None |
| Host grows and M shrinks | Drop standbys on that donor first, then primaries | None |
| Predicted job end approaches | Promote standbys early; rebuild redundancy from store donors | The lost replicas |
| Donor vanishes without warning | Data plane has already failed over; planner backfills | The lost replicas |

The second row is the cheapest lever in the system. Both replicas hold identical bytes, so moving execution off a host that woke up is a routing-table edit.

Copies are throttled per donor and sourced from store donors or the master, never from a donor under compute pressure.

### Example: a median Quest hour

The retrospective gives 27 donors with at least 4 hours left and 1,282 GiB between them, about 47 GiB each. Suppose 11 are quiet enough to execute. That figure is the both-under-25% share: 13.0% of 62,827 paired GPU-hours over 720 hours.

- The 11 execute donors hold about 517 GiB of primaries, or 85% of the 609 GiB expert tier.
- The other 15%, about 9 layers, get primaries on unallocated GPUs if any exist, else on the anchor's expert cache.
- The 16 store donors hold about 752 GiB, enough for a standby of every bundle with room left over.

## Decode path and routing

One decode step is 58 scatter-gather rounds, one per MoE layer, each with a single synchronization barrier. Batching across sequences makes each round carry many tokens, so the number of barriers per step never grows with load.

### One layer, step by step

1. **Attention.** The anchor runs attention for all B tokens in the batch against its local KV cache.
2. **Gate.** The anchor computes each token's top-8 experts and their weights. The gate matrix is 256 x 7168, so this is negligible.
3. **Choose replicas.** The dispatcher maps each (token, bundle) pair to a replica from the routing table: the primary if it is healthy and inside its budget, otherwise the next in the list.
4. **Scatter.** One message per donor: hidden rows, expert indices, gate weights. A token that needs several bundles on one donor sends its hidden row once.
5. **Execute.** Each donor groups tokens by expert, runs batched matmuls, sums per token, and returns one vector per token. Meanwhile the anchor runs the shared expert and any bundles in its own cache.
6. **Gather.** The anchor adds the partial sums and the residual, then moves to the next layer.
7. **Hedge.** If a reply misses its deadline, the anchor reissues those pairs to the next replica or its own cache. The first answer wins and the late one is discarded.

Step 7 is safe only because of the pure-function property in section 3. The deadline per donor comes from its recent latency percentiles, not a global constant.

### The wire is not the bottleneck

A token costs 14.3 KB each way per donor contacted per layer. With each layer on one donor that is 1.7 MB per decoded token for the whole model, and at most 4 times that when a token's routing groups sit on different donors. Even the EDR island at 100 Gb/s carries thousands of tokens per second.

What matters is hop latency and its tail, because 58 hops run in series:

```latex
T_{\text{step}} = \sum_{l=1}^{58} \Big[\, t_{\text{attn}}(B) + \max_{d \in D_l} \big( \rho_d + w_d + t_{\text{exec},d}(B) \big) \Big]
```

Here ρ is network round trip, w is the wait for a GPU time slice on a shared donor, and t\_exec is kernel time. The max over each layer's donors is why the spread factor g and hedging exist.

### Where GPU time goes at decode

Expert kernels at decode are limited by reading weights, not by arithmetic. Every distinct expert touched in a layer streams its 44 MB once, however many tokens share it. The table assumes uniform independent routing and 1.5 TB/s of memory bandwidth, an A100-class figure. Treat it as approximate until measured.

| Batch B (tokens per step) | Distinct experts touched per layer | Expert GPU-time per layer | Expert GPU-time per decoded token |
| --- | --- | --- | --- |
| 1 | 8 | 0.2 ms | 14 ms |
| 32 | 163 | 4.8 ms | 8.7 ms |
| 64 | 222 | 6.5 ms | 5.9 ms |
| 256 | 256 | 7.5 ms | 1.7 ms |

Two things follow. Small batches are quick per stream, and large batches are cheap per GPU-second. And this is a throughput tier: at B = 64 with g = 2, a step takes roughly a quarter of a second, about 4 tokens per second per stream and a few hundred in aggregate.

### Batching and prefill policy

- **Decode batches** are sized per workload class so the step time meets that class's target.
- **Prefill** uses the same path with a prompt's tokens as the batch. It is arithmetic-bound and efficient per token, but it is the largest single injection of compute. It goes to donors with κ = 1 and to the anchor first, and it can wait.
- **Kernel-length cap.** Every donor call is sized so no single kernel runs longer than a set bound. We never sit on a host's GPU for long.
- **Feedback in the reply.** Each reply carries the donor's wait time and remaining budget. The dispatcher shifts load before a donor has to refuse.
- **Data class.** Activations are derived from prompts. For sensitive classes the dispatcher uses only the anchor and service-owned GPUs.

## Elasticity and correctness

Correctness never depends on a donor. The brief's warning, that every required expert must be reachable at every decode step, is met by construction: the last rung of the fallback ladder is dedicated hardware holding a full copy.

### The invariant

At every layer of every decode step, each bundle resolves to at least one executor: a ready donor replica, or the anchor.

Three properties make donor loss a latency event and nothing more:

- Donors hold only immutable, public weights.
- No sequence state ever leaves the anchor.
- Every donor call is idempotent.

So failure handling is "ask someone else". There is no checkpoint, rollback or consensus protocol anywhere in the design.

### Fallback ladder

1. **Primary donor replica.** Executes in place.
2. **Standby donor replica** in a different Slurm job. Executes in place.
3. **Anchor expert cache.** The bundle is already in anchor VRAM.
4. **Anchor host RAM.** Copy the expert to the anchor GPU over PCIe, a few milliseconds each.
5. **Anchor NVMe master.** Read, copy, execute. Tens of milliseconds the first time.

Rungs 3 to 5 reverse the data flow: weights come to the anchor. That is slower, but it needs nothing from the rest of the cluster. Because one request reuses a small set of experts, the anchor cache prefetches for the sequences it is serving.

**The floor.** With every donor gone, the system is an ordinary offloading MoE server running on its anchors: correct and slow. When the fallback rate crosses a threshold, the gateway shrinks batches and sheds or reroutes low-priority work, instead of letting queues grow.

### Replica lifecycle

1. **Loading.** Slab allocated, bytes arriving. Not routable.
2. **Ready.** Checksum verified. Present in the routing table.
3. **Draining.** Absent from new epochs. Finishes in-flight calls, which the kernel-length cap bounds to milliseconds.
4. **Gone.** Slab freed.

A hard eviction jumps from ready to gone. In-flight calls get a refusal or no answer, and the hedge in section 6 covers both.

### What triggers an eviction

| Trigger | Detected by | Time allowed | Path |
| --- | --- | --- | --- |
| Host memory pressure | Node watchdog | Milliseconds | Hard evict, standbys first |
| Host slowdown above budget | Node governor | Seconds | Shrink κ and swap roles; evict only if it persists |
| Job end or time limit approaching | Slurm adapter and predictor | Minutes ahead | Planned drain |
| Job cancelled or crashed | Epilog hook | Before the next job's prolog | Hard evict; the executor must be gone before the GPU is reassigned |
| GPU error on a donor | Node agent | Immediate | Hard evict and quarantine that GPU |
| Kill switch | Operator, per job, node or site | Seconds | Hard evict |

### Routing epochs

The routing table is versioned. Anchors swap to a new epoch between layers, never inside one. Every message carries its epoch, and a donor that no longer holds a bundle answers with a refusal. A stale table is therefore safe: it costs one retry inside the same layer.

### Silent errors

Retries cannot catch a donor that returns a wrong answer. Two guards cover that. Bundles are checksummed at load and rescanned periodically. And the anchor recomputes a small random sample of calls on a second replica, compares within a tolerance, and quarantines a donor that disagrees.

## Host-job protection

The host job is the tenant and we are the guest. Two independent control loops on the donor node enforce that, one for memory and one for compute. Both run locally, and both fail toward the host.

### Co-location mechanism

Each donor GPU gets one executor process with its own CUDA context, alongside the host job's context.

- **Why it can work.** In DEFAULT compute mode several processes may use a GPU, and without MPS their kernels are serialized rather than overlapped ([NVIDIA MPS guide](https://docs.nvidia.com/deploy/mps/615/when-to-use-mps.html)).
- **Why we must throttle ourselves.** NVIDIA's default time-slicing shares GPU runtime evenly among active processes ([Run:ai docs](https://run-ai-docs.nvidia.com/self-hosted/platform-management/runai-scheduler/resource-optimization/time-slicing)). The driver will not favour the host. Our budget has to.
- **Why not MPS.** Only one user can have an active MPS server, and a fatal fault from one client is reported to every client on that GPU (same guide). That puts our bugs in the host's fault domain.
- **What a site must allow.** Donor GPUs not set to EXCLUSIVE\_PROCESS, and a privileged node agent outside the job's cgroup. A site that refuses either can still lend unallocated GPUs.

### Memory: three rings

**Ring 1, admission.** A GPU becomes a donor only after its job has passed a warm-up period and its memory has plateaued. Jobs that pre-allocate most of the device never qualify. The guard is a fixed floor (12 GiB in the retrospective) plus the largest between-sample jump seen for that job and for that account's history.

**Ring 2, watchdog.** The node agent polls per-process GPU memory every few tens of milliseconds. When free memory falls below a low watermark it frees slabs at once, standbys first, until a high watermark is restored. A free needs no save and no network call. Slabs are fixed-size chunks, so release is a handful of calls.

**Ring 3, interception (optional, per site).** A preloaded shim in the job's environment catches a failed allocation, signals the agent, and retries. It changes the job's environment, not its code.

**What is left over.** Without ring 3, one allocation larger than the guard, made inside one poll interval, can still fail. Say this plainly to directors; it was Clemson's first question. The mitigation is data: measure the step-size distribution of host allocations in pilot telemetry and set the guard so the residual rate is one the center accepts.

### Compute: budget, cap, feedback

1. **Token bucket.** Each donor GPU's κ is spent in kernel-milliseconds. An empty bucket means the executor refuses the call and the dispatcher goes elsewhere.
2. **Kernel-length cap.** No single kernel runs longer than a set bound, so the host never waits more than the cap plus one time slice.
3. **AIMD on κ.** Start near zero. Raise κ additively while the host shows no slowdown; cut it multiplicatively when it does. This is the same shape as TCP congestion control, with the host's slowdown playing the role of packet loss.
4. **Slowdown signal without a baseline.** The governor alternates short, randomized windows with lending allowed and paused. It compares the host process's own activity across them: per-process kernel-active time, power draw, memory-bandwidth counters. Less host work per second in the "on" windows, beyond a tolerance, cuts κ.
5. **No host, no governor.** An unallocated GPU runs at κ = 1.

The guardrail is the plan's 5% per-job slowdown cap. The site-level metric stays host GPU-hours lost per million tokens.

### Hygiene

- **CPU and RAM.** The agent is pinned to system-reserved cores. Donor nodes carry no weight copies in host RAM.
- **Faults.** On any CUDA error the executor exits and releases everything. It does not restart on that GPU within the same job.
- **Residue.** Activation buffers live in a fixed arena that is zeroed before release. Weights are public.
- **Power.** Extra load can push a node into power capping. Throttle reasons are read as slowdown.

## Lifetime prediction and admission

The retrospective's "at least 4 hours remaining" filter is an oracle. Its online replacement is a survival curve per job, and the design ensures the predictor's mistakes cost bandwidth, never correctness.

### What is predicted

S(t) is the probability that a job still holds its GPU t minutes from now, given how long it has already run.

- **Features, all from Slurm:** requested time limit, elapsed time, partition and QoS, GPU count, job-array membership, batch or interactive, hour of week, and the pseudonymized account's past ratio of actual to requested runtime.
- **Hard bound for free:** remaining life can never exceed the time limit minus elapsed time.
- **First model:** empirical conditional survival per bucket (partition, time-limit band, account-history band). No learning needed. A learned survival model replaces it only if shadow results justify it.

Conditioning on elapsed time matters most. Jobs that die early from typos and crashes are gone within minutes, so a job that has survived its first half hour is a different population. The memory warm-up in section 8 filters the same early deaths.

### Admission rule

A donor is worth filling when its expected usable life covers the load cost several times over:

```latex
\mathbb{E}\big[\min(L, H)\big] \;\ge\; \alpha \cdot t_{\text{load}}(M)
```

L is remaining life, H the planning horizon, t\_load the time to fill M at the throttled copy rate, and α a safety multiple.

**Example.** Filling 47 GiB at a throttled 1 GB/s takes about 50 seconds. With α = 10, the donor needs about 8 expected minutes of life. Both numbers are placeholders to tune, but the conclusion is robust: the bar is far below 4 hours. The 4-hour screen understates the usable pool.

Lifetime therefore decides role more than admission. Long expected life earns standbys, where durability matters. Short or uncertain life earns primaries only when a stable standby exists.

### Errors are asymmetric but both are safe

| Error | What happens | Cost |
| --- | --- | --- |
| Predicted too long; job ends early | Replica lost, routing fails over, planner re-replicates | Copy traffic and a brief rise in fallback rate |
| Predicted too short | Good memory left unused or under-protected | Capacity |

Since neither breaks correctness, the predictor is tuned for throughput. Its two health metrics are GiB re-replicated per hour and the fallback rate.

### Rollout

1. **Offline replay.** Rerun the Quest 15-minute reconstruction with the predictor in place of the oracle. Report how much of the 19 / 27 / 42 donor counts and 830 / 1,282 / 1,948 GiB survive.
2. **Shadow mode.** Predict against a live cluster without acting, and score against what jobs did.
3. **Act**, with planned drains starting a fixed lead time before each job's limit.

## Worked capacity example against Quest

Memory covers the 671B example's expert tier in all three Quest scenarios; compute is what limits tokens. Quest is the sizing reference here, not a deployment target.

### Memory

Copies are net of an assumed 1 GiB per donor for the CUDA context and activation arena. The expert tier is 609 GiB at 8 bits.

| Scenario | Donors with 4+ hours left | Pooled VRAM after guard | Copies of the expert tier | What the planner does |
| --- | --- | --- | --- | --- |
| P10 | 19 | 830 GiB | 1.3 | One primary per bundle; standbys for the third of bundles with the most fragile primaries |
| Median | 27 | 1,282 GiB | 2.1 | Primary and standby for every bundle |
| P90 | 42 | 1,948 GiB | 3.1 | Primary, standby, and a third replica for hedging |

**Stress case.** Suppose only half the P10 pool is really usable. Coverage falls to 0.7 copies, and with flat traffic about a third of expert calls land on the anchor. That is too much for one or two anchors to cache. The right response is to step down to a 4-bit build of the model (about 305 GiB) or a smaller model, at request boundaries, not to page.

### Compute

Memory need is fixed at about eight 80 GB GPUs' worth whatever the load. Compute need scales with load. The figures use the decode table in section 6, the plan's 4 input tokens per output token, and an assumed 0.4 ms of expert kernel time per prefill token.

| Daily output tokens | Output tokens per second | Expert compute needed |
| --- | --- | --- |
| 40M (the plan's 2,000-researcher example) | 463 | 1.5 to 3.5 GPU-equivalents |
| 75M (the plan's break-even with cloud) | 868 | 2.9 to 6.5 GPU-equivalents |

The low end of each range is decode at B = 256, the high end at B = 64.

Illustrative supply on a Quest-sized cluster: about 11 quiet hosts at κ = 0.25 give roughly 2.8 GPU-equivalents. Unallocated GPUs average about 24 by accounting (131 configured less 106.8 allocated), each worth 1.0 while it lasts, but that count is unverified.

### The headline metric

Expert kernel time works out to 0.9 to 2.1 GPU-hours per million output tokens. If every one of those milliseconds displaced host work, the plan's Gate C target of 0.5 would be missed by 2 to 4 times.

So at most a quarter to a half of expert execution may collide with a busy host. The rest must land on unallocated GPUs, on anchors, or in the idle gaps of quiet hosts. That is the whole reason κ exists and why busy hosts store rather than execute.

### Dedicated footprint

The service owns one or two anchor GPUs on one node, plus about 700 GB of NVMe for the master copy. The plan's cost model compares against an 8-GPU dedicated server for the same model.

## Validation plan, kill criteria and build order

Six experiments, ordered so the cheap ones that can kill or reshape the design run first. The first four need no co-location approval from any center and at most a rented two-GPU machine.

| # | Experiment | What it measures | Decision it makes | Needs |
| --- | --- | --- | --- | --- |
| 1 | Trace-driven simulator: replay the Quest 15-minute job intervals through the planner, predictor and eviction rules | Replica coverage over time, fallback rate, GiB re-replicated per hour, oracle versus predictor | If predicted coverage at P10 falls below one copy, the default becomes a smaller or 4-bit model | A laptop and the data in hand |
| 2 | Routing traces: run the target model family on research-style prompts and log every gate decision | Aggregate versus per-request heat, co-activation, groups touched per token | Confirms or overturns the flat-traffic assumption; fixes the bundle definition and anchor cache policy | One rented GPU node |
| 3 | Co-location micro-benchmark: a host code plus an executor in a second context injecting expert kernels at κ from 0.5% to 25% | Host slowdown versus κ, our wait time per call, cost per context switch | **The gating number.** If typical codes hit 5% slowdown below about κ = 2%, busy hosts become store-only | One rented GPU |
| 4 | Reclaim spike test: the host allocates in steps of rising size and speed while slabs are loaded | Time to free, host allocation failures over many trials, with and without ring 3 | Sets the guard and poll interval. Pass is zero host failures for steps inside the guard | One rented GPU |
| 5 | Split inference: anchor and donors on one node, then two; kill donors mid-step | Logit agreement with a monolithic run, per-layer round trip, hedge hit rate, corrupted sequences (must be zero) | Proves the correctness claim; fixes the hop budget and default g | Two GPU nodes with RDMA |
| 6 | Fabric sweep: experiment 5 on EDR, HDR and NDR class links | Round trip and tail by generation | Chooses the anchor's island and whether older islands are store-only | Site access |

Experiments 3 and 4 are the plan's Gate A slowdown experiment, restated for this architecture. Use the three host codes the build list already names.

**Test model.** The 671B example does not fit a rented pair. The split is scale-free, so use a small MoE with many experts per layer and scale the results by the formulas in sections 6 and 10.

### Kill criteria

- **Co-located execution** is dropped if experiment 3 shows no κ that is both useful and under the 5% cap. The design survives as store donors plus unallocated-GPU executors.
- **Co-located storage** is dropped if experiment 4 cannot reach zero host failures at a guard that still leaves useful headroom.
- **Cross-node execution** is dropped for a fabric if experiment 6 shows hop latency that makes step time unusable even at large batch.

### Build order

1. **Simulator, planner, predictor.** Pure software, and the planner code is reused in production.
2. **Executor, slab manager, watchdog** on one GPU. Enables experiments 3 and 4.
3. **Anchor gate and dispatcher** against a small MoE on one node. Enables the first half of experiment 5.
4. **RDMA transport, hedging, routing epochs.** Completes experiment 5.
5. **Compute governor** with on and off windows and AIMD.
6. **Slurm adapter, kill switch, accounting** for the headline metric.

Per the plan's constraint, none of this runs on Quest. Quest's accounting data informs sizing; experiments run on rented or customer hardware.

## Open risks and what this design does not solve

The largest risk is policy, not physics: running our process on a GPU allocated to someone else's job needs each center's consent, and the brief itself calls co-location unapproved.

| Risk | Why it matters | Response |
| --- | --- | --- |
| Centers refuse co-location | No store or execute donors on busy hosts | Three opt-in levels: unallocated GPUs only, then store donors, then execute donors. The architecture runs at any level |
| Time-slicing is thinly documented | Slice length, switch cost and the cost of a mostly idle second context are unknown | Experiment 3 measures all three before anything else is built on them |
| 58 serial hops per step | Per-stream speed has a ceiling | Position as a throughput tier; interactive traffic uses small batches or a dedicated pool |
| The anchor is a single point of failure | Sequences in flight on a failed anchor are lost | Two anchors with independent KV caches; failed sequences restart from the prompt |
| Pooled memory may not combine as assumed | The capacity table overstates supply | Experiment 1 reports coverage under stricter pooling rules |
| Replicas on different GPU generations differ in low bits | Reproducible research runs | Requests flagged reproducible are pinned to the anchor and one GPU generation |
| Memory reuse between processes | Security review of activations on shared GPUs | Zeroed arena now; test and find NVIDIA's documented guarantee before any sensitive class uses donors |
| Zeros in utilization may be missing telemetry | The count of quiet hosts may be wrong in either direction | Report sizing with zeros included and excluded |
| Sizing is for one model family | Other MoEs lack routing groups and have different expert sizes | The planner is parametric; redo sections 3 and 10 per model |

**Relation to prior work.** Harvest keeps the authoritative copy in host DRAM and treats peer GPU memory as a best-effort cache inside one two-GPU VM ([Harvest](https://arxiv.org/html/2602.00328v1)). This design keeps that rule, moves the authoritative copy to the anchor, crosses nodes, and executes in place instead of pulling weights.

**Out of scope here:** the gateway, billing, data-class policy, the choice of serving engine to fork, and running several models on one donor pool.

## Sources

- [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437): layer count, hidden size, expert counts, node-limited routing
- [DeepSeek-V3 config\_671B.json](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/b3480c8cccc78b782bb7623abb79cc4d104831ad/inference/configs/config_671B.json): routing-group parameters
- [MoE-Infinity (arXiv 2401.14361)](https://arxiv.org/html/2401.14361v2): per-request versus aggregate expert activation
- [Harvest (arXiv 2602.00328)](https://arxiv.org/html/2602.00328v1): peer-memory placement and revocation
- [NVIDIA MPS guide: when to use MPS](https://docs.nvidia.com/deploy/mps/615/when-to-use-mps.html): compute modes, single-user limit, fault reporting
- [NVIDIA Run:ai: GPU time-slicing](https://run-ai-docs.nvidia.com/self-hosted/platform-management/runai-scheduler/resource-optimization/time-slicing): default time-slicing shares runtime evenly
- Northwestern Quest retrospective, 21 Sep to 21 Oct 2026 (design brief, 5 Nov 2026): all Quest figures
- Project docs: the Cleanroom plan, the engineering build list, and the Clemson interview notes, used for constraints and gate targets only

Sizing figures not listed above are this document's own arithmetic and are marked as estimates where they depend on unmeasured hardware behaviour.
