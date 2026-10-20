# Hopper validation, 20 October 2026

This branch preserves a finite experiment record and reproduction helpers for
Megatron-LM contributions. The upstream source was frozen at
`67413f611ffb1e7d7d29cd2a61f520cffe79d995` for the cross-host and model checks.
The CE matrix and native-only deterministic admission are submitted in
[PR #8380](https://github.com/NVIDIA/Megatron-LM/pull/8380).

The GH200 environment is a task-local aarch64 Python environment, not the
Megatron CI container: PyTorch 2.14.1+cu130, CUDA runtime 13.0, Triton 3.8.0,
NCCL 2.30.7, Transformer Engine 2.20.2+44f31bbe, NVIDIA driver 595.58.03.
Each of the two hosts has one NVIDIA GH200 480GB device, with 97871 MiB reported
by `nvidia-smi`. Cross-host NCCL logs identified `NET/IB/0/GDRDMA` transport.

## Completed cross-host checks

- The NCCL-EP full `MoELayer` pytest cell passed on both ranks. Its three
  replays compare outputs, input gradients and parameter gradients under
  side-stream contention. The same submitted cell also passed on two H100
  SXM 80 GB GPUs connected by NV18, in the NGC PyTorch 26.09 environment.
- The CE test matrix ran in three separate two-host launches, each using
  fresh Inductor and Triton cache directories. Every rank reported eight
  passing cells: TP1/TP2, bf16/fp32, and strict deterministic algorithms
  enabled/disabled. Each cell ran four contention replays.
- All eight strict-mode case/rank groups retained identical input, output
  and gradient SHA256 digests across cold processes. Six of the eight
  default-mode groups changed output or gradient digests across processes,
  although every within-process replay passed. Separate TP1 runs on each
  GH200 reproduced the strict/default distinction. In one cold-cache series,
  the default reduction changed `R0_BLOCK` from 32768 to 16384.
- DP gradient reduce-scatter passed a CPU sum reference and five replays per
  dtype for bf16/fp32. Three independent process launches on the same two
  physical hosts produced identical per-rank results.

The collective result is **not** a comparison of different Slurm allocations.
It does not establish that topology-related reproducibility gaps are limited
to GPUs within one node. The CE result also does not qualify other compiler
versions, batch invariance, the TE CE backend, or arbitrary compiler settings. PR #8380 now permits the
measured native path under strict algorithms while retaining the TE restriction.

The first cross-host pytest launch failed because Gloo resolved a hostname to
a loopback address. Binding `GLOO_SOCKET_IFNAME` to the inter-host interface
resolved startup. Raw startup failures and an original-lock refusal were
retained with the private execution archive; the result summaries here do not
turn those windows into successful tests. Hostnames in the public collective
records are replaced with rank labels. No GPU lease or credential information
is included.

## Reproduce the fixed-host collective check

Use the same source, dependency builds and two physical hosts. Set each
host's interface names explicitly, using its actual inventory:

```bash
export NCCL_ALGO=Ring NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME="$INTERFACE" GLOO_SOCKET_IFNAME="$INTERFACE"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m torch.distributed.run --nnodes=2 --nproc-per-node=1 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29701 \
  validation/hopper-20261020/scripts/cross_host_collective_probe.py \
  --out="$RESULT_DIR/collective-r1"
```

Run this once on each host with ranks 0 and 1, then repeat with fresh process
groups and separate output names. Compare the `records` arrays within each
rank across launches. Retain the actual topology and NCCL transport logs.

For NCCL-EP, additionally set `NCCL_EP_HT_EM_AG_SCAN_MODE=1` before process
group creation and invoke the upstream test cell from PR #8375. For the CE
matrix, invoke `TestFusedCrossEntropy` from PR #8380 with `--runxfail`; use
new `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` directories for every
independent launch. The optional `ce_receipt_plugin.py` records the tensors
returned by the unchanged original replay assertion.

## sm90 table validation protocol

The separate GH200 sweep covers 63 MLA RoPE/mHC/SSD shapes at 2k, 8k and 32k
tokens and reaches 25 multi-config kernels. The GH200 candidate contains
42 table entries. All 63 shapes passed two independent ordered-reduction
processes, each with two replay iterations, with equal output/gradient and
selected-configuration digests. The pinning-only control failed six SSD
gradient checks; ordered reductions are required in addition to fixed
candidate selection.

The H100 sweep used a different runtime and produced a separate candidate;
its choices are not merged into the GH200 table. A table measured on one
Hopper machine is not a performance guarantee for all Hopper machines.

`model-step-validation-protocol.json` freezes the model-step comparison:
three families, three selection modes, three independent processes per mode,
four warmup steps and ten measured steps, with rotated mode order. These are
synthetic GPT/HybridModel forward/backward/SGD steps, not convergence runs.
The helper must run against the frozen source **without a packaged sm90
table**, so its `cheapest` control really uses the fallback. Use the external
candidate directory only for the `table` mode.

The 27 model-step cases completed with finite losses, no foreign GPU process,
and 94 original files independently SHA256-verified. The candidate **does not
pass the preregistered model qualification**:

| Family | Table-mode loss/parameter/gradient digests across three processes | Table/auto step-time ratio, paired 95% interval |
| --- | --- | --- |
| MLA | One of three processes differs | 1.120 [0.495, 2.531] |
| mHC | Equal | 1.484 [0.642, 3.429] |
| SSD | Equal | 0.972 [0.697, 1.356] |

All three table-mode configuration digests match across processes. The MLA
mismatch therefore is not explained by a different recorded table choice.
The auto SSD control also varies in losses/parameters/gradients. No family
meets the frozen upper-confidence-bound <= 1.05 timing gate. The intervals
are paired Student-t intervals on log median-step ratios (n=3, df=2); these
small, noisy samples do not establish a performance improvement. A zero
process exit code was not treated as a qualification pass. Raw per-process
model results and `model-step-comparison-audit.json` preserve these negatives.

The full aarch64 unit inventory is being continued after a data-preprocessing
worker timeout; a completed first collection is not reported as a suite pass.

The table builder is `scripts/build_candidate.py`. It excludes invalid/nonpositive candidate timings, rejects mixed-runtime merges, and uses each case's fastest valid candidate as the slowdown denominator. `results/gh200-sweep.json` preserves the complete recorded candidate data; `results/sm90-gh200-candidate.json` is the exact independently replayed candidate. These recording-time estimates are not held-out performance results.

## CE performance on GH200

The profiled and independent unprofiled runs use TP1, 2048 tokens, 32768
vocabulary entries per rank, bf16/fp32, and fused/unfused CE. Reference loss
and gradient checks passed. The corrected helper clears
`torch.utils.deterministic.fill_uninitialized_memory`, matching Megatron's
deterministic setup. Original fill-on controls are retained in the execution
archive but are not mixed into the results below.

Each unprofiled mode ran in three cold processes, with eight warmup and 50
measured forward/backward iterations. The mode order alternated by repeat.
The geometric mean of the three paired strict/default median-time ratios was
1.337 for bf16 fused CE and 1.539 for fp32 fused CE. Two-sided 95% Student-t
intervals on the three log ratios were [0.930, 1.921] and [1.236, 1.917],
respectively. The small sample and observed variation do not support a claim
of negligible overhead. These are component times, not training-step times.

The Nsight NVTX CSVs are included separately; profiler instrumentation affects
the timings. The original wrapper terminated torchrun when the capture range
ended and never reached the second mode. The one-line fix in
[PR #8382](https://github.com/NVIDIA/Megatron-LM/pull/8382) lets the target exit
normally; both profile reports and GPU-kernel/NVTX exports then completed.

## Native CE admission follow-up

The updated native-only admission passed 34 parameter tests. The normal TP1
replay invocation passed both strict cells; default cells were XPASS and TP2
cells skipped for this single-host invocation. Three fresh processes ran four
actual GPT forward/backward/SGD steps each with native CE enabled through
`apply_determinism_to_args`. Input, initial and final parameters, gradients and
loss digests all matched. `native-ce-admission-audit.json` binds the exact
changed-source hashes and 12 independently verified original files.

## MLA follow-up diagnostics and candidate delivery

[PR #8384](https://github.com/NVIDIA/Megatron-LM/pull/8384) keeps the candidate
outside packaged defaults. It can be selected explicitly for investigation;
its model qualification remains incomplete. Five additional table-mode MLA
processes on the second GH200 matched their input/initial-parameter digests
and all three optimizer-step results. This is a second-host diagnostic and
does not invalidate the first host's original mismatch.

Compute Sanitizer 2026.1.1 from the CUDA 13.2 toolkit reported zero errors for
an isolated MLA memcheck and zero hazards for racecheck. The same four MLA
forward/backward kernels and table choices were exercised without NCCL
initialization. The earlier full-model sanitizer run recorded 90 handled
NCCL architecture-probe API errors and exited 86; that negative window is
preserved in the original archive and counted in `mla-diagnosis-audit.json`.
A sanitizer result is not a cross-process numerical or performance proof.
The two short public sanitizer logs replace the private work directory with
`<WORK_DIR>`; result JSONs retain the exact selected configurations.
