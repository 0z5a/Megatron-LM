# H100 HybridEP virtual-expert adapter

[Draft PR #8388](https://github.com/NVIDIA/Megatron-LM/pull/8388) adds the existing
HybridEP and VirtualExpertLoadBalancer paths to a finite logical-workload comparison.
The tested source is `9d5504449bb806689d4c1c9d53b3a43e31279f0d`, recorded as an
explicit overlay on the initial source archive. HybridEP is fixed at
`10d4dd7377d5bce900fbb4b80cce863764892b95`.

Two H100 SXM 80 GB devices had NV18 connectivity and bidirectional peer access.
NGC PyTorch 26.09: Torch 2.14.0a0+b2c75dd062.nv26.09, CUDA 13.4, Triton 3.8.0,
TE 2.19.0+5e52befd, NCCL 2.31.2. The HybridEP extension was built only for sm90,
without its optional multinode path.

All **336 rank records passed**: seven workloads x six steps x two backends x
two ranks x two modes. Both backends used identical inputs, logical routes and
initial parameters. Outputs, input/probability/parameter gradients and SGD-updated
parameters (local SGD updates without a DDP wrapper) met relative L2 <= 0.01 and peak-normalized error <= 0.05; all-zero
references required exact equality. This is BF16 numerical parity, not bit-exact
qualification or full-model convergence. The source tokens and K were fixed.

The independent reader verified 50 complete original evidence files. It checked
each route/input pairing and all expected step IDs. Original negative windows
were preserved: NVSHMEM 3.4.5 shadowed the newer NGC runtime and broke Torch import;
removing that added wheel restored the original runtime. A benchmark cleanup
import typo stopped the first partial run and was fixed before the complete run.

The following times are means of six per-step worst-rank times, in milliseconds.
The last columns are mean rank-time max/mean ratios. Warmup covers two complete
routing trajectories without optimizer updates. HybridEP runs first, so these
fixed-order measurements are descriptive and do not establish a speedup.

| Mode | Workload | HybridEP ms | Virtual experts ms | HybridEP max/mean | Virtual max/mean |
| --- | --- | ---: | ---: | ---: | ---: |
| training_step | W0_balanced | 8.851 | 9.170 | 1.147 | 1.165 |
| training_step | W1_hot_expert_set | 10.380 | 10.628 | 1.331 | 1.240 |
| training_step | W3_spread_hot_experts | 10.464 | 10.816 | 1.348 | 1.234 |
| training_step | W4_random_balanced | 6.171 | 13.002 | 1.022 | 1.145 |
| training_step | W5_zero_expert_rank | 9.648 | 12.928 | 1.182 | 1.140 |
| training_step | W6_rotating_hotspot | 10.000 | 13.108 | 1.201 | 1.147 |
| training_step | W7_burst | 9.991 | 8.962 | 1.202 | 1.009 |
| moe_fwd_bwd | W0_balanced | 6.859 | 9.761 | 1.024 | 1.035 |
| moe_fwd_bwd | W1_hot_expert_set | 5.882 | 8.767 | 1.010 | 1.005 |
| moe_fwd_bwd | W3_spread_hot_experts | 6.027 | 8.778 | 1.012 | 1.004 |
| moe_fwd_bwd | W4_random_balanced | 5.904 | 8.854 | 1.008 | 1.007 |
| moe_fwd_bwd | W5_zero_expert_rank | 5.853 | 8.827 | 1.027 | 1.009 |
| moe_fwd_bwd | W6_rotating_hotspot | 5.904 | 8.847 | 1.010 | 1.157 |
| moe_fwd_bwd | W7_burst | 5.835 | 9.639 | 1.105 | 1.077 |

Per-step logical rank loads, runtime virtual-expert loads, all-rank times and
all numerical error measurements are in the four original JSONL files.
