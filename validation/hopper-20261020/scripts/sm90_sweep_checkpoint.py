#!/usr/bin/env python3
"""Time every Triton autotune candidate of the kernels in the packaged tuning tables on this GPU.

Methodology follows megatron/core/tuning/README.md for the packaged sm103 table:
  * the MLA RoPE and mHC kernels are swept at 2k, 8k and 32k tokens, since the token count is
    not part of their tuning keys;
  * the mamba_ssm SSD kernels are run at the Nemotron-3 Ultra mixer shapes that the packaged
    keys encode (chunk 128, 128 heads, head dim 64, state 128, so dim 8192), also at three
    sequence lengths, with and without the gate z (the packaged keys were recorded with it).

Megatron's tuning adapter is installed in record mode, so keys are built by
`selection.tuning_key` exactly as the tables expect. A wrapper around Triton's Autotuner.run
keeps `configs_timings` (every candidate's do_bench quantiles) for each new (kernel, key); the
autotune caches are cleared between token counts so every case is re-benchmarked.

Output: one JSON with raw timings per (kernel, key, case); `sm90_build_table.py` turns it into
a table. Run on an otherwise idle GPU: Triton picks winners by timing.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

TOKENS = (2048, 8192, 32768)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Megatron-LM checkout")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--families", default="mla,mhc,ssd")
    ap.add_argument("--tokens", default=",".join(str(t) for t in TOKENS))
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    tokens = [int(t) for t in args.tokens.split(",")]

    import triton
    from triton.runtime.autotuner import Autotuner

    from megatron.core.tuning import AutotunePolicy, install
    from megatron.core.tuning import selection

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    install(AutotunePolicy(mode="record", record_path=str(out.with_suffix("")) + ".rec"))

    records: dict = {}  # qualified -> key -> case -> {sig: [quantiles]}
    configs: dict = {}  # qualified -> sig -> config_data
    seen: set = set()
    state = {"case": None}
    policy_run = Autotuner.run

    def capture_run(self, *a, **kw):
        before = len(getattr(self, "cache", ()))
        result = policy_run(self, *a, **kw)
        if len(getattr(self, "cache", ())) == before:
            return result
        timings = getattr(self, "configs_timings", None)
        if not timings:
            return result
        seen.add(self)
        qualified = f"{selection.kernel_module(self)}.{selection.kernel_name(self)}"
        key = selection.tuning_key(self, a, kw)
        slot = records.setdefault(qualified, {}).setdefault(key, {})
        per = {}
        for cfg, t in timings.items():
            sig = selection.config_signature(cfg)
            configs.setdefault(qualified, {})[sig] = selection.config_data(cfg)
            per[sig] = [float(x) for x in (t if isinstance(t, (list, tuple)) else [t])]
        slot[state["case"]] = per
        return result

    Autotuner.run = capture_run

    def fresh(case: str) -> None:
        for tuner in seen:
            tuner.cache.clear()
        state["case"] = case
        torch.cuda.synchronize()

    dev = torch.device("cuda")
    bf16 = torch.bfloat16
    families = args.families.split(",")
    errors: list = []

    import subprocess

    my_pid = os.getpid()
    contaminated: list = []

    def foreign_procs() -> list:
        """Compute processes on this GPU other than this one."""
        try:
            lines = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30, check=True).stdout.splitlines()
        except Exception as exc:
            return [f"nvidia-smi failed: {exc}"]
        return [l for l in lines if l.strip() and int(l.split(",")[0]) != my_pid]

    def run(case, fn):
        fresh(case)
        before = foreign_procs()
        try:
            fn()
            torch.cuda.synchronize()
        except Exception as exc:  # keep sweeping; the error is part of the record
            errors.append({"case": case, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[sweep] {case}: {type(exc).__name__}: {exc}", flush=True)
        after = foreign_procs()
        if before or after:
            # Someone else's process shared the GPU while candidates were timed: drop the case.
            contaminated.append({"case": case, "before": before, "after": after})
            for keys in records.values():
                for cases in keys.values():
                    cases.pop(case, None)
            print(f"[sweep] {case}: dropped, foreign GPU process {before or after}", flush=True)

        # Preserve every finished case if a later case is interrupted; partial is not a table.
        partial = out.with_suffix('.partial.json')
        tmp = partial.with_suffix('.tmp')
        tmp.write_text(json.dumps({'status': 'PARTIAL', 'last_case': case, 'records': records,
                                  'configs': configs, 'errors': errors,
                                  'contaminated': contaminated}, indent=1, sort_keys=True))
        tmp.replace(partial)

    # ------------------------------------------------------------------ MLA RoPE
    if "mla" in families:
        from megatron.core.fusions.fused_mla_yarn_rope_apply import (
            fused_mla_rope_inplace,
            fused_mla_rope_kv_split,
        )

        q_dim, emb, k_dim, v_dim = 128, 64, 128, 128
        for heads in (1, 2, 4, 8, 16, 32, 64, 128):
            for n in tokens:
                def q_case(heads=heads, n=n):
                    cos = torch.randn(n, 1, 1, emb, device=dev, dtype=bf16)
                    sin = torch.randn(n, 1, 1, emb, device=dev, dtype=bf16)
                    t = torch.randn(n, 1, heads, q_dim + emb, device=dev, dtype=bf16,
                                    requires_grad=True)
                    y = fused_mla_rope_inplace(t.clone(), cos, sin, q_dim, emb)
                    y.backward(torch.randn_like(y))

                def kv_case(heads=heads, n=n):
                    cos = torch.randn(n, 1, 1, emb, device=dev, dtype=bf16)
                    sin = torch.randn(n, 1, 1, emb, device=dev, dtype=bf16)
                    kv = torch.randn(n, 1, heads, k_dim + v_dim, device=dev, dtype=bf16,
                                     requires_grad=True)
                    pe = torch.randn(n, 1, 1, emb, device=dev, dtype=bf16, requires_grad=True)
                    k, v = fused_mla_rope_kv_split(kv, pe, cos, sin, emb, k_dim, v_dim)
                    (k.float().sum() + v.float().sum()).backward()

                run(f"mla_q|h={heads}|tokens={n}", q_case)
                run(f"mla_kv|h={heads}|tokens={n}", kv_case)

    # ----------------------------------------------------------------------- mHC
    if "mhc" in families:
        from megatron.core.fusions.fused_mhc_kernels import (
            fused_h_aggregate,
            fused_h_post_bda,
            fused_sinkhorn,
        )

        streams = 4
        for hidden in (2688, 4096, 7168):
            for n in tokens:
                def mhc_case(hidden=hidden, n=n):
                    s, b = n, 1
                    logits = torch.randn(s, b, streams, streams, device=dev, requires_grad=True)
                    p = fused_sinkhorn(logits, 20, backend="triton")
                    p.backward(torch.randn_like(p))
                    x = torch.randn(s, b, streams, hidden, device=dev, dtype=bf16,
                                    requires_grad=True)
                    h_pre = torch.rand(s, b, streams, device=dev, dtype=bf16, requires_grad=True)
                    agg = fused_h_aggregate(x, h_pre, backend="triton")
                    agg.backward(torch.randn_like(agg))
                    h_res = torch.rand(s, b, streams, streams, device=dev, dtype=bf16,
                                       requires_grad=True)
                    orig = torch.randn(s, b, streams, hidden, device=dev, dtype=bf16,
                                       requires_grad=True)
                    h_post = torch.rand(s, b, streams, device=dev, dtype=bf16, requires_grad=True)
                    xx = torch.randn(s, b, hidden, device=dev, dtype=bf16, requires_grad=True)
                    o = fused_h_post_bda(h_res, orig, h_post, xx, None, backend="triton")
                    o.backward(torch.randn_like(o))

                run(f"mhc|C={hidden}|N={streams}|tokens={n}", mhc_case)

    # ----------------------------------------------------------------------- SSD
    if "ssd" in families:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        nheads, headdim, dstate, ngroups, chunk = 128, 64, 128, 8, 128
        for with_z in (True, False):
            for n in tokens:
                def ssd_case(n=n, with_z=with_z):
                    b, length = 1, n
                    x = torch.randn(b, length, nheads, headdim, device=dev, dtype=bf16,
                                    requires_grad=True)
                    dt = torch.randn(b, length, nheads, device=dev, dtype=bf16, requires_grad=True)
                    A = -(torch.rand(nheads, device=dev, dtype=torch.float32) + 0.5)
                    A.requires_grad_(True)
                    B = torch.randn(b, length, ngroups, dstate, device=dev, dtype=bf16,
                                    requires_grad=True)
                    C = torch.randn(b, length, ngroups, dstate, device=dev, dtype=bf16,
                                    requires_grad=True)
                    D = torch.ones(nheads, device=dev, dtype=bf16, requires_grad=True)
                    z = (torch.randn(b, length, nheads, headdim, device=dev, dtype=bf16,
                                     requires_grad=True) if with_z else None)
                    dt_bias = torch.rand(nheads, device=dev, dtype=torch.float32,
                                         requires_grad=True)
                    y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk, D=D, z=z, dt_bias=dt_bias,
                                                  dt_softplus=True)
                    y.backward(torch.randn_like(y))

                run(f"ssd|z={int(with_z)}|tokens={n}", ssd_case)

    Autotuner.run = policy_run
    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "arch": selection.arch_tag(),
        "driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "host_arch": platform.machine(),
        "tokens": tokens,
        "families": families,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("TRITON_", "MAMBA_"))},
    }
    try:
        import mamba_ssm

        meta["mamba_ssm"] = mamba_ssm.__version__
    except ImportError:
        pass
    out.write_text(json.dumps({"meta": meta, "records": records, "configs": configs,
                               "errors": errors, "contaminated": contaminated},
                              indent=1, sort_keys=True))
    n_keys = sum(len(v) for v in records.values())
    print(f"[sweep] {len(records)} kernels, {n_keys} keys, {len(errors)} errors, "
          f"{len(contaminated)} dropped cases -> {out}")
    return 0 if not errors and not contaminated else 1


if __name__ == "__main__":
    raise SystemExit(main())
