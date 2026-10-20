#!/usr/bin/env python3
"""Build an sm90 tuning table from one or more sm90_sweep.py outputs.

Selection rule, as described for the packaged sm103 table (megatron/core/tuning/README.md):
  * megatron.core fusions (MLA RoPE, mHC): their tuning keys do not contain the token count, so
    keep, per kernel, the config with the smallest worst-case slowdown over every swept shape,
    token count and machine as a `*` entry, plus an entry for each key whose own best config
    beats `*` there by more than 5% (worst case over that key's cases);
  * every other kernel (mamba_ssm SSD): one entry per recorded key, the config with the
    smallest worst-case slowdown over that key's cases.
Slowdown is a candidate's median time over the fastest candidate's median in the same case.
Only candidates timed in every case being compared are eligible.

Also writes a per-kernel report: chosen config vs Triton's per-case winner vs the cheapest
candidate (the pinned fallback without a table).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STAR_PREFIXES = ("megatron.core.fusions.",)
OVERRIDE_MARGIN = 1.05


def _median(q):
    return q[0]


def load(paths):
    records, configs, metas = {}, {}, []
    for i, path in enumerate(paths):
        data = json.loads(Path(path).read_text())
        metas.append(data["meta"])
        for kernel, keys in data["records"].items():
            for key, cases in keys.items():
                slot = records.setdefault(kernel, {}).setdefault(key, {})
                for case, timings in cases.items():
                    slot[f"m{i}:{case}"] = {s: _median(q) for s, q in timings.items()}
        for kernel, sigs in data["configs"].items():
            configs.setdefault(kernel, {}).update(sigs)
    return records, configs, metas


def worst_slowdown(cases: dict) -> dict:
    """sig -> max over cases of t/min(t); only sigs present in every case."""
    common = set.intersection(*(set(t) for t in cases.values()))
    worst = {}
    for timings in cases.values():
        best = min(timings[s] for s in common)
        for s in common:
            worst[s] = max(worst.get(s, 0.0), timings[s] / best)
    return worst


def pick(worst: dict) -> str:
    # Ties break on the signature so the result is a pure function of the data.
    return min(worst, key=lambda s: (worst[s], s))


def cheapest_sig(kernel_configs: dict, sigs) -> str:
    """The pinned fallback without a table, as megatron.core.tuning.selection.cheapest:
    min over (product of BLOCK* kwargs x num_stages, num_warps), first candidate on a tie."""
    def cost(sig):
        cfg = kernel_configs[sig]
        blocks = 1
        for k, v in cfg.get("kwargs", {}).items():
            if k.startswith("BLOCK") and isinstance(v, int):
                blocks *= v
        return (blocks * (cfg.get("num_stages") or 1), cfg.get("num_warps") or 1)
    ordered = [s for s in kernel_configs if s in sigs]  # candidate order, as Triton lists them
    return min(ordered, key=cost)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--source", required=True, help="provenance string for the table")
    args = ap.parse_args()

    records, configs, metas = load(args.sweeps)
    arches = {m["arch"] for m in metas}
    if len(arches) != 1:
        print(f"ERROR: sweeps cover several architectures {sorted(arches)}", file=sys.stderr)
        return 2
    arch = arches.pop()

    kernels: dict = {}
    report: list = []
    for kernel in sorted(records):
        keys = records[kernel]
        kcfg = configs[kernel]
        entries: dict = {}
        if kernel.startswith(STAR_PREFIXES):
            all_cases = {f"{key}#{case}": t for key, cases in keys.items()
                         for case, t in cases.items()}
            star_worst = worst_slowdown(all_cases)
            star = pick(star_worst)
            entries["*"] = kcfg[star]
            for key, cases in sorted(keys.items()):
                w = worst_slowdown(cases)
                best = pick(w)
                if star in w and w[star] / w[best] > OVERRIDE_MARGIN:
                    entries[key] = kcfg[best]
                chosen = best if key in entries else star
                report.append(_row(kernel, key, cases, chosen, kcfg))
        else:
            for key, cases in sorted(keys.items()):
                best = pick(worst_slowdown(cases))
                entries[key] = kcfg[best]
                report.append(_row(kernel, key, cases, best, kcfg))
        kernels[kernel] = entries

    table = {"arch": arch, "kernels": kernels, "packages": {}, "source": args.source,
             "triton": ""}
    Path(args.out).write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    lines = ["| kernel | key | cases | chosen / per-case best (worst) | cheapest / best (worst) |",
             "|---|---|---|---|---|"]
    for r in report:
        lines.append(f"| `{r['kernel'].split('.')[-1]}` | `{r['key'][:60]}` | {r['cases']} | "
                     f"{r['chosen_worst']:.3f} | {r['cheapest_worst']:.3f} |")
    meta_lines = [f"- {m['gpu']} ({m['arch']}), torch {m['torch']}, CUDA {m['cuda']}, "
                  f"triton {m['triton']}, mamba_ssm {m.get('mamba_ssm')}, {m['host_arch']}"
                  for m in metas]
    Path(args.report).write_text("\n".join(meta_lines + [""] + lines) + "\n")
    print(f"{len(kernels)} kernels, {sum(len(v) for v in kernels.values())} entries -> {args.out}")
    return 0


def _row(kernel, key, cases, chosen, kcfg):
    w = worst_slowdown(cases)
    cheap = cheapest_sig(kcfg, w)
    return {"kernel": kernel, "key": key, "cases": len(cases),
            "chosen_worst": w[chosen], "cheapest_worst": w[cheap]}


if __name__ == "__main__":
    raise SystemExit(main())
