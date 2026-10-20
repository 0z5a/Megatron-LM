#!/usr/bin/env python3
"""Full recorded shape coverage: pinned execution, seeded output/gradient replay.

Component timing includes input generation plus forward/backward. It is not a
training step time or a numerical comparison between auto and pinned outputs.
"""
import argparse,hashlib,json,os,sys,time,subprocess,platform
from pathlib import Path
import torch

def digest(t):
    if t is None: return None
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--table',required=True)
    ap.add_argument('--families',default='mla,mhc,ssd')
    args=ap.parse_args()
    sys.path.insert(0,args.repo)
    import triton
    from megatron.core.tuning import AutotunePolicy,install,choice_log,choice_digest
    install(AutotunePolicy(mode='pinned', table_path=(args.table,), on_miss='error',
                           enumerate_autotuners=True))
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    results=[]; errors=[]; contaminated=[]
    dev=torch.device('cuda'); bf16=torch.bfloat16
    families=args.families.split(','); tokens=[2048,8192,32768]
    # Import full candidate lists before enabling ordered runtime reductions.
    import megatron.core.fusions.fused_mla_yarn_rope_apply
    import megatron.core.fusions.fused_mhc_kernels
    import mamba_ssm.ops.triton.ssd_combined
    torch.use_deterministic_algorithms(True)
    os.environ['MAMBA_DETERMINISTIC']='1'
    def foreign():
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
        return [x for x in rows if x.strip() and int(x)!=os.getpid()]
    def run(case,fn):
        before=foreign(); record={'case':case,'replays':[]}
        try:
            for replay in range(2):
                torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
                torch.cuda.synchronize()
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record(); outputs=fn();end.record();torch.cuda.synchronize()
                record['replays'].append({'component_ms_including_rng':start.elapsed_time(end),
                    'digests':{name:digest(t) for name,t in outputs.items()}})
                del outputs
            record['bit_exact']=record['replays'][0]['digests']==record['replays'][1]['digests']
            if not record['bit_exact']: errors.append({'case':case,'error':'replay digest mismatch'})
        except Exception as exc:
            errors.append({'case':case,'error':repr(exc)})
        after=foreign()
        if before or after: contaminated.append({'case':case,'before':before,'after':after})
        results.append(record)
        out.with_suffix('.partial.json').write_text(json.dumps({'status':'PARTIAL','results':results,'errors':errors,'contaminated':contaminated},indent=1))
        print(case, 'bit_exact',record.get('bit_exact'),flush=True)
    if 'mla' in families:
        from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace, fused_mla_rope_kv_split
        (q_dim, emb, k_dim, v_dim) = (128, 64, 128, 128)
        for heads in (1, 2, 4, 8, 16, 32, 64, 128):
            for n in tokens:

                def q_case(heads=heads, n=n):
                    cos = torch.randn(n, 1, 1, emb, device=dev, dtype=bf16)
                    sin = torch.randn(n, 1, 1, emb, device=dev, dtype=bf16)
                    t = torch.randn(n, 1, heads, q_dim + emb, device=dev, dtype=bf16, requires_grad=True)
                    y = fused_mla_rope_inplace(t.clone(), cos, sin, q_dim, emb)
                    y.backward(torch.randn_like(y))
                    return {'y': y, 't_grad': t.grad}

                def kv_case(heads=heads, n=n):
                    cos = torch.randn(n, 1, 1, emb, device=dev, dtype=bf16)
                    sin = torch.randn(n, 1, 1, emb, device=dev, dtype=bf16)
                    kv = torch.randn(n, 1, heads, k_dim + v_dim, device=dev, dtype=bf16, requires_grad=True)
                    pe = torch.randn(n, 1, 1, emb, device=dev, dtype=bf16, requires_grad=True)
                    (k, v) = fused_mla_rope_kv_split(kv, pe, cos, sin, emb, k_dim, v_dim)
                    (k.float().sum() + v.float().sum()).backward()
                    return {'k': k, 'v': v, 'kv_grad': kv.grad, 'pe_grad': pe.grad}
                run(f'mla_q|h={heads}|tokens={n}', q_case)
                run(f'mla_kv|h={heads}|tokens={n}', kv_case)
    if 'mhc' in families:
        from megatron.core.fusions.fused_mhc_kernels import fused_h_aggregate, fused_h_post_bda, fused_sinkhorn
        streams = 4
        for hidden in (2688, 4096, 7168):
            for n in tokens:

                def mhc_case(hidden=hidden, n=n):
                    (s, b) = (n, 1)
                    logits = torch.randn(s, b, streams, streams, device=dev, requires_grad=True)
                    p = fused_sinkhorn(logits, 20, backend='triton')
                    p.backward(torch.randn_like(p))
                    x = torch.randn(s, b, streams, hidden, device=dev, dtype=bf16, requires_grad=True)
                    h_pre = torch.rand(s, b, streams, device=dev, dtype=bf16, requires_grad=True)
                    agg = fused_h_aggregate(x, h_pre, backend='triton')
                    agg.backward(torch.randn_like(agg))
                    h_res = torch.rand(s, b, streams, streams, device=dev, dtype=bf16, requires_grad=True)
                    orig = torch.randn(s, b, streams, hidden, device=dev, dtype=bf16, requires_grad=True)
                    h_post = torch.rand(s, b, streams, device=dev, dtype=bf16, requires_grad=True)
                    xx = torch.randn(s, b, hidden, device=dev, dtype=bf16, requires_grad=True)
                    o = fused_h_post_bda(h_res, orig, h_post, xx, None, backend='triton')
                    o.backward(torch.randn_like(o))
                    return {'p': p, 'logits_grad': logits.grad, 'agg': agg, 'x_grad': x.grad, 'h_pre_grad': h_pre.grad, 'o': o, 'h_res_grad': h_res.grad, 'orig_grad': orig.grad, 'h_post_grad': h_post.grad, 'xx_grad': xx.grad}
                run(f'mhc|C={hidden}|N={streams}|tokens={n}', mhc_case)
    if 'ssd' in families:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        (nheads, headdim, dstate, ngroups, chunk) = (128, 64, 128, 8, 128)
        for with_z in (True, False):
            for n in tokens:

                def ssd_case(n=n, with_z=with_z):
                    (b, length) = (1, n)
                    x = torch.randn(b, length, nheads, headdim, device=dev, dtype=bf16, requires_grad=True)
                    dt = torch.randn(b, length, nheads, device=dev, dtype=bf16, requires_grad=True)
                    A = -(torch.rand(nheads, device=dev, dtype=torch.float32) + 0.5)
                    A.requires_grad_(True)
                    B = torch.randn(b, length, ngroups, dstate, device=dev, dtype=bf16, requires_grad=True)
                    C = torch.randn(b, length, ngroups, dstate, device=dev, dtype=bf16, requires_grad=True)
                    D = torch.ones(nheads, device=dev, dtype=bf16, requires_grad=True)
                    z = torch.randn(b, length, nheads, headdim, device=dev, dtype=bf16, requires_grad=True) if with_z else None
                    dt_bias = torch.rand(nheads, device=dev, dtype=torch.float32, requires_grad=True)
                    y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
                    y.backward(torch.randn_like(y))
                    return {'y': y, 'x_grad': x.grad, 'dt_grad': dt.grad, 'A_grad': A.grad, 'B_grad': B.grad, 'C_grad': C.grad, 'D_grad': D.grad, 'z_grad': z.grad if z is not None else None, 'dt_bias_grad': dt_bias.grad}
                run(f'ssd|z={int(with_z)}|tokens={n}', ssd_case)

    payload={'results':results,'errors':errors,'contaminated':contaminated,
       'choice_log':choice_log(),'choice_digest':choice_digest(),
       'meta':{'torch':torch.__version__,'triton':triton.__version__,'cuda':torch.version.cuda,
               'gpu':torch.cuda.get_device_name(0),'host_arch':platform.machine(),
               'table_sha256':hashlib.sha256((Path(args.table)/'sm90.json').read_bytes()).hexdigest(),
               'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
               'mamba_deterministic':os.environ.get('MAMBA_DETERMINISTIC'), 'families':families,
               'scope':'63 component shapes; not model training, numerical auto equivalence, or performance acceptance'}}
    out.write_text(json.dumps(payload,indent=1)+'\n')
    return int(bool(errors or contaminated))
if __name__=='__main__': raise SystemExit(main())
