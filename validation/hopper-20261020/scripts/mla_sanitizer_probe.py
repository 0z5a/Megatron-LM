"""MLA kernels without NCCL initialization, for memory/race diagnostics."""
import argparse,json
from pathlib import Path
import torch
from megatron.core.tuning import AutotunePolicy,install,choice_log
from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace,fused_mla_rope_kv_split
p=argparse.ArgumentParser();p.add_argument('--table',required=True);p.add_argument('--out',required=True);a=p.parse_args()
install(AutotunePolicy(mode='pinned',table_path=(a.table,),on_miss='error'))
torch.use_deterministic_algorithms(True);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
seq,heads,emb=2048,8,64
cos=torch.randn(seq,1,1,emb,device='cuda',dtype=torch.bfloat16);sin=torch.randn_like(cos)
for _ in range(3):
 q=torch.randn(seq,1,heads,192,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 y=fused_mla_rope_inplace(q.clone(),cos,sin,128,emb);y.float().square().mean().backward()
 kv=torch.randn(seq,1,heads,256,device='cuda',dtype=torch.bfloat16,requires_grad=True);pe=torch.randn(seq,1,1,emb,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 k,v=fused_mla_rope_kv_split(kv,pe,cos,sin,emb,128,128);(k.float().square().mean()+v.float().square().mean()).backward();torch.cuda.synchronize()
 assert all(torch.isfinite(x).all() for x in [y,q.grad,k,v,kv.grad,pe.grad])
Path(a.out).write_text(json.dumps({'completed':True,'choices':choice_log()},indent=2)+'\n')
