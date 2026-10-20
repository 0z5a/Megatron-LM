"""Finite CE forward/backward profiling and reference checks, without model claims."""
import argparse
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
parser.add_argument('--tokens', type=int, default=2048)
parser.add_argument('--vocab', type=int, default=32768, help='Vocabulary per TP rank')
parser.add_argument('--steps', type=int, default=20)
args = parser.parse_args()
deterministic = os.environ['DETERMINISM_PERF_MODE'] == 'det'
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
torch.use_deterministic_algorithms(deterministic)
dist.init_process_group('nccl', device_id=torch.device('cuda', torch.cuda.current_device()))
rank, world = dist.get_rank(), dist.get_world_size()
group = dist.group.WORLD
unfused = lambda x, y: vocab_parallel_cross_entropy(x, y, tp_group=group)
fused = lambda x, y: fused_vocab_parallel_cross_entropy(x, y, group)
inputs = []
rows = []

try:
    for dtype in (torch.bfloat16, torch.float32):
        gen = torch.Generator(device='cuda').manual_seed(1234 + rank)
        logits = torch.randn(args.tokens, args.vocab, device='cuda', dtype=dtype, generator=gen)
        labels = torch.randint(0, args.vocab * world, (args.tokens,), device='cuda',
                               generator=torch.Generator(device='cuda').manual_seed(2345))
        parts = [torch.empty_like(logits) for _ in range(world)]
        dist.all_gather(parts, logits)
        full = torch.cat(parts, dim=-1).float().requires_grad_(True)
        expected = F.cross_entropy(full, labels, reduction='none')
        expected.sum().backward()
        reference_grad = full.grad[:, rank * args.vocab:(rank + 1) * args.vocab].clone()
        for name, fn in [('fused', fused), ('unfused', unfused)]:
            value = logits.clone().requires_grad_(True)
            loss = fn(value, labels)
            loss.sum().backward()
            tolerance = dict(rtol=2e-2, atol=1e-2) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(loss, expected, **tolerance)
            torch.testing.assert_close(value.grad.float(), reference_grad, **tolerance)
            row = {'dtype':str(dtype),'implementation':name,
                   'loss_max_abs_error':(loss-expected).abs().max().item(),
                   'gradient_max_abs_error':(value.grad.float()-reference_grad).abs().max().item()}
            for _ in range(8):
                value = logits.clone().requires_grad_(True)
                fn(value, labels).sum().backward()
            inputs.append((name, fn, logits, labels, row))
        del full, expected, reference_grad, parts
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.cudart().cudaProfilerStart()
    for name, fn, logits, labels, row in inputs:
        times = []
        for _ in range(args.steps):
            value = logits.clone().requires_grad_(True)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push('ce.' + name + '.' + str(logits.dtype))
            start.record()
            fn(value, labels).sum().backward()
            end.record()
            torch.cuda.nvtx.range_pop()
            end.synchronize()
            times.append(start.elapsed_time(end))
        rows.append({**row,'milliseconds':times,'median_ms':statistics.median(times)})
    payload = {'rank':rank,'tp':world,'deterministic':deterministic,'tokens':args.tokens,
               'vocab_per_rank':args.vocab,'steps':args.steps,'torch':torch.__version__,
               'nccl':torch.cuda.nccl.version(),'rows':rows,
               'scope':'CE component forward/backward; profile timings are not a model step gate'}
    path = Path(args.out + '.' + os.environ['DETERMINISM_PERF_MODE'] + f'.rank{rank}.json')
    path.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload),flush=True)
    torch.cuda.cudart().cudaProfilerStop()
finally:
    dist.destroy_process_group()
