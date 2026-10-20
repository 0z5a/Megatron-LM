"""Fixed-host DP gradient reduce-scatter replay; no allocation-invariance claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
args = parser.parse_args()
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
torch.use_deterministic_algorithms(True)
dist.init_process_group('nccl', device_id=torch.device('cuda', 0))
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 2


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8)
                          .cpu().numpy().tobytes()).hexdigest()


records = []
try:
    for dtype in (torch.bfloat16, torch.float32):
        generator = torch.Generator(device='cuda').manual_seed(1234)
        weight = torch.randn(512, 1024, device='cuda', dtype=dtype,
                             generator=generator, requires_grad=True)
        local = torch.Generator(device='cuda').manual_seed(2345 + rank)
        features = torch.randn(64, 1024, device='cuda', dtype=dtype, generator=local)
        target = torch.randn(64, 512, device='cuda', dtype=dtype, generator=local)
        first = None
        for replay in range(5):
            weight.grad = None
            loss = ((features @ weight.T).float() - target.float()).square().mean()
            loss.backward()
            gradient = weight.grad.detach().flatten().contiguous()
            gathered = [torch.empty_like(gradient) for _ in range(world)]
            dist.all_gather(gathered, gradient)
            expected = (gathered[0].cpu().float() + gathered[1].cpu().float()).to(dtype)
            expected = expected.chunk(world)[rank]
            shard = torch.empty(gradient.numel() // world, device='cuda', dtype=dtype)
            dist.reduce_scatter_tensor(shard, gradient, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            torch.testing.assert_close(shard.cpu(), expected, rtol=0, atol=0)
            state = {'loss':digest(loss), 'gradient':digest(gradient), 'shard':digest(shard)}
            if first is None:
                first = state
            assert state == first, (dtype, replay, state, first)
            records.append({'dtype':str(dtype), 'replay':replay, **state})
    payload = {'rank':rank,'world_size':world,'hostname':socket.gethostname(),
               'scope':'same two physical hosts; independent process restarts, not Slurm allocations',
               'torch':torch.__version__,'cuda':torch.version.cuda,'nccl':torch.cuda.nccl.version(),
               'device':torch.cuda.get_device_name(),'records':records}
    Path(args.out + f'.rank{rank}.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload),flush=True)
finally:
    dist.destroy_process_group()
