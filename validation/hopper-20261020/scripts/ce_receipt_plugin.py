"""Observe the existing CE replay assertion's returned tensors for cold-run comparison."""
import hashlib
import json
import os
from pathlib import Path

current_case = None

def pytest_runtest_setup(item):
    global current_case
    current_case = item.nodeid

def pytest_collection_modifyitems(items):
    for module in {item.module for item in items}:
        if not module.__name__.endswith('test_fused_activations'):
            continue
        original = module.assert_replays_bit_exact
        def observed(*args, **kwargs):
            outputs, gradients = original(*args, **kwargs)
            if kwargs.get('what') != 'fused_vocab_parallel_cross_entropy':
                return outputs, gradients
            import torch
            import torch.distributed as dist
            from megatron.core import parallel_state
            def digest(t):
                raw = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
                return hashlib.sha256(raw).hexdigest()
            group = parallel_state.get_tensor_model_parallel_group()
            rank = dist.get_rank()
            record = {'case':current_case,'rank':rank,'tp_rank':group.rank(),'tp_size':group.size(),
                'torch_deterministic':torch.are_deterministic_algorithms_enabled(),
                'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
                'contention':kwargs.get('contention',False),'replays':kwargs.get('replays'),
                'inputs':[digest(t) for t in args[1]],
                'outputs':{k:digest(v) for k,v in outputs.items()},
                'gradients':{k:digest(v) for k,v in gradients.items()},
                'torch':torch.__version__,'cuda':torch.version.cuda,'nccl':torch.cuda.nccl.version()}
            path = Path(os.environ['CE_RECEIPT_FILE']+f'.rank{rank}.jsonl')
            with path.open('a') as f:
                f.write(json.dumps(record,sort_keys=True)+'\n')
            return outputs, gradients
        module.assert_replays_bit_exact = observed
