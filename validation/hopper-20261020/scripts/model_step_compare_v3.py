"""Bounded synthetic full-model optimizer steps for sm90 table validation.

Measures GPT/HybridModel forward, backward, and SGD on fixed token batches.
This is a toy-model benchmark, not a convergence or production recipe result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--family', choices=['mla', 'mhc', 'ssd'], required=True)
parser.add_argument('--mode', choices=['auto', 'table', 'cheapest'], required=True)
parser.add_argument('--table', required=True)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--warmup', type=int, default=4)
args = parser.parse_args()
sys.path.insert(0, args.repo)
os.environ.pop('MAMBA_DETERMINISTIC', None)
import torch
import triton
from megatron.core.tuning import AutotunePolicy, install, choice_log, choice_digest

# Keep the full candidate lists before selecting the ordered-reduction path.
import megatron.core.fusions.fused_mla_yarn_rope_apply
import megatron.core.fusions.fused_mhc_kernels
import mamba_ssm.ops.triton.ssd_combined
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.transformer_config import TransformerConfig, MLATransformerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
policy = AutotunePolicy(
    mode='record' if args.mode == 'auto' else 'pinned',
    record_path=str(out.with_suffix('.choices')) if args.mode == 'auto' else None,
    table_path=(args.table,) if args.mode == 'table' else (),
    on_miss='error' if args.mode == 'table' else 'min_cost',
    enumerate_autotuners=True,
)
install(policy)
torch.use_deterministic_algorithms(True)
os.environ['MAMBA_DETERMINISTIC'] = '1'
Utils.initialize_model_parallel()
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
model_parallel_cuda_manual_seed(1234)
seq, vocab = 2048, 512
common = dict(
    num_layers=1, hidden_size=512, num_attention_heads=8, use_cpu_initialization=True,
    bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
    hidden_dropout=0.0, attention_dropout=0.0, deterministic_mode=True,
    gradient_accumulation_fusion=False, normalization='RMSNorm', add_bias_linear=False,
)
if args.family == 'mla':
    common.update(multi_latent_attention=True, q_lora_rank=512, kv_lora_rank=128,
                  qk_head_dim=128, qk_pos_emb_head_dim=64, v_head_dim=128,
                  apply_rope_fusion=True)
    cfg = MLATransformerConfig(**common)
    model = GPTModel(config=cfg,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(multi_latent_attention=True),
        vocab_size=vocab, max_sequence_length=seq, position_embedding_type='rope')
else:
    if args.family == 'mhc':
        common.update(num_layers=2, hidden_size=2688, num_attention_heads=32,
                      ffn_hidden_size=7168, enable_mhc_connections=True,
                      mhc_num_residual_streams=4, use_fused_mhc=True, mhc_fused_backend='triton')
        pattern = '--'
    else:
        common.update(hidden_size=4096, num_attention_heads=32, mamba_state_dim=128,
                      mamba_head_dim=64, mamba_num_groups=8)
        pattern = 'M'
    cfg = TransformerConfig(**common)
    model = HybridModel(config=cfg, hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=vocab, max_sequence_length=seq, hybrid_layer_pattern=pattern,
        position_embedding_type='rope')
model = model.cuda().to(torch.bfloat16).train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, foreach=False)
tokens = torch.randint(0, vocab, (1, seq), device='cuda')
positions = torch.arange(seq, device='cuda').unsqueeze(0)
labels = torch.roll(tokens, shifts=-1, dims=-1)

def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

records = []
try:
    for step in range(args.warmup + args.steps):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        optimizer.zero_grad(set_to_none=True)
        losses = model(input_ids=tokens, position_ids=positions, attention_mask=None, labels=labels)
        loss = losses.float().mean()
        loss.backward()
        optimizer.step()
        end.record()
        torch.cuda.synchronize()
        assert torch.isfinite(loss), 'non-finite loss'
        records.append({'step':step,'warmup':step < args.warmup,
                        'milliseconds':start.elapsed_time(end),'loss':loss.item(),
                        'loss_digest':digest(loss)})
        out.with_suffix('.partial.json').write_text(json.dumps({'steps':records},indent=2))
        print(args.family,args.mode,records[-1],flush=True)
    parameter_digests = {name:digest(p) for name,p in model.named_parameters()}
    gradient_digests = {name:digest(p.grad) if p.grad is not None else None
                        for name,p in model.named_parameters()}
    payload = {'family':args.family,'mode':args.mode,'steps':records,
        'median_ms':statistics.median(x['milliseconds'] for x in records if not x['warmup']),
        'parameter_digests':parameter_digests,'gradient_digests':gradient_digests,
        'choices':choice_log(),'choice_digest':choice_digest(),
        'meta':{'torch':torch.__version__,'triton':triton.__version__,'cuda':torch.version.cuda,
                'device':torch.cuda.get_device_name(0),'seq':seq,'vocab':vocab,'seed':1234,
                'family_config':{k:str(v) for k,v in common.items()},
                'table_sha256':hashlib.sha256((Path(args.table)/'sm90.json').read_bytes()).hexdigest(),
                'scope':'synthetic full-model forward/backward/SGD; no convergence claim'}}
    out.write_text(json.dumps(payload,indent=2)+'\n')
finally:
    Utils.destroy_model_parallel()
