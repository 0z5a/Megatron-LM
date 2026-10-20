"""Strict native CE through an actual GPT forward/backward/SGD step."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from megatron.training.determinism import apply_determinism_to_args
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
apply_determinism_to_args(SimpleNamespace(tp_comm_overlap=False,cross_entropy_loss_fusion=True,cross_entropy_fusion_impl='native',moe_router_fusion=False,moe_router_aux_loss_fusion=False))
Utils.initialize_model_parallel()
torch.manual_seed(1234);torch.cuda.manual_seed_all(1234);model_parallel_cuda_manual_seed(1234)
cfg=TransformerConfig(num_layers=1,hidden_size=256,num_attention_heads=8,use_cpu_initialization=True,bf16=True,params_dtype=torch.bfloat16,pipeline_dtype=torch.bfloat16,hidden_dropout=0.,attention_dropout=0.,deterministic_mode=True,gradient_accumulation_fusion=False,cross_entropy_loss_fusion=True,cross_entropy_fusion_impl='native')
model=GPTModel(config=cfg,transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),vocab_size=4096,max_sequence_length=256,position_embedding_type='rope').cuda().to(torch.bfloat16).train()
optim=torch.optim.SGD(model.parameters(),lr=1e-4,foreach=False)
tokens=torch.randint(0,4096,(1,256),device='cuda');pos=torch.arange(256,device='cuda').unsqueeze(0);labels=tokens.roll(-1,-1)
def digest(x):return hashlib.sha256(x.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
result={'input':digest(tokens),'initial_parameters':{n:digest(v) for n,v in model.named_parameters()},'losses':[]}
for step in range(4):
 optim.zero_grad(set_to_none=True);loss=model(input_ids=tokens,position_ids=pos,attention_mask=None,labels=labels).float().mean();loss.backward();optim.step();torch.cuda.synchronize();assert torch.isfinite(loss)
 result['losses'].append({'value':loss.item(),'digest':digest(loss)})
result.update(parameters={n:digest(v) for n,v in model.named_parameters()},gradients={n:digest(v.grad) if v.grad is not None else None for n,v in model.named_parameters()},strict=torch.are_deterministic_algorithms_enabled(),fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory)
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');Utils.destroy_model_parallel();print('STRICT_NATIVE_CE_MODEL_PASS')
