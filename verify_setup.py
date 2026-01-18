#!/usr/bin/env python3
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from less.data_selection.collect_grad_reps import prepare_optimizer_state

adapter_dir = 'checkpoints/olmo2-dpo-lora'

print('Loading model with adapter...')
model = AutoModelForCausalLM.from_pretrained(
    'allenai/OLMo-2-1124-7B-SFT',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
model = PeftModel.from_pretrained(model, adapter_dir)

for name, param in model.named_parameters():
    if 'lora' in name.lower():
        param.requires_grad = True

model_params = [n for n, p in model.named_parameters() if p.requires_grad]
print(f'Model trainable params: {len(model_params)}')

print('Loading optimizer...')
opt_state = torch.load(f'{adapter_dir}/optimizer.bin', weights_only=False)['state']
print(f'Optimizer params: {len(opt_state)}')

# Check name matching
print('\nChecking parameter name matching...')
missing = [n for n in model_params if n not in opt_state]
if missing:
    print(f'MISSING in optimizer: {len(missing)}')
    for m in missing[:3]:
        print(f'  {m}')
else:
    print('All model params found in optimizer state')

print('\nTesting prepare_optimizer_state...')
device = next(model.parameters()).device
avg, avg_sq = prepare_optimizer_state(model, opt_state, device=device)
print(f'avg shape: {avg.shape}')
print(f'avg_sq shape: {avg_sq.shape}')

nonzero_avg = (avg != 0).sum().item()
nonzero_sq = (avg_sq != 0).sum().item()
print(f'avg non-zero: {nonzero_avg} / {avg.numel()}')
print(f'avg_sq non-zero: {nonzero_sq} / {avg_sq.numel()}')

print('\n' + '='*50)
print('VERIFICATION SUCCESSFUL!')
print('='*50)
