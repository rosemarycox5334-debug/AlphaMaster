import torch, glob

files = sorted(glob.glob(r'D:\cl\MT5_AlphaGPT\checkpoints\ckpt_metals_comm_step_*.pt'))
# Check every 100 steps
for f in files:
    name = f.split('\\')[-1]
    step_num = int(name.split('step_')[1].split('.')[0])
    if step_num % 100 != 0 and step_num != 20:
        continue
    ckpt = torch.load(f, map_location='cpu', weights_only=False)
    step = ckpt.get('step', 0)
    best = ckpt.get('best_score', 0)
    restarts = ckpt.get('restarts', 0)
    formula = ckpt.get('best_formula', [])
    # check entropy
    entropy = ckpt.get('entropy', None)
    stagnation = ckpt.get('stagnation_steps', None)
    print(f'step={step:4d}  best={best:.4f}  restarts={restarts}  formula={formula}')
