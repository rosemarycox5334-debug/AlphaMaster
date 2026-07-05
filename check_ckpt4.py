import torch, glob

f = sorted(glob.glob(r'D:\cl\MT5_AlphaGPT\checkpoints\ckpt_metals_comm_step_*.pt'))[-1]
ckpt = torch.load(f, map_location='cpu', weights_only=False)
print(f'step={ckpt["step"]}  best={ckpt["best_score"]:.4f}  restarts={ckpt.get("restart_count",0)}')
th = ckpt.get('training_history', [])
print(f'training_history length: {len(th)}')
if th:
    print(f'last 20 entries:')
    for entry in th[-20:]:
        print(f'  {entry}')
