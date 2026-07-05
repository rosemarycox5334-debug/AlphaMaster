import torch, glob, json

f = sorted(glob.glob(r'D:\cl\MT5_AlphaGPT\checkpoints\ckpt_metals_comm_step_*.pt'))[-1]
ckpt = torch.load(f, map_location='cpu', weights_only=False)
step = ckpt["step"]
best = ckpt["best_score"]
restarts = ckpt.get("restart_count", 0)
th = ckpt.get('training_history', {})

print(f'step={step}  best={best:.4f}  restarts={restarts}')
print(f'training_history type: {type(th)}')

if isinstance(th, dict):
    # Print all keys and their lengths
    for k, v in th.items():
        if isinstance(v, list):
            print(f'  {k}: len={len(v)}, last5={[round(x,4) if isinstance(x,float) else x for x in v[-5:]]}')
        else:
            print(f'  {k}: {v}')
elif isinstance(th, list):
    print(f'  len={len(th)}')
    for entry in th[-10:]:
        print(f'  {entry}')
