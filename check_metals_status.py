import torch, glob

f = sorted(glob.glob(r'D:\cl\MT5_AlphaGPT\checkpoints\ckpt_metals_comm_step_*.pt'))[-1]
ckpt = torch.load(f, map_location='cpu', weights_only=False)
th = ckpt.get('training_history', {})
step = ckpt['step']
best = ckpt['best_score']
restarts = ckpt.get('restart_count', 0)

print(f'step={step}  best={best:.4f}  restarts={restarts}')
print()

# best_score progression every 100 steps
bs = th.get('best_score', [])
vs = th.get('val_score', [])
ent = th.get('entropy', [])
top1 = th.get('top1_prob', [])
effv = th.get('eff_vocab', [])

print('=== best_score / val_score / entropy every 100 steps ===')
for i in range(0, len(bs), 100):
    v = vs[i] if i < len(vs) else 0
    e = ent[i] if i < len(ent) else 0
    t = top1[i] if i < len(top1) else 0
    ev = effv[i] if i < len(effv) else 0
    print(f'  step {i:4d}: best={bs[i]:.4f}  val={v:.4f}  entropy={e:.4f}  top1={t:.4f}  effV={ev:.4f}')
print(f'  step {len(bs)-1:4d}: best={bs[-1]:.4f}  val={vs[-1]:.4f}  entropy={ent[-1]:.4f}  top1={top1[-1]:.4f}  effV={effv[-1]:.4f}')

# Stagnation
for i in range(len(bs)-1, -1, -1):
    if abs(bs[i] - best) > 0.001:
        print(f'\nStagnation: best_score={best:.4f} unchanged since step {i+1} ({len(bs)-1-i} steps ago)')
        break

# val_score trend last 200 vs first 200
if len(vs) > 200:
    import numpy as np
    print(f'\nval_score first 200 avg: {np.mean(vs[:200]):.4f}')
    print(f'val_score last 200 avg: {np.mean(vs[-200:]):.4f}')
