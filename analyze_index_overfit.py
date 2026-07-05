import torch, json, glob
import numpy as np

# Load index checkpoint
f = sorted(glob.glob(r'D:\cl\MT5_AlphaGPT\checkpoints\ckpt_index_step_*.pt'))[-1]
ckpt = torch.load(f, map_location='cpu', weights_only=False)
step = ckpt["step"]
best = ckpt["best_score"]
restarts = ckpt.get("restart_count", 0)
th = ckpt.get('training_history', {})

print(f'=== INDEX CHECKPOINT ===')
print(f'step={step}  best={best:.4f}  restarts={restarts}')
print()

# Check best_score progression
if isinstance(th, dict) and 'best_score' in th:
    bs = th['best_score']
    # Find when best_score first reached current value
    for i, v in enumerate(bs):
        if abs(v - best) < 0.001:
            print(f'Best score {best:.4f} first reached at step {i}')
            break
    # Show progression every 200 steps
    print(f'\nBest score progression (every 200 steps):')
    for i in range(0, len(bs), 200):
        print(f'  step {i:4d}: best={bs[i]:.4f}')
    print(f'  step {len(bs)-1:4d}: best={bs[-1]:.4f}')

# Check val_score
if 'val_score' in th:
    vs = th['val_score']
    print(f'\nVal score progression (every 200 steps):')
    for i in range(0, len(vs), 200):
        print(f'  step {i:4d}: val={vs[i]:.4f}')
    print(f'  step {len(vs)-1:4d}: val={vs[-1]:.4f}')
    
    # Check if val is declining
    last_100 = vs[-100:]
    first_100 = vs[:100]
    print(f'\nVal score: first 100 avg={np.mean(first_100):.4f}, last 100 avg={np.mean(last_100):.4f}')

# Entropy metrics
for key in ['entropy', 'top1_prob', 'eff_vocab', 'kl_uniform', 'kl_prev', 'ic_mean', 'ic_stability', 'sortino']:
    if key in th:
        arr = th[key]
        print(f'\n{key}: first5={[round(x,4) for x in arr[:5]]}, last5={[round(x,4) for x in arr[-5:]]}')

# Training restarts
for key in ['restart_count']:
    if key in th:
        arr = th[key]
        print(f'\n{key}: last5={arr[-5:]}')
