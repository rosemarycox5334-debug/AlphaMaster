import torch, glob, os

files = sorted(glob.glob(r'D:\cl\MT5_AlphaGPT\checkpoints\ckpt_metals_comm_step_*.pt'))
for f in files[-10:]:
    ckpt = torch.load(f, map_location='cpu', weights_only=False)
    step = ckpt.get('step', 0)
    best = ckpt.get('best_score', 0)
    restarts = ckpt.get('restarts', 0)
    formula = ckpt.get('best_formula', [])
    hist = ckpt.get('score_history', [])
    last_scores = [round(s, 3) for s in hist[-5:]] if hist else []
    print(f'step={step:4d}  best={best:.4f}  restarts={restarts}  formula={formula}  recent={last_scores}')
