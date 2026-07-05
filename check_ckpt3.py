import torch, glob

files = sorted(glob.glob(r'D:\cl\MT5_AlphaGPT\checkpoints\ckpt_metals_comm_step_*.pt'))
# Check last 5 checkpoints in detail
for f in files[-5:]:
    ckpt = torch.load(f, map_location='cpu', weights_only=False)
    step = ckpt.get('step', 0)
    best = ckpt.get('best_score', 0)
    restarts = ckpt.get('restarts', 0)
    formula = ckpt.get('best_formula', [])
    # All keys in ckpt
    keys = list(ckpt.keys())
    # Check for any score/entropy history
    score_hist = ckpt.get('score_history', [])
    entropy_hist = ckpt.get('entropy_history', [])
    last_scores = [round(s, 4) for s in score_hist[-10:]] if score_hist else []
    last_entropy = [round(e, 4) for e in entropy_hist[-10:]] if entropy_hist else []
    print(f'step={step}  best={best:.4f}  restarts={restarts}')
    print(f'  keys: {keys}')
    print(f'  last_scores: {last_scores}')
    print(f'  last_entropy: {last_entropy}')
    print()
