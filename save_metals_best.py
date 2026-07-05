"""save_metals_best.py — 保存 metals_comm 当前 checkpoint 中的最优因子"""
import sys, json, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION

ckpts = sorted(Path('checkpoints').glob('ckpt_metals_comm_step_*.pt'))
if not ckpts:
    print("无 metals_comm checkpoint"); sys.exit(1)

latest = ckpts[-1]
ckpt = torch.load(latest, map_location='cpu', weights_only=False)

formula = ckpt['best_formula']
score = float(ckpt['best_score'])
step = int(ckpt['step'])

out = {
    "group": "metals_comm",
    "symbol_group": "metals_comm",
    "vocab_version": VOCAB_VERSION,
    "formula": formula,
    "best_score": score,
    "step": step,
    "saved_at": str(Path(__file__).parent),
    "mode": "ftmo"
}

out_path = Path('strategies/best_metals_comm.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

readable = " -> ".join(FORMULA_VOCAB.token_names[t] for t in formula)
print(f"已保存: {out_path}")
print(f"  step={step}  score={score:.4f}")
print(f"  {readable}")
