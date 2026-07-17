"""
train_file.py — 从单个 Parquet K 线文件训练

用法:
    python train_file.py --data-file D:\\K线数据\\AAPL_H1.parquet

文件名格式: {品种}_{周期}.parquet，例如 AAPL_H1.parquet、US30.cash_H1.parquet
"""
from __future__ import annotations

import glob as _glob
import argparse
import json
import pathlib
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.train_logging import configure_train_stdio

configure_train_stdio()

from config import Config
from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine
from model_core.vocab import VOCAB_VERSION


def configure_device(name: str) -> str:
    """Resolve a user-facing device choice and update the runtime config."""
    import torch

    choice = name.strip().lower()
    if choice not in {"auto", "cpu", "cuda"}:
        raise ValueError("设备必须是 auto、cpu 或 cuda")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请确认已安装 CUDA 版 PyTorch 和 NVIDIA 驱动")
    resolved = "cuda" if choice == "auto" and torch.cuda.is_available() else choice
    if resolved == "auto":
        resolved = "cpu"
    ModelConfig.DEVICE = torch.device(resolved)
    return resolved


def train_from_file(
    data_file: str, *, from_scratch: bool = False, device: str = "cpu",
    market: str = "generic",
    train_steps: int | None = None,
    batch_size: int | None = None,
    max_formula_len: int | None = None,
    train_ratio: float = 0.8,
    n_folds: int = 3,
    gap: int = 20,
    seed: int = 2026,
) -> AlphaEngine | None:
    from strategy_manager.market_rules import normalize_market

    market = normalize_market(market)
    if train_steps is not None:
        ModelConfig.TRAIN_STEPS = train_steps
    if batch_size is not None:
        ModelConfig.BATCH_SIZE = batch_size
    if max_formula_len is not None:
        ModelConfig.MAX_FORMULA_LEN = max_formula_len
    ModelConfig.WF_GAP = gap
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = configure_device(device)
    info = inspect_parquet_file(data_file)
    symbol = info["symbol"]
    timeframe = info["timeframe"]

    print(f"\n{'='*60}")
    print(f"  AlphaGPT 文件训练 — {info['filename']}")
    print(f"{'='*60}")
    print(f"  品种: {symbol}")
    print(f"  周期: {timeframe}")
    print(f"  数据: 强制离线 Parquet（不连接 MT5）")
    print(f"  文件: {Path(data_file).resolve()}")
    print(f"  训练步数: {ModelConfig.TRAIN_STEPS}")
    print(f"  训练设备: {resolved_device}")
    print(f"  市场: {'A股（只做多/T+1/涨跌停）' if market == 'a_share' else '通用市场'}")
    print(f"  公式批量: {ModelConfig.BATCH_SIZE}  最大长度: {ModelConfig.MAX_FORMULA_LEN}")
    print(f"  训练集比例: {train_ratio:.0%}  验证折数: {n_folds}  间隔: {gap} bars")
    print(f"  随机种子: {seed}")
    print(f"  K线数: {info['bars']}")
    print(f"  模式: {'重新训练（从头）' if from_scratch else '自动续训'}")
    print(f"{'='*60}")

    try:
        mgr = ParquetDataManager(data_file)
        mgr.load()
        T = mgr.raw_dict["open"].shape[1]
        print(f"  数据加载成功，共 {T} 根K线")
    except Exception as e:
        print(f"  [错误] 数据加载失败: {e}")
        return None

    periods_per_year = 250 if market == "a_share" and timeframe == "D1" else 6240
    engine = AlphaEngine(
        data_manager=mgr, target_symbol=symbol, market=market,
        periods_per_year=periods_per_year,
        train_ratio=train_ratio, n_folds=n_folds,
    )
    engine.timeframe = timeframe
    engine.data_file = str(Path(data_file).resolve())
    engine.mode = "parquet_file"
    engine.train_steps = ModelConfig.TRAIN_STEPS

    ckpt_pattern = str(pathlib.Path("checkpoints") / f"ckpt_{symbol}_step_*.pt")
    ckpt_files = sorted(_glob.glob(ckpt_pattern))
    start_step = 0

    if from_scratch:
        removed = 0
        for p in ckpt_files:
            try:
                pathlib.Path(p).unlink(missing_ok=True)
                removed += 1
            except OSError as e:
                print(f"  [警告] 无法删除检查点 {p}: {e}")
        hist_path = pathlib.Path(f"training_history_{symbol}.json")
        if hist_path.exists():
            try:
                hist_path.unlink()
            except OSError:
                pass
        print(f"  [重新训练] 已清除 {removed} 个检查点，从第 0 步开始")
        # 保留已有最优策略作为分数下限，避免开局弱公式覆盖 strategies/best_*.json
        _seed_best_from_strategy(engine, symbol)
        ckpt_files = []
    elif ckpt_files:
        latest = ckpt_files[-1]
        try:
            start_step = engine.load_checkpoint(latest)
            print(f"  [续训] 从 {latest} 恢复，起始步={start_step}")
        except Exception as e:
            print(f"  [警告] 检查点加载失败: {e}，将从头开始")

    if start_step >= ModelConfig.TRAIN_STEPS:
        print(f"  [完成] {symbol} 已完成全部 {ModelConfig.TRAIN_STEPS} 步，跳过训练")
        _save_strategy(engine, symbol, timeframe, data_file)
        return engine

    if start_step == 0 and not from_scratch:
        hist_path = pathlib.Path(f"training_history_{symbol}.json")
        if hist_path.exists():
            hist_path.unlink()
        print("  [新训] 从第 0 步开始")

    if start_step > 0:
        engine._save_training_history_live()

    engine.train(start_step=start_step)
    _save_strategy(engine, symbol, timeframe, data_file)
    return engine


def _seed_best_from_strategy(engine: AlphaEngine, symbol: str) -> None:
    """把已有 best_{symbol}.json 当作重新训练的分数下限。"""
    path = pathlib.Path("strategies") / f"best_{symbol}.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [警告] 读取已有策略失败: {e}")
        return
    formula = data.get("formula")
    score = data.get("best_score")
    if not formula or score is None:
        return
    try:
        engine.best_formula = [int(t) for t in formula]
        engine.best_score = float(score)
        print(f"  [重新训练] 保留已有最优分数下限={engine.best_score:.4f}，仅更好时才会覆盖策略文件")
    except (TypeError, ValueError) as e:
        print(f"  [警告] 已有策略无法用作下限: {e}")


def _save_strategy(engine: AlphaEngine, symbol: str, timeframe: str, data_file: str) -> None:
    path = pathlib.Path("strategies") / f"best_{symbol}.json"
    path.parent.mkdir(exist_ok=True)
    # 若磁盘上已有更高分，不要用更弱结果覆盖
    if path.exists() and engine.best_formula is not None:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            old_score = old.get("best_score")
            if old_score is not None and float(old_score) > float(engine.best_score):
                print(
                    f"  [策略] 保留磁盘更优结果 {float(old_score):.4f} "
                    f"> 本次 {float(engine.best_score):.4f}，未覆盖 {path}"
                )
                merged = dict(old)
                for key, val in (
                    ("timeframe", timeframe),
                    ("data_file", str(Path(data_file).resolve())),
                    ("mode", "parquet_file"),
                    ("train_steps", ModelConfig.TRAIN_STEPS),
                    ("market", engine.market),
                ):
                    if val is not None and not merged.get(key):
                        merged[key] = val
                if merged != old:
                    path.write_text(
                        json.dumps(merged, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"  [策略] 已补全数据路径等元数据: {path}")
                return
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    data = {
        "vocab_version": VOCAB_VERSION,
        "symbol": symbol,
        "timeframe": timeframe,
        "data_file": str(Path(data_file).resolve()),
        "mode": "parquet_file",
        "formula": engine.best_formula,
        "formula_decoded": engine._decode_formula(engine.best_formula)
        if engine.best_formula
        else None,
        "best_score": engine.best_score,
        "train_steps": ModelConfig.TRAIN_STEPS,
        "market": engine.market,
        "training_config": {
            "train_steps": ModelConfig.TRAIN_STEPS,
            "batch_size": ModelConfig.BATCH_SIZE,
            "max_formula_len": ModelConfig.MAX_FORMULA_LEN,
            "train_ratio": engine.train_ratio,
            "n_folds": engine.n_folds,
            "gap": ModelConfig.WF_GAP,
        },
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  策略已保存: {path}")


if __name__ == "__main__":
    ModelConfig.REWARD_MODE = "ftmo"
    parser = argparse.ArgumentParser(description="从单个 Parquet K 线文件训练")
    parser.add_argument("--data-file", required=True, help="{品种}_{周期}.parquet 文件")
    parser.add_argument("--from-scratch", action="store_true", help="清除检查点并重新训练")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cpu",
        help="训练设备；auto 在 CUDA 可用时优先 GPU（默认 cpu）",
    )
    parser.add_argument(
        "--market", choices=("generic", "a_share"), default="generic",
        help="市场规则；a_share 启用只做多、T+1 和涨跌停限制",
    )
    parser.add_argument("--train-steps", type=int, default=9000)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--max-formula-len", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--gap", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    checks = [
        (1 <= args.train_steps <= 100_000, "train-steps 必须为 1～100000"),
        (16 <= args.batch_size <= 1024, "batch-size 必须为 16～1024"),
        (2 <= args.max_formula_len <= 20, "max-formula-len 必须为 2～20"),
        (0.5 <= args.train_ratio <= 0.95, "train-ratio 必须为 0.50～0.95"),
        (1 <= args.folds <= 10, "folds 必须为 1～10"),
        (0 <= args.gap <= 1000, "gap 必须为 0～1000"),
    ]
    for ok, message in checks:
        if not ok:
            parser.error(message)

    data_file = args.data_file
    from_scratch = args.from_scratch
    t0 = time.time()
    try:
        eng = train_from_file(
            data_file, from_scratch=from_scratch, device=args.device, market=args.market,
            train_steps=args.train_steps, batch_size=args.batch_size,
            max_formula_len=args.max_formula_len, train_ratio=args.train_ratio,
            n_folds=args.folds, gap=args.gap, seed=args.seed
        )
    except (ValueError, RuntimeError) as exc:
        print(f"错误: {exc}")
        sys.exit(2)
    elapsed = time.time() - t0

    if eng:
        sym = eng.target_symbol or "?"
        print(f"\n<<< [{sym}] 训练完成: 最优分数={eng.best_score:.4f}，耗时 {elapsed/3600:.2f} 小时")
        if eng.best_formula:
            print(f"    {eng._decode_formula(eng.best_formula)}")
    else:
        print("\n<<< 训练失败")
        sys.exit(1)
