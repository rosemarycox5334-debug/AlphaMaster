"""
main.py — 多因子训练入口（分组训练）

使用方法：
    python main.py                      # 按 SYMBOL_GROUPS 分组训练（默认推荐）
    python main.py --offline            # 仅使用本地缓存，不连接 MT5
    python main.py --single XAUUSD      # 只训练单个品种
    python main.py --cross-section      # 所有品种一起训练（截面）
    python main.py --group risk         # 只训练 risk 组

分组说明（Config.SYMBOL_GROUPS）：
    forex 组（EURUSD, USDJPY）：外汇，美元方向因子，2品种×11000=22000样本
    risk  组（XAUUSD, US100.cash, US500.cash）：风险资产，3品种×11000=33000样本
"""
import sys, pathlib, json

# 无控制台环境（如 Start-Process -WindowStyle Hidden）下，sys.stdout 可能为 None，
# 导致 tqdm.write 报错。此时重定向到日志文件。
if sys.stdout is None or sys.stderr is None:
    _log_path = r"D:\素材\自动挖因子\training_stdout.log"
    _log_fp = open(_log_path, "a", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = _log_fp
    if sys.stderr is None:
        sys.stderr = _log_fp

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.single_symbol_manager import SingleSymbolDataManager
from model_core.engine import AlphaEngine
from model_core.config import ModelConfig
from model_core.vocab import VOCAB_VERSION


class GroupDataManager:
    """品种分组数据视图，兼容 AlphaEngine 接口（N = 组内品种数）。"""
    def __init__(self, multi_manager, symbols: list[str]):
        self._multi   = multi_manager
        self._symbols = [s for s in symbols if s in multi_manager.symbols]
        self._idxs    = [multi_manager.symbols.index(s) for s in self._symbols]

    @property
    def symbols(self): return list(self._symbols)

    @property
    def raw_dict(self):
        full = self._multi.raw_dict
        return {k: v[self._idxs] for k, v in full.items()}

    @property
    def feat_tensor(self):
        from model_core.features import MT5FeatureEngineer
        return MT5FeatureEngineer.compute_features(self.raw_dict)

    @property
    def target_ret(self):
        return self._multi.target_ret[self._idxs]

    @property
    def bar_time(self):
        return self._multi.bar_time[self._idxs]


def save_group_strategy(engine: AlphaEngine, group_name: str, symbols: list[str]):
    """保存分组策略：group 总文件 + 各品种 best_*.json。"""
    pathlib.Path("strategies").mkdir(exist_ok=True)

    # 分组总文件
    gp = pathlib.Path("strategies") / f"best_group_{group_name}.json"
    gp.write_text(json.dumps({
        "vocab_version": VOCAB_VERSION,
        "group":         group_name,
        "symbols":       symbols,
        "formula":       engine.best_formula,
        "best_score":    engine.best_score,
    }, indent=2))

    # 各品种文件（runner 按品种名加载）
    for sym in symbols:
        sp = pathlib.Path("strategies") / f"best_{sym}.json"
        sp.write_text(json.dumps({
            "vocab_version": VOCAB_VERSION,
            "symbol":        sym,
            "group":         group_name,
            "formula":       engine.best_formula,
            "best_score":    engine.best_score,
            "source":        f"group_{group_name}",
        }, indent=2))

    print(f"  已保存: best_group_{group_name}.json + {len(symbols)} 个品种文件")


def train_group(fetcher, group_name: str, symbols: list[str], offline: bool):
    """训练一个品种组，使用组内独立的 DataManager（不与其他组取时间交集）。

    核心改进：每组单独 load，只对组内品种取时间对齐，避免因 crypto/指数
    交易时间不同导致 forex 组的 12000 根数据被砍到 3498 根。
    """
    print(f"\n{'─'*60}")
    print(f"  [{group_name}] 组: {symbols}")
    print(f"{'─'*60}")

    # 临时覆盖 SYMBOLS 只加载本组品种
    original_symbols = Config.SYMBOLS[:]
    Config.SYMBOLS = [s for s in symbols if s in original_symbols or True]

    try:
        group_mgr = MT5DataManager(fetcher)
        group_mgr.load()
        actual_symbols = group_mgr.symbols
        T = group_mgr.raw_dict["open"].shape[1]
        print(f"  独立加载: {actual_symbols}  T={T} bars")
    except Exception as e:
        print(f"  [错误] 数据加载失败: {e}")
        Config.SYMBOLS = original_symbols
        return None
    finally:
        Config.SYMBOLS = original_symbols  # 恢复原始配置

    if not actual_symbols:
        print(f"  [跳过] 无有效品种")
        return None

    engine = AlphaEngine(data_manager=group_mgr, target_symbol=group_name)
    engine.train()
    save_group_strategy(engine, group_name, actual_symbols)
    return engine


def main():
    cross    = "--cross-section" in sys.argv
    single   = "--single" in sys.argv
    offline  = "--offline" in sys.argv
    grp_only = None
    if "--group" in sys.argv:
        idx      = sys.argv.index("--group")
        grp_only = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
    single_sym = [s for s in sys.argv[1:] if not s.startswith("--")] or None

    mode = "截面" if cross else ("单品种" if single else "分组")
    print(f"{'='*60}")
    print(f"  AlphaGPT 训练 [{mode}模式]" + (" [离线缓存]" if offline else ""))
    print(f"  TRAIN_STEPS={ModelConfig.TRAIN_STEPS}  "
          f"MAX_FORMULA_LEN={ModelConfig.MAX_FORMULA_LEN}  "
          f"BATCH_SIZE={ModelConfig.BATCH_SIZE}")
    print(f"  SYMBOLS={Config.SYMBOLS}")
    print(f"{'='*60}")

    with MT5DataFetcher(offline=offline) as fetcher:
        if single and single_sym:
            # 单品种模式：用全局 multi_mgr
            multi_mgr = MT5DataManager(fetcher)
            multi_mgr.load()
            sym    = single_sym[0]
            s_mgr  = SingleSymbolDataManager(multi_mgr, sym)
            engine = AlphaEngine(data_manager=s_mgr, target_symbol=sym)
            engine.train()

        elif cross:
            multi_mgr = MT5DataManager(fetcher)
            multi_mgr.load()
            engine = AlphaEngine(data_manager=multi_mgr, target_symbol=None)
            engine.train()
            save_group_strategy(engine, "cross_section", multi_mgr.symbols)

        else:
            # 分组训练：每组独立加载，不共享 DataManager
            groups = getattr(Config, "SYMBOL_GROUPS", {
                "forex": ["EURUSD", "USDJPY"],
                "risk":  ["XAUUSD", "US100.cash", "US500.cash"],
            })
            if grp_only:
                groups = {grp_only: groups[grp_only]} if grp_only in groups else groups

            results = {}
            for gname, gsyms in groups.items():
                eng = train_group(fetcher, gname, gsyms, offline)
                if eng:
                    results[gname] = {
                        "score":   eng.best_score,
                        "formula": eng._decode_formula(eng.best_formula),
                    }

            print(f"\n{'='*60}")
            print(f"  分组训练完成")
            print(f"{'='*60}")
            for gname, r in results.items():
                print(f"  [{gname}]: score={r['score']:.4f}")
                print(f"    {r['formula']}")
            print()


if __name__ == "__main__":
    main()
