# -*- coding: utf-8 -*-
import ast, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

STDLIB = set(sys.stdlib_module_names)
SKIP_DIRS = {"__pycache__", ".git", ".git-rewrite", ".hypothesis", ".pytest_cache",
             "logs", "checkpoints", "backtest_output", "node_modules", ".venv", "venv"}
LOCAL_HINT = {"config", "utils", "strategies", "execution", "model_core", "paper",
              "lord", "web", "data", "data_pipeline", "backtest_viz", "strategy_manager",
              "scripts", "tests"}

third = {}
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
    for f in files:
        if not f.endswith(".py"):
            continue
        path = os.path.join(root, f)
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0 or not node.module:
                    continue
                names = [node.module]
            for n in names:
                top = n.split(".")[0]
                if not top or top in STDLIB or top in LOCAL_HINT:
                    continue
                # 过滤本仓库顶层模块
                if os.path.exists(top + ".py") or os.path.isdir(top):
                    continue
                third.setdefault(top, set()).add(path)

for k in sorted(third):
    print("%-22s %s" % (k, sorted(third[k])[:2]))
