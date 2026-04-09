"""
Vercel entrypoint: must live under api/. Loads main.py by path to avoid import/bundling issues.
"""
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_MAIN_PATH = _ROOT / "main.py"

# Ensure project root is importable for any dynamic imports in main.py
_root_str = str(_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

_spec = importlib.util.spec_from_file_location("sidekick_main", _MAIN_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load application from {_MAIN_PATH}")

_mod = importlib.util.module_from_spec(_spec)
sys.modules["sidekick_main"] = _mod
_spec.loader.exec_module(_mod)
app = _mod.app
