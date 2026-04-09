import sys
from pathlib import Path

# Vercel invokes this module from `api/`; project root must be on sys.path for `main`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from main import app  # noqa: E402
