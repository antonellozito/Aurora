import importlib.util
from pathlib import Path
import sys


def _compat_candidates():
    root = Path(__file__).resolve().parent
    yield root / "aurora" / "_compat.py"

    for entry in sys.path:
        if not entry:
            continue

        try:
            yield Path(entry).resolve() / "aurora" / "_compat.py"
        except OSError:
            continue


def _load_aurora_compat():
    for compat_path in _compat_candidates():
        if not compat_path.exists():
            continue

        spec = importlib.util.spec_from_file_location(
            "_aurora_runtime_compat", compat_path
        )
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install_runtime_compat()
        return


try:
    _load_aurora_compat()
except Exception:
    pass
