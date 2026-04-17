import os
import json

def get_arg(args, name, default=None):
    """
    Safely get an attribute from args (replace multiple hasattr checks).
    Usage: get_arg(args, 'get_image', False)
    # MOD 1: 替换散落的 hasattr(args, 'xxx') 检查。
    """
    return getattr(args, name, default)


def safe_float(x, default=float('inf')):
    """
    Convert to float if possible; otherwise return default.
    # MOD 2: 替换重复的 try/except float(...) 代码。
    """
    try:
        return float(x)
    except Exception:
        try:
            return float(x.item())
        except Exception:
            return default


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=4)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)
