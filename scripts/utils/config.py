"""Config loader.

Reads `config.yaml` (relative paths, committed) and overlays
`config.local.yaml` (absolute paths, gitignored) when present, so the
public repo stays portable while my cluster paths live outside git.
"""
import yaml
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / 'config.yaml'
LOCAL_CONFIG_PATH = REPO_ROOT / 'config.local.yaml'


class _AttrDict(dict):
    """Dict with attribute access — lets you write cfg.paths.features_final."""
    def __init__(self, d):
        super().__init__()
        for k, v in d.items():
            self[k] = _AttrDict(v) if isinstance(v, dict) else v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No config key '{name}'")


def _deep_merge(base, overlay):
    """Recursively merge `overlay` into `base` — overlay wins on conflict."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_cfg = None


def load_config(path=None):
    """Load (or reload) config from disk. Returns the cached AttrDict."""
    global _cfg
    p = Path(path) if path else CONFIG_PATH
    with open(p) as f:
        raw = yaml.safe_load(f)
    if not path and LOCAL_CONFIG_PATH.exists():
        with open(LOCAL_CONFIG_PATH) as f:
            local = yaml.safe_load(f) or {}
        raw = _deep_merge(raw, local)
    _cfg = _AttrDict(raw)
    return _cfg


def get_config():
    """Return the cached config, loading on first call."""
    global _cfg
    if _cfg is None:
        load_config()
    return _cfg


CFG = get_config()
