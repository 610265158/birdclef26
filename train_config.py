import os
import sys
import yaml
from easydict import EasyDict as edict

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
_DEFAULT_YAML = os.path.join(_CONFIG_DIR, 'perch_continuous_extra.yaml')


def _load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _dict_to_edict(d):
    if isinstance(d, dict):
        return edict({k: _dict_to_edict(v) for k, v in d.items()})
    return d


def load_config(yaml_path=None):
    raw = _load_yaml(yaml_path or _DEFAULT_YAML)
    cfg = _dict_to_edict(raw)

    if cfg.TRAIN.lr_scheduler == 'ReduceLROnPlateau':
        cfg.TRAIN.epoch = 100

    if cfg.TRAIN.vis:
        cfg.TRAIN.mix_precision = False

    return cfg


def _resolve_config_path():
    for i, arg in enumerate(sys.argv):
        if arg == '--config' and i + 1 < len(sys.argv):
            p = sys.argv[i + 1]
            if os.path.isabs(p) or os.path.exists(p):
                return p
            return os.path.join(_CONFIG_DIR, p)
    return _DEFAULT_YAML


config = load_config(_resolve_config_path())

from lib.utils.seed_utils import seed_everything

seed_everything(config.SEED)
