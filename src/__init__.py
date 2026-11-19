# agi — 现有 LIF 实验的共享实现。
# 配置包含历史默认值；展开 Config 后再解释每项实验的行为。
# runner 保留多种历史流程，对齐结果只说明移植一致。
# 原型与历史脚本索引见 experiments/README.md；框架边界见 src/README.md。
# 入口：python scripts/run.py list|show|check|run|results|gaps|align|scan|bench

from .config import Config, NetSpec, EnvSpec, TaskSpec, Plasticity, SignalSpec, ReadoutSpec
from .runner import run
from .store import append, load, summary, seen
from . import experiments

__all__ = [
    "Config", "NetSpec", "EnvSpec", "TaskSpec", "Plasticity", "SignalSpec",
    "ReadoutSpec", "run", "append", "load", "summary", "seen", "experiments",
]
