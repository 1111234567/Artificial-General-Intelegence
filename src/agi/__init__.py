# agi — 统一框架
#
# 2026-10-24 建立。之前的形态：test/ 下 21 个手写脚本，每个都实现一遍本体，
# 每个丢掉前一个的一些东西，而且没有任何地方能回答：
#   ① 哪些组合跑过？② 结果是什么？③ 对照组齐不齐？④ 哪个版本到底有什么？
#
# 现在的形态：
#   config.py     声明（一个实验 = 一个 Config，没有隐式默认）
#   core.py       本体（LIF 网络，唯一实现）
#   env.py        环境（本机真实文件）
#   signal.py     m 的来源与形态（= 学习方式本身）
#   runner.py     跑一个配置，返回一条结构化结果
#   store.py      结果登记（JSONL）
#   experiments.py 唯一声明实验的地方
#
# 入口：python test/run.py <实验名> [种子数]

from .config import Config, NetSpec, EnvSpec, TaskSpec, Plasticity, SignalSpec, ReadoutSpec
from .runner import run
from .store import append, load, summary, seen
from . import experiments

__all__ = [
    "Config", "NetSpec", "EnvSpec", "TaskSpec", "Plasticity", "SignalSpec",
    "ReadoutSpec", "run", "append", "load", "summary", "seen", "experiments",
]
