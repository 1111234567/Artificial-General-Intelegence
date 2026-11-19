"""在独立任务中复用稀疏 LIF 网络与迹驱动可塑性。"""

from ..config import Config, NetSpec, Plasticity
from ..core import Network

__all__ = ["NetSpec", "Network", "Plasticity", "build_network", "train"]


def build_network(spec: NetSpec, rng, out_idx, *, plasticity: Plasticity | None = None) -> Network:
    """构建网络，并可从 ``Plasticity`` 配置活动迹和 BCM 阈值的时间常数。

    ``out_idx`` 是读出神经元的下标，可为空。``rng`` 用于初始化与后续结构生长。
    每步将一份形状为 ``(spec.N,)`` 的输入电流传给 ``Network.step``，
    放电结果从 ``state[out_idx]`` 读取。调用方按任务时机显式触发网络的
    ``apply_weight``、``apply_bias`` 和拓扑更新方法。
    """
    net = Network(spec, rng, out_idx)
    if plasticity is not None:
        net.tau_trace = plasticity.tau_trace
        net.trace_decay = plasticity.trace_decay
        net.tau_theta = plasticity.tau_theta
    return net


def train(config: Config) -> dict:
    """按已有 Config 执行一轮 LIF 流程，与 src.run 共用执行器和更新时机。"""
    from ..runner import run

    return run(config)
