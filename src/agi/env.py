# agi/env.py — 环境。目前只有一种实装：**本机真实文件**。
#
# 2026-10-24 的教训：我曾自己把「环境 = 本机」换成 2D 数学世界，理由是「脉冲网做不了语义任务」。
# 那是真约束，**但应该提请，不是自己定**。后果：`design.md` §9.2「压力由环境提供」不成立。
# 所以这里把 `kind` 写成必填枚举，任何新世界必须显式声明、并存进结果登记。

import io
import os
import shutil
import numpy as np


class LocalFilesEnv(object):
    """一个目录 + state.txt / target.txt。

    观测 = 真实读文件；动作 = 真实写文件（原子写 tmp+replace）；成败 = 真实读回的值。
    可逆：目录可删、可重跑。
    """

    def __init__(self, root, tag, v_max, actions):
        self.dir = os.path.join(root, tag)
        self.v_max = int(v_max)
        self.actions = np.asarray(actions, np.int64)
        self.io_ok = True
        self.n_io = 0
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir, exist_ok=True)
        self._write("state.txt", v_max // 2)
        self._write("target.txt", v_max // 2)

    # ---- 底层 ----
    def _p(self, name):
        return os.path.join(self.dir, name)

    def _write(self, name, v):
        p = self._p(name)
        tmp = p + ".tmp"
        with io.open(tmp, "w", encoding="utf-8") as f:
            f.write(str(int(v)))
        os.replace(tmp, p)
        self.n_io += 1

    def _read(self, name):
        try:
            with io.open(self._p(name), "r", encoding="utf-8") as f:
                v = int(f.read().strip())
            self.n_io += 1
            return v
        except (IOError, OSError, ValueError):
            self.io_ok = False
            return 0

    # ---- 接口 ----
    def set_state(self, v):
        self._write("state.txt", v)

    def set_target(self, v):
        self._write("target.txt", v)

    def get_state(self):
        return self._read("state.txt")

    def get_target(self):
        return self._read("target.txt")

    def act(self, idx):
        """执行一个动作（或 noop）。返回 (新状态, 位移, 动作值)。"""
        if idx < 0 or idx >= len(self.actions):
            return self.get_state(), 0, 0      # noop
        s = self._read("state.txt")
        a = int(self.actions[idx])
        s2 = int(np.clip(s + a, 0, self.v_max))
        self._write("state.txt", s2)
        return s2, s2 - s, a

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def make_env(spec, tag):
    if spec.kind == "local_files":
        return LocalFilesEnv(spec.root, tag, spec.v_max, spec.actions)
    raise ValueError("未知环境类型：%s（必须显式实装）" % spec.kind)
