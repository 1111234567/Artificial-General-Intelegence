# grow_snn2.py - 修正版
#
# v1 两次失败（2026-10-23）：
#   前 3 万步 rate=99.4%（饱和全开）-> 之后萌到 0.3%（快死）
#   退火组最后 E=662 / 648 个神经元无入边
#
# 四个硬伤：
#   1) 稳态写错了：我做的是「权重预算」稳态，不是「活动率」稳态。
#      生物学（Turrigiano 突触缩放）是按放电率调的，不按权重总量调。
#   2) 初始条件过载：E0=20000 x W_INIT=0.3 -> 入输入 ~6 >> V_th=1 -> 必然饱和。
#   3) 温度只作用于「生长」不作用于「削减」-> 退火 = 单向萎缩。
#   4) 局部操作采样覆盖度太低（200 对/窗口 vs 1.2 万条边）-> 绝大多数边永不被强化，只跌。
#
# 修正：
#   1) 活动率稳态：rate > target -> 全局缩放权重向下，反之向上
#   2) 稀疏起步 E0=8000 / W_INIT=0.10
#   3) 温度同时作用于生与削
#   4) 局部操作精确计算（活跃对全算）——这依赖稀疏：rate 1% 时只有 ~10 个活跃神经元
#   5) 剪除阈值相对化（相对该神经元入权重均值）

import time
import numpy as np

N            = 1000
DT           = 1.0
TAU          = 20.0
DECAY_V      = float(np.exp(-DT / TAU))
V_TH         = 1.0

W_INIT       = 0.10
ETA_POT      = 0.05
DECAY_W      = 0.995
W_MAX        = 1.0
PRUNE_REL    = 0.35      # 剪除：低于该神经元入权重均值的这个比例
THETA_ABS    = 0.012     # 绝对下限

T_S          = 20
T_G          = 2000
P_GLOBAL     = 6

TARGET_RATE  = 0.02      # 目标放电率 2%
K_HOME       = 0.35      # 稳态比例因子

OUT_RATE     = 0.004


def build_initial(N, E0, rng):
    W = np.zeros((N, N), dtype=np.float32)
    ii = rng.integers(0, N, E0)
    jj = rng.integers(0, N, E0)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    W[jj, ii] = W_INIT * (1.0 + 0.15 * rng.standard_normal(len(ii))).astype(np.float32)
    np.clip(W, 0.001, None, out=W)
    return W


def run(mode, steps=100000, seed=0, E0=8000, verbose_every=10000):
    rng = np.random.default_rng(seed)
    W = build_initial(N, E0, rng)
    v = np.zeros(N, dtype=np.float32)
    s = np.zeros(N, dtype=np.float32)

    use_global = mode in ('B', 'C')
    use_anneal = mode == 'C'

    pi = qi = wv = None
    t0 = time.perf_counter()
    sp_acc = 0
    prev_set = None

    for t in range(steps):
        if t % T_S == 0:
            pi, qi = np.nonzero(W)
            wv = W[pi, qi]

        I = np.bincount(pi, weights=wv * s[qi], minlength=N).astype(np.float32)
        x = (rng.random(N) < OUT_RATE).astype(np.float32)
        v = v * DECAY_V + I + x
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        nsp = int(s.sum())
        sp_acc += nsp

        # ---------- 结构更新 ----------
        if t > 0 and t % T_S == 0:
            frac = t / steps
            temp = (1.0 - 0.93 * frac) if use_anneal else 1.0

            active = np.nonzero(s > 0)[0]

            # ---- 局部：共激活精确计算（活跃对全算）----
            if active.size >= 2:
                pre = np.repeat(active, active.size)
                post = np.tile(active, active.size)
                ok = pre != post
                pre, post = pre[ok], post[ok]
                cur = W[post, pre]
                has = cur > 0
                # 已有 -> 增强
                W[post[has], pre[has]] = np.minimum(
                    cur[has] + ETA_POT * temp * (W_MAX - cur[has]), W_MAX)
                # 无 -> 新建（受温度调制）
                newp, newq = post[~has], pre[~has]
                pick = rng.random(newp.size) < (0.15 * temp)
                if pick.any():
                    W[newp[pick], newq[pick]] = W_INIT

            # ---- 衰减 + 相对剪除 ----
            big = W > 0
            W[big] *= (1.0 - (1.0 - DECAY_W) * temp)
            W[W < THETA_ABS] = 0.0
            # 相对剪除：低于该 post 入权重均值的 PRUNE_REL
            colsum = W.sum(axis=0)
            colcnt = big.sum(axis=0)
            colmean = np.where(colcnt > 0, colsum / np.maximum(colcnt, 1), 0.0).astype(np.float32)
            thresh = np.maximum(colmean[None, :] * PRUNE_REL, THETA_ABS)
            W[W < thresh] = 0.0

            # ---- 活动率稳态（Turrigiano 式缩放）----
            rate_win = sp_acc / (T_S * N)
            sp_acc = 0
            err = rate_win - TARGET_RATE
            scale = 1.0 - K_HOME * np.clip(err / max(TARGET_RATE, 1e-6), -1.0, 1.0) * 0.25
            W[W > 0] *= scale

            # ---- 全局随机新建（小概率）----
            if use_global and t % T_G == 0:
                gi = rng.integers(0, N, P_GLOBAL)
                gj = rng.integers(0, N, P_GLOBAL)
                ok2 = gi != gj
                W[gj[ok2], gi[ok2]] = W_INIT

        # ---------- 观测 ----------
        if verbose_every and t > 0 and t % verbose_every == 0:
            E = int((W > 0).sum())
            cur_set = set(zip(*np.nonzero(W)))
            if prev_set is None:
                tov = float('nan')
            else:
                inter = len(cur_set & prev_set)
                union = len(cur_set | prev_set)
                tov = 1.0 - (inter / union if union else 1.0)
            prev_set = cur_set

            # 这个窗口的真实放电率
            rate_obs = nsp / N
            deg = (W > 0).sum(axis=0)
            el = time.perf_counter() - t0
            print('  t=%7d  E=%6d  tov=%.3f  rate=%6.2f%%  wsum=%8.0f  deg(mean/max/nz)=%.1f/%d/%d  [%.0fs]' % (
                t, E, tov, rate_obs * 100, float(W.sum()),
                deg.mean(), deg.max(), int((deg > 0).sum()), el))

    el = time.perf_counter() - t0
    E = int((W > 0).sum())
    deg = (W > 0).sum(axis=0)
    print('  done: %.1fs  final E=%d  wsum=%.0f  nz_neurons=%d' % (
        el, E, float(W.sum()), int((deg > 0).sum())))
    return W


def main():
    steps = int(1e5)
    print('=' * 104)
    print('结构自生长/裁剪 v2（修正稳态）  N=%d  steps=%d  目标放电率=%.0f%%' % (
        N, steps, TARGET_RATE * 100))
    print('=' * 104)
    for mode, label in (('A', 'A. 只有局部'),
                        ('B', 'B. 局部 + 全局'),
                        ('C', 'C. 局部 + 全局 + 退火')):
        print()
        print('-' * 104)
        print(label)
        print('-' * 104)
        run(mode=mode, steps=steps)


if __name__ == '__main__':
    main()
