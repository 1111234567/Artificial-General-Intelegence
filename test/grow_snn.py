# grow_snn.py - 结构自生长/裁剪最小实验（阶段 B 前置：单体 SA 形态）
#
# 设计（按 Jerry 2026-10-23）：
#   邻域操作 = 基于激活的加减边（大概率，局部）
#   全局操作 = 随机新建连接（小概率，跳出局部邻域）
#   按 SA 一般写法：全局事件用小概率
#
# 动力学与结构更新的时间尺度分层：
#   每步：LIF 前向
#   每 T_s 步：局部结构更新（采样活跃对、衰减、剪除、突触缩放）
#   每 T_g 步：全局结构更新（随机新建）
#   理由：结构可塑性远慢于电活动（生物学：秒~分钟 vs 毫秒）
#
# 多组对照（分离各机制的贡献）：
#   A. 只有局部（无全局、无退火）
#   B. 局部 + 全局（无退火）
#   C. 局部 + 全局 + 退火（完整）
#
# 观测：E(t)、turnover、spike rate、度分布、聚类、分支比

import time
import numpy as np

# ---------------- 参数 ----------------
N           = 1000
DT          = 1.0
TAU         = 20.0
DECAY_V     = float(np.exp(-DT / TAU))
V_TH        = 1.0

W_INIT      = 0.30      # 新建边的初始权重
ETA_POT     = 0.02      # 共激活增强
DECAY_W     = 0.9985    # 每结构步的全局衰减（未被加强的边变弱）
THETA_PRUNE = 0.06      # 低于此值 삭除
W_MAX       = 2.0

T_S         = 20        # 每多少仿真步做一次局部结构更新
T_G         = 2000      # 每多少仿真步做一次全局随机新建
P_GLOBAL    = 6         # 每次全局事件新建多少条随机边
K_SAMPLE    = 200       # 每局部步采样多少活跃对

W_BUDGET    = 12000.0   # 突触预算（总权重和）——超过就突触缩放

OUT_RATE    = 0.004     # 外部输入率


def build_initial(N, E0, rng):
    """过度生产起步：随机 E0 条边。"""
    W = np.zeros((N, N), dtype=np.float32)
    ii = rng.integers(0, N, E0)
    jj = rng.integers(0, N, E0)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    W[jj, ii] = W_INIT * (1.0 + 0.2 * rng.standard_normal(len(ii))).astype(np.float32)
    np.clip(W, 0, None, out=W)
    return W


def run(mode, steps=100000, seed=0, E0=20000, anneal=True, verbose_every=10000):
    rng = np.random.default_rng(seed)
    W = build_initial(N, E0, rng)

    v = np.zeros(N, dtype=np.float32)
    s = np.zeros(N, dtype=np.float32)
    act = np.zeros(N, dtype=np.int32)      # 窗口内累积激活次数

    use_global = mode in ('B', 'C')
    use_anneal = anneal and mode == 'C'

    hist = []
    prev_edge_set = None
    n_sp_prev = 0
    sp_rate_acc = 0.0
    t0 = time.perf_counter()

    for t in range(steps):
        # ---------- 前向：稀疏 bincount ----------
        nz = np.nonzero(W.ravel())[0] if False else None  # (不用，直接稠密乘太贵，改用非零索引)
        # 每次都找非零太贵，改为缓存：每 T_s 步重建一次索引
        if t % T_S == 0 or t == 0:
            pi, qi = np.nonzero(W)      # qi=pre, pi=post
            wv = W[pi, qi]
        I = np.bincount(pi, weights=wv * s[qi], minlength=N).astype(np.float32)
        x = (rng.random(N) < OUT_RATE).astype(np.float32)
        v = v * DECAY_V + I + x
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)

        nsp = int(s.sum())
        act += s.astype(np.int32)
        sp_rate_acc += nsp

        # ---------- 局部结构更新 ----------
        if t > 0 and t % T_S == 0:
            # 温度（退火）：可塑性幅度随进度递减到 5%
            frac = t / steps
            temp = (1.0 - 0.95 * frac) if use_anneal else 1.0

            active = np.nonzero(act > 0)[0]
            if active.size >= 2:
                na = active.size
                ia = rng.integers(0, na, K_SAMPLE)
                ib = rng.integers(0, na, K_SAMPLE)
                pre = active[ia]
                post = active[ib]
                ok = pre != post
                pre, post = pre[ok], post[ok]

                # 已有边 → 增强；不存在 → 新建（概率受温度调制）
                cur = W[post, pre]
                has = cur > 0
                W[post[has], pre[has]] = np.minimum(
                    cur[has] * (1.0 + ETA_POT * temp), W_MAX)
                newp = post[~has]
                newq = pre[~has]
                pick = rng.random(newp.size) < (0.02 * temp)
                W[newp[pick], newq[pick]] = W_INIT

            # 全局衰减 + 剪除（只对已有边，O(E)）
            nzm = W > 0
            W[nzm] *= DECAY_W
            W[W < THETA_PRUNE] = 0.0

            # 突触缩放（预算约束）
            tot = float(W.sum())
            if tot > W_BUDGET:
                W *= (W_BUDGET / tot)

            # 全局随机新建（小概率事件）
            if use_global and t % T_G == 0:
                gi = rng.integers(0, N, P_GLOBAL)
                gj = rng.integers(0, N, P_GLOBAL)
                ok2 = gi != gj
                W[gj[ok2], gi[ok2]] = W_INIT

            act[:] = 0

        # ---------- 观测 ----------
        if verbose_every and t > 0 and t % verbose_every == 0:
            E = int((W > 0).sum())
            # turnover：与上一个观测点的边集差异
            cur_set = set(zip(*np.nonzero(W)))
            if prev_edge_set is None:
                tov = float('nan')
            else:
                inter = len(cur_set & prev_edge_set)
                union = len(cur_set | prev_edge_set)
                tov = 1.0 - (inter / union if union else 1.0)
            prev_edge_set = cur_set

            rate = sp_rate_acc / verbose_every / N
            sp_rate_acc = 0.0

            deg = (W > 0).sum(axis=0)
            hist.append(dict(t=t, E=E, turnover=tov, rate=rate,
                             wsum=float(W.sum()),
                             deg_mean=float(deg.mean()), deg_max=int(deg.max()),
                             deg_nonzero=int((deg > 0).sum())))
            el = time.perf_counter() - t0
            print('  t=%7d  E=%6d  tov=%.4f  rate=%.3f%%  wsum=%9.0f  deg(mean/max/nz)=%.1f/%d/%d  [%.0fs]' % (
                t, E, tov, rate * 100, hist[-1]['wsum'],
                hist[-1]['deg_mean'], hist[-1]['deg_max'], hist[-1]['deg_nonzero'], el))

    el = time.perf_counter() - t0
    E = int((W > 0).sum())
    print('  done: %.1fs  final E=%d  wsum=%.0f' % (el, E, float(W.sum())))
    return hist, W


def main():
    steps = int(1e5)
    print('=' * 100)
    print('结构自生长/裁剪 — 单个体 SA 形态   N=%d  steps=%d' % (N, steps))
    print('局部：共激活增强 + 衰减 + 剪除 + 突触缩放（每 %d 步）' % T_S)
    print('全局：随机新建 %d 条边（每 %d 步，即 %.2f%% 的步）' % (P_GLOBAL, T_G, 100.0 / T_G))
    print('预算：%.0f   初始 E0=20000（过度生产）' % W_BUDGET)
    print('=' * 100)

    for mode, label in (('A', 'A. 只有局部（无全局 / 无退火）'),
                        ('B', 'B. 局部 + 全局（无退火）'),
                        ('C', 'C. 局部 + 全局 + 退火（完整）')):
        print()
        print('-' * 100)
        print(label)
        print('-' * 100)
        run(mode=mode, steps=steps)


if __name__ == '__main__':
    main()
