# probe_mem.py - 关键一问：自维持的活动，到不到「刚才那个词」？
#
# probe_homeo 已经证明：有 STD 时，关掉输入后活动稳在 3.42%（2000 步不衰减）。
# 但「稳定」不等于「有记忆」：可能它只是自己转，与刚才看到的东西无关。
#
# 测法（直接检验）：
#   1. 从同一初始状态出发，喂词 c 的脉冲（2 次爆发）
#   2. 关掉一切输入，跑 300 步
#   3. 记录这段时间每个神经元的放电总数 → 得到模式 P_c
#   4. 四个词各跑一次，比四个模式的相似度
#
# 判据（先写死）：
#   两两 cosine 相似度  > 0.9  -> 无区分（只是背景活动，没有记忆）
#   两两 cosine 相似度  < 0.5  -> 有区分（活动携带着刚看到的东西）
#
# 同时跑「无 STD」对照：它应该很快归零，模式为空 -> 相似度无意义。

import numpy as np
from multiprocessing import Pool

N = 1000
N_E = 800
DECAY_V = float(np.exp(-1.0 / 20.0))
V_TH = 1.0
T_REF = 20
W_E = 0.50
W_I_RATIO = 5.0
E0 = 20000
N_IN = 100
BLK = 25
N_IN_FANOUT = 50
IN_SPIKE = 3.0
IN_LO = 0.001
TAU_ADAPT = 100.0
D_THETA = 0.20
TAU_REC = 100.0
U_STD = 0.30

WARM = 300      # 编码后的保持观测步数


def build(rng):
    pre = rng.integers(0, N, E0).astype(np.int32)
    post = rng.integers(0, N, E0).astype(np.int32)
    ok = (pre != post) & (post >= N_IN)
    pre, post = pre[ok], post[ok]
    src = np.repeat(np.arange(N_IN, dtype=np.int32), N_IN_FANOUT)
    dst = rng.integers(N_IN, N_E, N_IN * N_IN_FANOUT).astype(np.int32)
    pre = np.concatenate([pre, src])
    post = np.concatenate([post, dst])
    is_inh = pre >= N_E
    w = np.where(is_inh, -W_E * W_I_RATIO, W_E).astype(np.float32)
    return pre, post, w


def run(args):
    seed, word, use_std, pre_adapt = args
    rng = np.random.default_rng(seed)
    pre, post, w = build(rng)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    ref = np.zeros(N, np.int32)
    theta = np.full(N, V_TH, np.float32)
    x_rec = np.ones(N, np.float32)

    # 阶段 0：预热，让网络进入稳态（每 100 步一次无关脉冲）
    for t in range(pre_adapt):
        xin = np.zeros(N, np.float32)
        if (t % 100) < 2:
            xin[0:25] = IN_SPIKE * 0.6
        xin[0:N_IN] += (rng.random(N_IN) < IN_LO).astype(np.float32) * IN_SPIKE
        if use_std:
            w_eff = np.where(pre >= N_E, w * x_rec[pre], w)
        else:
            w_eff = w
        I = np.bincount(post, weights=w_eff * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY_V + I + xin
        fire = (v > theta) & (ref <= 0)
        s = fire.astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        ref -= 1; np.maximum(ref, 0, out=ref); ref[s > 0] = T_REF
        if fire.any():
            theta[fire] = np.minimum(theta[fire] + D_THETA, 20.0)
        theta += (V_TH - theta) / TAU_ADAPT
        if use_std:
            x_rec += (1.0 - x_rec) / TAU_REC
            ii = np.nonzero(s[N_E:] > 0)[0] + N_E
            if ii.size:
                x_rec[ii] *= (1.0 - U_STD)

    # 阶段 1：编码（只给词的 25 个神经元，2 次爆发）
    for t in range(20):
        xin = np.zeros(N, np.float32)
        if t == 0 or t == 18:
            xin[word * BLK:(word + 1) * BLK] = IN_SPIKE
        xin[0:N_IN] += (rng.random(N_IN) < IN_LO).astype(np.float32) * IN_SPIKE
        if use_std:
            w_eff = np.where(pre >= N_E, w * x_rec[pre], w)
        else:
            w_eff = w
        I = np.bincount(post, weights=w_eff * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY_V + I + xin
        fire = (v > theta) & (ref <= 0)
        s = fire.astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        ref -= 1; np.maximum(ref, 0, out=ref); ref[s > 0] = T_REF
        if fire.any():
            theta[fire] = np.minimum(theta[fire] + D_THETA, 20.0)
        theta += (V_TH - theta) / TAU_ADAPT
        if use_std:
            x_rec += (1.0 - x_rec) / TAU_REC
            ii = np.nonzero(s[N_E:] > 0)[0] + N_E
            if ii.size:
                x_rec[ii] *= (1.0 - U_STD)

    # 阶段 2：完全关掉输入，统计这 300 步里每个神经元放了几次
    cnt = np.zeros(N, np.float64)
    total = 0
    for t in range(WARM):
        xin = np.zeros(N, np.float32)
        if use_std:
            w_eff = np.where(pre >= N_E, w * x_rec[pre], w)
        else:
            w_eff = w
        I = np.bincount(post, weights=w_eff * s[pre], minlength=N).astype(np.float32)
        v = v * DECAY_V + I + xin
        fire = (v > theta) & (ref <= 0)
        s = fire.astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        ref -= 1; np.maximum(ref, 0, out=ref); ref[s > 0] = T_REF
        if fire.any():
            theta[fire] = np.minimum(theta[fire] + D_THETA, 20.0)
        theta += (V_TH - theta) / TAU_ADAPT
        if use_std:
            x_rec += (1.0 - x_rec) / TAU_REC
            ii = np.nonzero(s[N_E:] > 0)[0] + N_E
            if ii.size:
                x_rec[ii] *= (1.0 - U_STD)
        cnt += s
        total += int(s.sum())

    return dict(seed=seed, word=word, std=use_std, cnt=cnt, total=total)


def main():
    nseed = 3
    tasks = [(s, wd, True, 3000) for s in range(nseed) for wd in range(4)]
    tasks += [(s, wd, False, 3000) for s in range(nseed) for wd in range(2)]
    print('=' * 110)
    print('关键实验：自维持的活动，到不到「刚才那个词」？')
    print('  N=%d  保持观测 %d 步  预热 3000 步' % (N, WARM))
    print('=' * 110)
    with Pool(8) as pool:
        res = pool.map(run, tasks)

    for use_std in (True, False):
        sub = [r for r in res if r['std'] == use_std]
        if not sub:
            continue
        print('\n--- %s ---' % ('有 STD' if use_std else '无 STD（对照）'))
        byseed = {}
        for r in sub:
            byseed.setdefault(r['seed'], {})[r['word']] = r

        allsim = []
        for sd, d in byseed.items():
            words = sorted(d)
            print('  seed %d: 各词期间总放电 %s' % (
                sd, ' '.join('词%d=%d' % (w, d[w]['total']) for w in words)))
            if len(words) < 2:
                continue
            sims = []
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    a = d[words[i]]['cnt']
                    b = d[words[j]]['cnt']
                    na = np.linalg.norm(a)
                    nb = np.linalg.norm(b)
                    if na < 1e-9 or nb < 1e-9:
                        sims.append(float('nan'))
                    else:
                        sims.append(float(a @ b / (na * nb)))
            pair = ['词%d-词%d=%.3f' % (words[i], words[j], sims[k])
                    for k, (i, j) in enumerate(
                        [(i, j) for i in range(len(words)) for j in range(i + 1, len(words))])]
            print('    两两相似度: %s' % '  '.join(pair))
            allsim.extend([x for x in sims if x == x])

        if allsim:
            A = np.array(allsim)
            print('  → 平均相似度 %.3f  （>0.9 = 无区分；<0.5 = 有区分）' % A.mean())
            print('  → 判定: %s' % ('无记忆（只是背景活动）' if A.mean() > 0.9 else
                                  ('有区分' if A.mean() < 0.5 else '部分区分')))


if __name__ == '__main__':
    main()
