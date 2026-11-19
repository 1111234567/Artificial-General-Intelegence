# -*- coding: utf-8 -*-
"""历史草案格式报告：标题编号 / 交叉引用 / 表格形态 / 编码 / 行尾。

默认使用 2026-10-25 的归档三卷，不检查现行研究结论。
"""
import io, re, sys, collections
from pathlib import Path

# 用法：python check_draft.py [目标文件] [其它卷 ...]
#   第一个参数 = 本次要检查的文件；其余卷只作为交叉引用的底簿（章节号全项目唯一）。
#   省略参数时检查归档 design 卷，其余两卷作为交叉引用底簿。
ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = [ROOT / "docs" / "archived" / (name + "-2026-10-25.md")
            for name in ("design", "experiments", "reference")]
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
ALL_PATHS = [(ROOT / a).resolve() for a in ARGS] if ARGS else DEFAULTS
P = ALL_PATHS[0]
print("历史格式报告；不能据此判断现行假设或机制是否成立。")
raw = io.open(P, "r", encoding="utf-8", newline="").read()
CR = chr(13)
LF = chr(10)
crlf = raw.count(CR + LF)
lf_only = raw.count(LF) - crlf
bare_cr = raw.count(CR) - crlf
lines = raw.splitlines()
print("file:", P)
print("lines:", len(lines), " CRLF:", crlf, " bare-LF:", lf_only, " bare-CR:", bare_cr)

# ---------- 1. 标题结构 ----------
heads = []          # (lineno, level, text)
in_fence = False
for i, ln in enumerate(lines, 1):
    if ln.lstrip().startswith("```"):
        in_fence = not in_fence
        continue
    if in_fence:
        continue
    m = re.match(r"^(#{1,4})\s+(.*)$", ln)
    if m:
        heads.append((i, len(m.group(1)), m.group(2).rstrip()))

print("\n=== H2 章节 ===")
for no, lv, t in heads:
    if lv == 2:
        print("  %5d  %s" % (no, t))

# ---------- 2. 编号连续性 ----------
# 全项目章节编号（本卷 + 其它卷）：供连续性、悬空引用两处共用
exist_all = set()
for _p in ALL_PATHS:
    _txt = io.open(_p, "r", encoding="utf-8", newline="").read()
    _fence = False
    for _ln in _txt.splitlines():
        if _ln.lstrip().startswith("```"):
            _fence = not _fence
            continue
        if _fence:
            continue
        _m = re.match(r"^(#{2,4}) +([0-9]+)(?:[.]([0-9]+))?", _ln)
        if not _m:
            continue
        _n1 = int(_m.group(2))
        _n2 = int(_m.group(3)) if _m.group(3) else None
        if len(_m.group(1)) == 2:
            exist_all.add((_n1, None))
        elif _n2 is not None:
            exist_all.add((_n1, _n2))

h2_nums = []
for no, lv, t in heads:
    if lv == 2:
        m = re.match(r"^([0-9]+)[.]", t)
        h2_nums.append(int(m.group(1)) if m else None)
print("\nH2 编号序列:", h2_nums)
bad = [n for n in h2_nums if n is None]
if bad:
    print("  注: 无编号 H2 %d 个（文档头部章节，正常）" % len(bad))
seq = [n for n in h2_nums if n is not None]
if not seq:
    print("  OK 无编号 H2（本文档不用编号体系）")
    expect = []
    gaps = []
else:
    expect = list(range(seq[0], seq[0] + len(seq)))
if seq and seq != expect:
    gaps = [n for n in range(seq[0], max(seq) + 1) if n not in seq]
    outside = [n for n in gaps if (n, None) not in exist_all]
    if gaps and not outside:
        print("  OK 编号连续（缺的 %s 在其它卷，已用底簿补齐）" % gaps)
    else:
        print("  !! 编号不连续:", seq, " 缺:", gaps, " 其中全项目都没有的:", outside)
elif seq:
    print("  OK 编号连续")

# ---------- 3. H3 编号与归属 ----------
h3 = [(no, t) for no, lv, t in heads if lv == 3]
cur_h2 = None
sections = collections.defaultdict(list)
for no, lv, t in heads:
    if lv == 2:
        m = re.match(r"^(\d+)\.", t)
        cur_h2 = int(m.group(1)) if m else -1
    elif lv == 3 and cur_h2 is not None:
        m = re.match(r"^(\d+)\.(\d+)", t)
        sections[cur_h2].append(int(m.group(2)) if m else None)
print("\n=== H3 编号检查 ===")
for k in sorted(sections):
    v = sections[k]
    nums = [x for x in v if x is not None]
    unnum = len(v) - len(nums)
    flag = ""
    if nums and nums != list(range(nums[0], nums[0] + len(nums))):
        flag = "  !! 不连续"
    if unnum:
        flag += "  无编号:%d" % unnum
    print("  §%d: %s%s" % (k, nums, flag))

# ---------- 4. 交叉引用 ----------
refs = collections.Counter()
for i, ln in enumerate(lines, 1):
    for m in re.finditer(r"§(\d+)(?:\.(\d+))?", ln):
        refs[(int(m.group(1)), int(m.group(2)) if m.group(2) else None)] += 1

exist = set()
for no, lv, t in heads:
    m2 = re.match(r"^(\d+)\.\s", t)
    if lv == 2 and m2:
        exist.add((int(m2.group(1)), None))
    m3 = re.match(r"^(\d+)\.(\d+)", t)
    if lv == 3 and m3:
        exist.add((int(m3.group(1)), int(m3.group(2))))
exist_top = {k[0] for k in exist}

exist_all = exist_all | exist

print("\n=== 悬空引用（指向不存在的章节）===")
missed = []
crossed = []
for (a, b), c in sorted(refs.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
    label = ("§%d" % a) if b is None else ("§%d.%d" % (a, b))
    if (a, b) in exist:
        continue
    if (a, b) in exist_all:
        crossed.append((label, c))
        continue
    missed.append((label, c))
if missed:
    for t, c in missed:
        print("  MISS  %-10s x%d" % (t, c))
else:
    print("  OK 无悬空引用")
if crossed:
    print("  （另 %d 个引用指向其它卷，属正常；本条只查本卷）" % len(crossed))

# ---------- 5. 表格形态 ----------
print("\n=== 表格列数异常行 ===")
in_fence = False
bad_tbl = 0
for i, ln in enumerate(lines, 1):
    if ln.lstrip().startswith("```"):
        in_fence = not in_fence
        continue
    if in_fence:
        continue
    s = ln.strip()
    if s.startswith("|") and s.endswith("|") and s.count("|") >= 2:
        # 去掉代码跨度里的 |
        t = re.sub(r"`[^`]*`", "", s)
        n = t.count("|")
        # 相邻行比较
        bad_tbl += 0
# 按块比较
blocks = []
cur = []
in_fence = False
for i, ln in enumerate(lines, 1):
    if ln.lstrip().startswith("```"):
        in_fence = not in_fence
    s = ln.strip()
    istbl = (not in_fence) and s.startswith("|") and s.endswith("|")
    if istbl:
        cur.append((i, len(re.sub(r"`[^`]*`", "", s).split("|"))))
    else:
        if cur:
            blocks.append(cur)
            cur = []
if cur:
    blocks.append(cur)
for b in blocks:
    counts = collections.Counter(c for _, c in b)
    if len(counts) > 1:
        print("  行 %d-%d: 列数分布 %s" % (b[0][0], b[-1][0], dict(counts)))
        for no, c in b:
            if c != counts.most_common(1)[0][0]:
                print("      %5d  %d 列: %s" % (no, c, lines[no-1].strip()[:80]))
        bad_tbl += 1
if not bad_tbl:
    print("  OK 表格列数一致")

# ---------- 6. 乱码 / 控制字符 ----------
print("\n=== 字符异常 ===")
mojibake = ["\u9239", "\u6ebe", "\u6522", "\u65ba"]
hits = []
for i, ln in enumerate(lines, 1):
    for ch in mojibake:
        if ch in ln:
            hits.append((i, ch, ln.strip()[:60]))
if hits:
    for h in hits:
        print("  %5d  %r  %s" % h)
else:
    print("  OK 无 mojibake 特征字")
fffd = [(i, ln.strip()[:60]) for i, ln in enumerate(lines, 1) if "\ufffd" in ln]
print("  U+FFFD:", len(fffd))
for x in fffd[:5]:
    print("    ", x)
ctrl = [(i, repr(ln[:50])) for i, ln in enumerate(lines, 1) if any(ord(c) < 32 and c != "\t" for c in ln)]
print("  控制字符:", len(ctrl))
for x in ctrl[:5]:
    print("    ", x)

# ---------- 7. 空标题 / 断链 ----------
print("\n=== 空标题检查 ===")
n_empty = 0
for idx, (no, lv, t) in enumerate(heads):
    nxt = heads[idx + 1][0] if idx + 1 < len(heads) else len(lines) + 1
    nxt_lv = heads[idx + 1][1] if idx + 1 < len(heads) else 0
    body = [x for x in lines[no:nxt - 1] if x.strip()]
    if not body:
        if nxt_lv > lv:
            continue   # 下一行就是子标题：内容由小节承载，不算空
        n_empty += 1
        print("  !! 空标题 %5d: %s" % (no, t))
if not n_empty:
    print("  OK 无空标题")

# ---------- 8. 待定项计数 ----------
print("\n=== 待定项编号 ===")
ids = re.findall(r"^\|\s*([A-Z]\d+)\s*\|", raw, re.M)
groups = collections.Counter(x[0] for x in ids)
print("  总数:", len(ids), " 按组:", dict(sorted(groups.items())))
dup = [k for k, v in collections.Counter(ids).items() if v > 1]
print("  重复:", dup if dup else "无")
