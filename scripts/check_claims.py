# -*- coding: utf-8 -*-
"""文档内容一致性核验：实算清单 / 数字断言 / 重复标题。

用法：python scripts/check_claims.py [文件 ...]
      缺省扫 agi.md / README.md / src/README.md / test/CONSTANTS.md /
      test/archived/README.md / docs/{design,experiments,reference}.md

设计原则：先从代码与目录**实算**，再拿实算值去核文档里的断言。
历史记录（含「当时 / 原记 / 复核 / 旧 / 已过时」的行）跳过，不误报。
"""
import io, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = ("当时", "原记", "复核", "已过时", "历史", "原来", "撤回", "旧版", "旧读数")


def read(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def count_in(path, pat):
    return len(re.findall(pat, read(os.path.join(ROOT, path))))


# ---------- 实算 ----------
CONFIGS = count_in("src/experiments.py", r'\bname="')
SRC_MODULES = len([f for f in os.listdir(os.path.join(ROOT, "src"))
                   if f.endswith(".py") and f != "__init__.py"])
TEST_PY = len([f for f in os.listdir(os.path.join(ROOT, "test")) if f.endswith(".py")])
ARCH_PY = len([f for f in os.listdir(os.path.join(ROOT, "test", "archived")) if f.endswith(".py")])
RUN_CMDS = count_in("test/run.py", r'cmd == "')
ALIGN_LAYERS = count_in("scripts/align_layers.py", r'cmd == "')
flows = set(re.findall(r'''flow == ["'](\w+)["']''', read(os.path.join(ROOT, "src", "runner.py"))))
FLOWS = len(flows) + ("agent" not in flows)

TARGETS = sys.argv[1:] or [
    "agi.md", "README.md", "src/README.md", "test/CONSTANTS.md",
    "test/archived/README.md", "docs/design.md", "docs/experiments.md", "docs/reference.md",
]

# ---------- 数字断言：正则 → (期望值, 说明) ----------
CHECKS = [
    (r"(\d+)\s*个实验声明", CONFIGS, "实验声明数"),
    (r"`src/`\s*(\d+)\s*个模块", SRC_MODULES, "src 模块数"),
    (r"下\s*\*\*(\d+)\*\*\s*个\s*`?\.py`?", TEST_PY, "test 顶层 .py 数"),
    (r"\*\*(\d+)\*\*\s*个退役版本", ARCH_PY, "archived 脚本数"),
    (r"共\s*(\d+)\s*个）已移至", ARCH_PY, "archived 脚本数（引言）"),
    (r"（(\d+)\s*层）?", ALIGN_LAYERS, "align_layers 层数"),
    (r"移植对齐工具（(\d+)\s*层）", ALIGN_LAYERS, "align_layers 层数"),
    (r"（(\d+)\s*个配置）", CONFIGS, "配置数"),
    (r"唯一声明实验的地方\*\*（(\d+)\s*个）", CONFIGS, "配置数（src/README）"),
    (r"（(\d+)\s*个，全部带来源）", 17, "CONSTANTS 第 4 节常数条数"),
    (r"框架修正记录（(\d+)\s*行", 33, "§13.2 表行数"),
]

fails = []
warns = []
used = 0

for rel in TARGETS:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        warns.append("缺文件：%s" % rel)
        continue
    for lineno, line in enumerate(read(p).split("\n"), 1):
        if any(k in line for k in SKIP):
            continue
        for pat, exp, desc in CHECKS:
            for m in re.finditer(pat, line):
                got = int(m.group(1))
                used += 1
                if got != exp:
                    fails.append("%s:%d  %s：文档写 %d，实算 %d\n    %s"
                                 % (rel, lineno, desc, got, exp, line.strip()[:110]))

# ---------- 重复标题 ----------
for rel in TARGETS:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        continue
    seen = {}
    for lineno, line in enumerate(read(p).split("\n"), 1):
        m = re.match(r"^(#{2,3})\s+(\S.*)$", line)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        if key in seen:
            fails.append("%s:%d  重复标题「%s」与第 %d 行" % (rel, lineno, line.strip(), seen[key]))
        else:
            seen[key] = lineno

# ---------- §16 子节编号连续性 ----------
exp = os.path.join(ROOT, "docs", "experiments.md")
if os.path.exists(exp):
    nums = [int(x.split(".")[1]) for x in re.findall(r"^### (16\.\d+)", read(exp), re.M)]
    if nums != list(range(1, len(nums) + 1)):
        fails.append("docs/experiments.md  §16 子节编号不连续：%s" % nums)

# ---------- 输出 ----------
print("实算：配置 %d / src 模块 %d / test .py %d / archived %d / run 子命令 %d / align 层 %d / flow %d"
      % (CONFIGS, SRC_MODULES, TEST_PY, ARCH_PY, RUN_CMDS, ALIGN_LAYERS, FLOWS))
print("断言命中 %d 处，扫描 %d 个文件" % (used, len(TARGETS)))
if warns:
    print("\n提示：")
    for w in warns:
        print("  " + w)
if fails:
    print("\nFAIL %d 处：" % len(fails))
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("\nPASS -- 数字断言、重复标题、§16 编号全部一致")
