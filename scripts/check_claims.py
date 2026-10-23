# -*- coding: utf-8 -*-
"""草案内容层核验：同源残留 / 断言一致性"""
import io, re

P = r"E:\pando\workspace\design-memory-learning-2026-10-22.md"
s = io.open(P, "r", encoding="utf-8", newline="").read()
lines = s.split("\r\n")

# 1. 每个旧断言必须出现在「降级/推翻」标注之后，或本身带修正注
OLD = {
    "两区的划分有实测依据": None,
    "时间常数不会自发分化": None,
    "时间常数必须显式给定": None,
    "短期区写坏": "降级区",
    "长期区写坏": "降级区",
    "三时间尺度三载体": None,
    "两层：预测模型": None,
}
print("=== 旧断言定位 ===")
for kw in OLD:
    hits = [i + 1 for i, ln in enumerate(lines) if kw in ln]
    print("  %-24s %s" % (kw, hits if hits else "（已清）"))

# 2. 关键新断言必须存在
NEW = [
    "长期记忆不是「存放在连接里的内容」，它就是连接的形状",
    "双分离",
    "未出现双峰",
    "接口",
    "可塑性",
    "稳态",
    "巩固 = 强化",
    "次数不是时间",
]
print("\n=== 新断言是否在位 ===")
for kw in NEW:
    print("  %-30s %s" % (kw, "OK" if kw in s else "!! 缺失"))

# 3. 章节引用完整回查（包括 §10.10 / §11.4 / §12.9 / §2.10）
print("\n=== 新增节被引用次数 ===")
for tag in ["§2.10", "§2.2", "§10.10", "§11.4", "§12.9", "§2.9", "S1", "S5"]:
    print("  %-8s x%d" % (tag, s.count(tag)))

# 4. 未定项引用：正文里提到的组号是否都在表里
ids_in_table = set(re.findall(r"^\|\s*([A-Z]\d+)\s*\|", s, re.M))
refs = set(re.findall(r"\b([A-Z]\d{1,2})\b", s))
known_prefix = set(x[0] for x in ids_in_table)
candidates = {r for r in refs if r[0] in known_prefix and r not in ids_in_table}
print("\n=== 正文提到但表中不存在的编号 ===")
print(" ", sorted(candidates) if candidates else "无")

# 5. 段落长度异常（可能截断）
print("\n=== 超长行（>1500 字符，可能一段未换行）===")
longs = [(i + 1, len(ln)) for i, ln in enumerate(lines) if len(ln) > 1500]
print(" ", longs if longs else "无")


