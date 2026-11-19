"""归档文档格式核验：重复标题与同一章节内的子节编号。

用法：python scripts/historical/check_claims.py [文件 ...]
默认读取 docs/archived/*-2026-10-25.md 三卷。
旧工具曾拿当前代码清点值约束文档数字；迁移后取消这一规则。
本工具不检验实验结论、参数依据或设计假设，不能替代证据审查。
"""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = [ROOT / "docs" / "archived" / (name + "-2026-10-25.md")
            for name in ("design", "experiments", "reference")]


def main():
    targets = [(ROOT / a).resolve() for a in sys.argv[1:]] or DEFAULTS
    failures = []
    headings = 0
    for path in targets:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            failures.append("%s：%s" % (path, error))
            continue
        seen = {}
        subsections = {}
        fence = None
        for lineno, line in enumerate(text.splitlines(), 1):
            marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
            if marker:
                token = marker.group(1)
                if fence is None:
                    fence = token
                elif token[0] == fence[0] and len(token) >= len(fence):
                    fence = None
                continue
            if fence:
                continue
            match = re.match(r"^(#{2,3})\s+(\S.*)$", line)
            if not match:
                continue
            headings += 1
            key = match.groups()
            if key in seen:
                failures.append("%s:%d：重复标题，与第 %d 行相同" % (path.name, lineno, seen[key]))
            seen[key] = lineno
            if len(match.group(1)) == 3:
                number = re.match(r"(\d+)\.(\d+)(?![\d.])", match.group(2))
                if number:
                    parent, child = map(int, number.groups())
                    subsections.setdefault(parent, []).append(child)
        for parent, numbers in subsections.items():
            if numbers != list(range(numbers[0], numbers[-1] + 1)):
                failures.append("%s：历史 §%d 的子节编号不连续：%s" % (path.name, parent, numbers))
        if "\ufffd" in text:
            failures.append("%s：含 U+FFFD 替代字符" % path.name)

    print("历史格式核验：%d 份归档，%d 个标题；不审查科学结论或当前目录数字。" % (len(targets), headings))
    for failure in failures:
        print("FAIL", failure)
    print("FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
