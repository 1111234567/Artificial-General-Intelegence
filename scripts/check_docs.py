"""检查现行文档的 UTF-8 编码、本地链接和必要入口。

用法：python scripts/check_docs.py [--verbose] [文件 ...]
相对参数以仓库根目录为基准；默认检查分主题报告和现行索引，归档原件可单独指定。
只检查文档结构，不把历史数字、原句或研究假设设为通过条件。
"""

import argparse
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "agi.md", "CHANGELOG.md", "src/README.md", "src/neurons/README.md",
    "docs/README.md", "docs/design.md", "docs/experiments.md", "docs/reference.md",
    "docs/reports/README.md", "docs/evidence/README.md",
    "docs/review-2026-10-26.md", "docs/archived/README.md", "scripts/run.py", "scripts/align_layers.py",
    "scripts/README.md", "scripts/historical/README.md", "test/run.py", "test/README.md",
    "data/README.md", "data/pulse-game/README.md", "experiments/pulse_game/INDEX.md",
    "experiments/README.md", "experiments/legacy/README.md",
    "experiments/prototypes/README.md", "experiments/diagnostics/README.md",
    "experiments/baselines/README.md", "experiments/retired/README.md",
)


def without_code(text, inline=True):
    """保留行号，忽略围栏代码和行内代码中的示例链接。"""
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            lines.append("\n" if line.endswith("\n") else "")
        elif fence:
            lines.append("\n" if line.endswith("\n") else "")
        else:
            lines.append(re.sub(r"(`+).*?\1", "", line) if inline else line)
    return "".join(lines)


def links(text):
    """产生 (行号, 目标)：行内链接、引用定义和 HTML href/src。"""
    body = without_code(text)
    for match in re.finditer(r"!?\[[^\]\n]*\]\(\s*", body):
        start = pos = match.end()
        if pos < len(body) and body[pos] == "<":
            end = body.find(">", pos + 1)
            target = body[pos + 1:end] if end >= 0 else ""
        else:
            depth = 0
            while pos < len(body):
                ch = body[pos]
                if ch == "\\" and pos + 1 < len(body):
                    pos += 2
                    continue
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    if depth == 0:
                        break
                    depth -= 1
                elif ch.isspace() and depth == 0:
                    break
                pos += 1
            target = body[start:pos]
        target = re.sub(r"\\([\\`()\[\] ])", r"\1", target)
        yield body.count("\n", 0, match.start()) + 1, target
    for match in re.finditer(r"^\s{0,3}\[[^\]\n]+\]:\s*(?:<([^>\n]+)>|(\S+))", body, re.M):
        yield body.count("\n", 0, match.start()) + 1, match.group(1) or match.group(2)
    for match in re.finditer(r"<[^>]*?\b(?:href|src)=[\"']([^\"']+)[\"'][^>]*>", body):
        yield body.count("\n", 0, match.start()) + 1, match.group(1)


def anchors(text):
    """常见 Markdown 标题锚点（含重复标题后缀）和显式 HTML id/name。"""
    body = without_code(text, inline=False)
    result = set(re.findall(r"\b(?:id|name)=[\"']([^\"']+)[\"']", body))
    seen = {}
    for match in re.finditer(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", body, re.M):
        title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", match.group(1))
        title = re.sub(r"<[^>]+>", "", title).lower()
        slug = re.sub(r"[^\w\- ]", "", title).replace(" ", "-")
        number = seen.get(slug, 0)
        seen[slug] = number + 1
        result.add(slug if number == 0 else "%s-%d" % (slug, number))
    return result


def default_targets():
    paths = list(ROOT.glob("*.md")) + list((ROOT / "docs").glob("*.md"))
    paths += list((ROOT / "docs/reports").rglob("*.md"))
    paths += [ROOT / "src/README.md", ROOT / "src/neurons/README.md", ROOT / "test/README.md",
              ROOT / "docs/archived/README.md", ROOT / "docs/evidence/README.md",
              ROOT / "data/README.md", ROOT / "data/pulse-game/README.md",
              ROOT / "experiments/pulse_game/INDEX.md"]
    paths += list((ROOT / "experiments").rglob("*README.md"))
    paths += list((ROOT / "experiments").rglob("*-plan.md"))
    paths += list((ROOT / "scripts").rglob("README.md"))
    return sorted(set(paths))


def display(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    targets = [(ROOT / p).resolve() for p in args.paths] if args.paths else default_targets()
    failures = []
    texts = {}
    checked_links = 0

    if not args.paths:
        failures.extend("必要路径不存在：%s" % rel for rel in REQUIRED if not (ROOT / rel).exists())

    def read(path):
        if path not in texts:
            try:
                texts[path] = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError) as error:
                failures.append("%s：不能作为 UTF-8 读取（%s）" % (display(path), error))
                texts[path] = None
        return texts[path]

    for path in targets:
        text = read(path)
        if text is None:
            continue
        for lineno, line in enumerate(text.split("\n"), 1):
            if "\ufffd" in line or any(ord(ch) < 32 and ch not in "\t\r" for ch in line):
                failures.append("%s:%d：替代字符或控制字符" % (display(path), lineno))
        for lineno, target in links(text):
            parsed = urlsplit(target)
            if (parsed.scheme and not (len(parsed.scheme) == 1 and target[1:2] == ":")) or parsed.netloc:
                continue
            local = unquote(parsed.path)
            if local.startswith("/"):
                dest = (ROOT / local.lstrip("/")).resolve()
            else:
                dest = (path.parent / local).resolve() if local else path
            checked_links += 1
            if not dest.exists():
                failures.append("%s:%d：本地目标不存在：%s" % (display(path), lineno, target))
                continue
            if parsed.fragment and dest.is_file() and dest.suffix.lower() == ".md":
                other = read(dest)
                if other is not None and unquote(parsed.fragment) not in anchors(other):
                    failures.append("%s:%d：标题锚点不存在：%s" % (display(path), lineno, target))
        if args.verbose:
            print("checked", display(path))

    for failure in failures:
        print("FAIL", failure)
    print("%s：%d 份现行文档，%d 个本地链接，%d 个问题" % (
        "FAIL" if failures else "PASS", len(targets), checked_links, len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
