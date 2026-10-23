# -*- coding: utf-8 -*-
"""check_docs.py -- structural checks for the three-volume docs.

Checks
  1. every volume carries exactly the expected H2 sections
  2. every section reference (\u00a7N or \u00a7N.M) resolves to an existing heading,
     and a prefixed reference (xxx.md \u00a7N) points into that volume
  3. key blocks that must survive the split are present verbatim
  4. no mojibake / replacement characters

Usage: python scripts/check_docs.py [--verbose]
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, 'docs')
ARCHIVED = os.path.join(DOCS, 'archived',
                        'design-memory-learning-2026-10-22.md')

EXPECT = {
    'design.md': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14],
    'experiments.md': [10, 11, 13, 16],
    'reference.md': [12, 15],
}

verbose = '--verbose' in sys.argv
fails = []

# --- load ----------------------------------------------------------------
texts = {}
for name in EXPECT:
    p = os.path.join(DOCS, name)
    if not os.path.exists(p):
        fails.append('MISSING FILE: %s' % name)
        continue
    texts[name] = io.open(p, 'r', encoding='utf-8', newline='').read()

# --- 1. sections ---------------------------------------------------------
sec_owner = {}
for name, txt in texts.items():
    got = [int(x) for x in re.findall(r'^## (\d+)\.', txt, re.M)]
    if got != EXPECT[name]:
        fails.append('%s: sections %s != expected %s'
                     % (name, got, EXPECT[name]))
    for n in got:
        if n in sec_owner:
            fails.append('section %d appears in both %s and %s'
                         % (n, sec_owner[n], name))
        sec_owner[n] = name

# --- 2. references -------------------------------------------------------
subs = {}          # '11.4' -> file
for name, txt in texts.items():
    for m in re.finditer(r'^(#{3,4}) (\d+(?:\.\d+)+)', txt, re.M):
        subs.setdefault(m.group(2), name)

ref_re = re.compile(r'(?:(`?(?:design|experiments|reference)\.md`?) )?\u00a7(\d+(?:\.\d+)*)')
bad_ref = 0
checked = 0
for name, txt in texts.items():
    for m in ref_re.finditer(txt):
        prefix, num = m.group(1), m.group(2)
        top = int(num.split('.')[0])
        checked += 1
        owner = sec_owner.get(top)
        if owner is None:
            fails.append('%s: \u00a7%s -> no such section' % (name, num))
            bad_ref += 1
            continue
        if prefix:
            got = prefix.strip('`')
            if got != owner:
                fails.append('%s: "%s \u00a7%s" but section %d is in %s'
                             % (name, got, num, top, owner))
                bad_ref += 1
        elif owner != name:
            fails.append('%s: bare \u00a7%s but section %d is in %s'
                         % (name, num, top, owner))
            bad_ref += 1
        if '.' in num and num not in subs:
            fails.append('%s: \u00a7%s -> subsection heading not found'
                         % (name, num))
            bad_ref += 1

# --- 3. verbatim blocks that must survive --------------------------------
MUST = [
    # life-criteria table (stage A)
    '| **\u4e0d\u70b8** | \u6d3b\u52a8\u5e45\u5ea6\u65f6\u95f4\u5e8f\u5217\u7684 max / mean | < 20 |',
    '| **\u6709\u8bb0\u5fc6** | \u6ce8\u5165\u552f\u4e00\u6807\u8bb0\uff0c100 \u6b65\u540e\u80fd\u4e0d\u80fd\u7ebf\u6027\u8bfb\u51fa | \u9ad8\u4e8e\u968f\u673a |',
    # stage B vacuum table
    '| \u4e00\u6761\u6c38\u4e0d\u91cd\u7f6e\u7684\u65f6\u95f4\u8f74 | \u6bcf\u6837\u672c / \u6bcf\u5e27\u91cd\u7f6e |',
    # the three original rewrite records
    '> - **v1**\uff1a\u4e24\u4e2a\u5b50\u4efb\u52a1\u4ea4\u66ff + \u6d88\u878d\u53cc\u5206\u79bb',
    '> - **v3**\uff1a\u52a0\u4e00\u4e2a\u524d\u9762\u7684\u9636\u6bb5',
    # soul-level conclusions
    '\u957f\u671f\u8bb0\u5fc6\u5c31\u662f\u8fde\u63a5\u672c\u8eab',
    '\u03c4 \u4e0d\u80fd\u9010\u5355\u5143\u8fde\u7eed\u8bbe\u8ba1',
    # judgement / criterion
    '\u5224\u636e\u662f\u8fc7\u7a0b\u7684\u5f62\u72b6',
    # snn positioning
    '\u8fd9\u4e0d\u662f\u4e00\u4e2a\u65b0\u67b6\u6784\uff0c\u662f\u4e00\u4e2a',
]
all_txt = '\n'.join(texts.values())
for needle in MUST:
    if needle not in all_txt:
        fails.append('verbatim block missing: %s' % needle[:50])

# --- 4. mojibake ---------------------------------------------------------
BAD = ['\ufffd', '\u951f', '\u9239', '\u6ebe', '\u6522', '\u65ba']
for name, txt in texts.items():
    for ch in BAD:
        if ch in txt:
            ln = txt[:txt.index(ch)].count('\n') + 1
            fails.append('%s:%d: suspicious char U+%04X' % (name, ln, ord(ch)))

# --- report --------------------------------------------------------------
print('volumes      : %s' % ', '.join('%s(%d lines)' % (n, t.count('\n'))
                                       for n, t in texts.items()))
print('sections     : %s' % ', '.join('%d->%s' % (k, v)
                                      for k, v in sorted(sec_owner.items())))
print('subsections  : %d' % len(subs))
print('references   : %d checked, %d bad' % (checked, bad_ref))
print('verbatim     : %d needles' % len(MUST))
print('')
if fails:
    print('FAIL (%d)' % len(fails))
    for f in fails:
        print('  - ' + f)
    raise SystemExit(1)
print('PASS -- structure, references, verbatim blocks and encoding all clean')
