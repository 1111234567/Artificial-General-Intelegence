# -*- coding: utf-8 -*-
"""verify_docs.py -- quantify the diff between the archived draft and the
three volumes.

Rebuilds the draft from the volumes (references de-normalised, sections
sorted by number), then diffs line by line. Every surviving difference is
printed as a hunk so it can be judged as an intentional edit.
"""
import io
import re
import difflib

ARCH = 'F:/agi/docs/archived/design-memory-learning-2026-10-22.md'
VOLS = ['F:/agi/docs/design.md', 'F:/agi/docs/experiments.md',
        'F:/agi/docs/reference.md']

# --- archived --------------------------------------------------------------
arch = io.open(ARCH, 'r', encoding='utf-8', newline='').read()
A = arch.split('\r\n')
start = next(i for i, L in enumerate(A) if L.startswith('## 0. '))
A = A[start:]
while A and A[-1].strip() == '':
    A.pop()

# --- rebuild from volumes --------------------------------------------------
blocks = {}
for p in VOLS:
    txt = io.open(p, 'r', encoding='utf-8', newline='').read()
    ls = txt.split('\r\n')
    cur = None
    for L in ls:
        m = re.match(r'^## (\d+)\.', L)
        if m:
            cur = int(m.group(1))
            blocks[cur] = []
        if cur is not None:
            blocks[cur].append(L)

B = []
for n in sorted(blocks):
    B.extend(blocks[n])
while B and B[-1].strip() == '':
    B.pop()

# de-normalise the reference prefix
PRE = re.compile(r'`(?:design|experiments|reference)\.md` \u00a7')
B = [PRE.sub('\u00a7', L) for L in B]

print('archived  : %d lines' % len(A))
print('rebuilt   : %d lines' % len(B))

sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
ops = [op for op in sm.get_opcodes() if op[0] != 'equal']
removed = sum(i2 - i1 for _, i1, i2, _, _ in ops)
added = sum(j2 - j1 for _, _, _, j1, j2 in ops)
print('diff hunks: %d  (-%d / +%d lines)' % (len(ops), removed, added))
print('')

for tag, i1, i2, j1, j2 in ops:
    print('=' * 72)
    print('%s  archived[%d:%d] -> rebuilt[%d:%d]' % (tag, i1, i2, j1, j2))
    for L in A[i1:i2][:14]:
        print('  - ' + L[:150])
    if i2 - i1 > 14:
        print('  - ... (%d more removed)' % (i2 - i1 - 14))
    for L in B[j1:j2][:14]:
        print('  + ' + L[:150])
    if j2 - j1 > 14:
        print('  + ... (%d more added)' % (j2 - j1 - 14))
