#!/usr/bin/env python3
"""Lightweight repo check (stdlib only).

1. Every *.ipynb and *.json parses as JSON (notebooks also need a "cells" list).
2. Every relative link in *.md files points to an existing file/dir.
Skips .git, data/, ref/ (large binaries). Exit 0 = pass, 1 = fail.
Usage: python3 scripts/verify.py
"""
import json, re, sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
SKIP = {".git", "data", "ref", "node_modules", ".venv"}
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
errors = []

def files(pattern):
    for p in ROOT.rglob(pattern):
        if not SKIP.intersection(p.relative_to(ROOT).parts):
            yield p

n_json = 0
for p in list(files("*.json")) + list(files("*.ipynb")):
    n_json += 1
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if p.suffix == ".ipynb" and not isinstance(data.get("cells"), list):
            errors.append(f"{p.relative_to(ROOT)}: notebook has no cells list")
    except Exception as e:
        errors.append(f"{p.relative_to(ROOT)}: invalid JSON ({e})")

n_links = 0
for p in files("*.md"):
    text = re.sub(r"```.*?```", "", p.read_text(encoding="utf-8"), flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    for target in LINK.findall(text):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
            continue
        path = unquote(target.split("#", 1)[0])
        if not path:
            continue
        n_links += 1
        dest = (ROOT / path.lstrip("/")) if path.startswith("/") else (p.parent / path)
        if not dest.exists():
            errors.append(f"{p.relative_to(ROOT)}: broken link -> {target}")

for e in errors:
    print("FAIL", e)
print(f"verify: {n_json} json/ipynb files, {n_links} relative md links, {len(errors)} errors")
sys.exit(1 if errors else 0)
