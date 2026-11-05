#!/usr/bin/env python3
"""Validate the shared context layout without dependencies or file writes."""
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    'README.md', 'AGENTS.md', 'docs/README.md', 'docs/project.md',
    'docs/adr/README.md', 'scripts/check_context.py',
    '.github/ISSUE_TEMPLATE/task.yml', '.github/ISSUE_TEMPLATE/bug_report.yml',
    '.github/pull_request_template.md',
)
DOC_FILES = {'README.md', 'project.md', 'architecture.md', 'evaluation.md', 'workflow.md'}
DOC_DIRS = {'adr', 'specs', 'sources', 'archive'}
# Existing unique originals and a frozen experiment's evidence; no new archive entries.
LEGACY_FILES = {
    'docs/archive/README.md',
    'docs/archive/original-documentation-2026-09-27.tar.gz',
    'docs/archive/2026-09-23-pilot/QA_LOG_KO.md',
}
SHORT_NAME = re.compile(r'[a-z0-9]+(?:-[a-z0-9]+)*\.md')
ADR_NAME = re.compile(r'([0-9]{4})-[a-z0-9]+(?:-[a-z0-9]+)*\.md')


def check_layout(root=ROOT):
    root = Path(root)
    if not root.is_dir():
        return [f"{root}: repository directory is missing"]
    failures = []
    for name in REQUIRED:
        if not (root / name).is_file():
            failures.append(f'{name}: required file is missing')
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() == '.md' and path.name not in {'README.md', 'AGENTS.md', 'LICENSE.md'}:
            failures.append(f'{path.name}: root Markdown must use a fixed document role')
    docs = root / 'docs'
    index = (docs / 'README.md').read_text(encoding='utf-8') if (docs / 'README.md').is_file() else ''
    if docs.is_dir():
        for path in docs.iterdir():
            if path.name == '.DS_Store':
                continue
            if path.is_dir():
                if path.name not in DOC_DIRS:
                    failures.append(f'docs/{path.name}/: unknown document directory')
            elif path.name not in DOC_FILES:
                failures.append(f'docs/{path.name}: unknown document name')
            elif path.name != 'README.md' and path.name not in index:
                failures.append(f'docs/{path.name}: missing from docs/README.md')
    adr = docs / 'adr'
    guide = (adr / 'README.md').read_text(encoding='utf-8') if (adr / 'README.md').is_file() else ''
    numbers = set()
    if adr.is_dir():
        for path in sorted(adr.iterdir()):
            if path.name in {'README.md', '.DS_Store'}:
                continue
            match = ADR_NAME.fullmatch(path.name)
            if not path.is_file() or not match or match[1] == '0000':
                failures.append(f'docs/adr/{path.name}: expected NNNN-short-title.md, starting at 0001')
                continue
            if match[1] in numbers:
                failures.append(f'docs/adr/{path.name}: duplicate ADR number {match[1]}')
            numbers.add(match[1])
            if path.name not in guide:
                failures.append(f'docs/adr/{path.name}: missing from docs/adr/README.md')
    specs = docs / 'specs'
    if specs.is_dir():
        for path in specs.iterdir():
            if path.name == '.DS_Store':
                continue
            if not path.is_file() or (path.name != 'README.md' and not SHORT_NAME.fullmatch(path.name)):
                failures.append(f'docs/specs/{path.name}: expected short-name.md')
    archive = docs / 'archive'
    if archive.is_dir():
        for path in archive.rglob('*'):
            if path.is_file() and path.name != '.DS_Store' and path.relative_to(root).as_posix() not in LEGACY_FILES:
                failures.append(f'{path.relative_to(root)}: new archive copies are not allowed')
        if 'archive/README.md' not in index:
            failures.append('docs/archive/: preservation reason must be linked from docs/README.md')
    templates = root / '.github/ISSUE_TEMPLATE'
    if templates.is_dir():
        for path in templates.iterdir():
            if path.is_file() and path.suffix.lower() in {'.md', '.yml', '.yaml'} and path.name not in {'task.yml', 'bug_report.yml', 'config.yml'}:
                failures.append(f'.github/ISSUE_TEMPLATE/{path.name}: use task.yml or bug_report.yml')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    failures = check_layout(parser.parse_args().root)
    for failure in failures:
        print(f'FAIL {failure}')
    print(f'context-layout: {"FAIL" if failures else "PASS"} ({len(failures)} errors)')
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
