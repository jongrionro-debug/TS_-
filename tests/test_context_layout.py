"""Protect the shared layout against duplicate contexts and ambiguous ADR names."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('context_layout', ROOT / 'scripts/check_context.py')
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class ContextLayoutTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name in layout.REQUIRED:
            self.write(name, '# Context\n')
        self.write('docs/README.md', '[project](project.md)\n')

    def write(self, name, body=''):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding='utf-8')

    def errors(self):
        return '\n'.join(layout.check_layout(self.root))

    def test_minimal_layout_and_module_reports_are_allowed(self):
        self.write('experiments/run/results.md')
        self.write('src/module/README.md')
        self.assertEqual(layout.check_layout(self.root), [])

    def test_missing_canonical_document_fails(self):
        (self.root / 'docs/project.md').unlink()
        self.assertIn('docs/project.md: required file is missing', self.errors())

    def test_old_context_names_and_new_plans_fail(self):
        for name in ('agent.md', 'WORKFLOW.md', 'docs/master.md', 'docs/plans/today.md'):
            self.write(name)
        failures = self.errors()
        for name in ('agent.md', 'WORKFLOW.md', 'docs/master.md', 'docs/plans/'):
            self.assertIn(name, failures)

    def test_adr_names_duplicates_and_index_are_checked(self):
        self.write('docs/adr/0001-first.md')
        self.write('docs/adr/0001-second.md')
        self.write('docs/adr/ADR-002-old.md')
        self.assertIn('duplicate ADR number 0001', self.errors())
        self.assertIn('expected NNNN-short-title.md', self.errors())
        self.assertIn('missing from docs/adr/README.md', self.errors())

    def test_indexed_adr_and_optional_document_pass(self):
        self.write('docs/adr/0001-first.md')
        self.write('docs/adr/README.md', '[first](0001-first.md)\n')
        self.write('docs/architecture.md')
        self.assertIn('missing from docs/README.md', self.errors())
        self.write('docs/README.md', '[project](project.md)\n[architecture](architecture.md)\n')
        self.assertEqual(layout.check_layout(self.root), [])

    def test_new_archive_and_extra_issue_template_fail(self):
        self.write('docs/archive/summary.md')
        self.write('.github/ISSUE_TEMPLATE/another-task.yml')
        self.assertIn('new archive copies are not allowed', self.errors())
        self.assertIn('another-task.yml', self.errors())

    def test_numbered_template_and_nonstandard_spec_fail(self):
        self.write('docs/adr/0000-template.md')
        self.write('docs/specs/New_Plan.md')
        self.assertIn('starting at 0001', self.errors())
        self.assertIn('expected short-name.md', self.errors())


if __name__ == '__main__':
    unittest.main()
