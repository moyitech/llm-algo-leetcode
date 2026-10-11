"""Regression tests for incomplete or misleading Step 4 task tables."""
import json
import tempfile
import unittest
from pathlib import Path

from check_mechanism_step4_tables import audit_with_warnings


class Step4TableTests(unittest.TestCase):
    def audit(self, rows, question="# TODO 1\n# TODO 2", tags=None, copies=1, cell_type="code"):
        table = "### Step 4：任务\n\n| 阶段 | 说明 |\n|---|---|\n| 准备 | 输入 |\n\n"
        table += "| 编号与函数 | 要完成什么 |\n|---|---|\n" + "\n".join(rows)
        cells = [{"cell_type": "markdown", "source": [table]}]
        cells += [{"cell_type": cell_type, "metadata": {"tags": ["tutorial-question"] if tags is None else tags},
                   "source": [question]} for _ in range(copies)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.ipynb"
            path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
            return audit_with_warnings(path)

    def test_flexible_header_and_numbered_rows_after_other_table(self):
        issues, warnings = self.audit(["| 1 · `a` | 计算 |", "| TODO 2 · `b` | 合并 |"])
        self.assertEqual((issues, warnings), ([], []))

    def test_equal_row_count_does_not_hide_wrong_numbers(self):
        issues, _ = self.audit(["| TODO 1 · `a` | 计算 |", "| TODO 3 · `c` | 合并 |"])
        self.assertTrue(any("TODO 2" in issue and "缺少" in issue for issue in issues))
        self.assertTrue(any("TODO 3" in issue and "没有" in issue for issue in issues))

    def test_duplicate_rows_fail_even_without_question_todos(self):
        for question in ("# TODO 1", "pass"):
            with self.subTest(question=question):
                issues, _ = self.audit(["| TODO 1 · `a` | 计算 |"] * 2, question=question)
                self.assertTrue(any("重复" in issue for issue in issues))

    def test_empty_question_is_not_silently_accepted(self):
        issues, _ = self.audit(["| TODO 1 · `a` | 计算 |"], question="pass")
        self.assertTrue(any("未找到 TODO" in issue for issue in issues))
        self.assertTrue(any("题目区没有" in issue for issue in issues))

    def test_question_tag_must_identify_one_code_cell(self):
        for kwargs in ({"tags": []}, {"copies": 2}, {"cell_type": "markdown"}):
            with self.subTest(kwargs=kwargs):
                issues, _ = self.audit(["| TODO 1 · `a` | 计算 |"], question="# TODO 1", **kwargs)
                self.assertTrue(any("tutorial-question" in issue for issue in issues))

    def test_malformed_rows_fail(self):
        for row in ("| TODO 1 · `a` | |", "| TODO 1 · `a` | 计算 | 多余 |"):
            with self.subTest(row=row):
                issues, _ = self.audit([row], question="# TODO 1")
                self.assertTrue(issues)

    def test_missing_function_markup_is_only_a_warning(self):
        issues, warnings = self.audit(["| TODO 1 初始化 | 建立状态 |"], question="# TODO 1")
        self.assertFalse(issues)
        self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    unittest.main()
