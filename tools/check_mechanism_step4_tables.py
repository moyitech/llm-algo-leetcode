#!/usr/bin/env python3
"""Check the Step 4 TODO table of Part 02 mechanism notebooks.

The table wording is free; what must hold is that every TODO in the question
cell has exactly one row in the Step 4 table, and vice versa.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PART02_ROOT = REPO_ROOT / "02_PyTorch_Algorithms"
PROJECT_REGISTRY = REPO_ROOT / "benchmarks/project_registry.json"
TODO_RE = re.compile(r"\bTODO\s*(\d+)\b")
TODO_ROW_RE = re.compile(r"^(?:TODO\s*)?(\d+)\b")


def registered_project_paths(registry_path: Path = PROJECT_REGISTRY) -> set[Path]:
    """Return project notebook paths declared by the project registry."""
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return {
        Path(item["notebook"]).resolve()
        for item in payload.get("projects", [])
        if item.get("notebook")
    }


def discover_part02_mechanisms() -> list[Path]:
    """Discover mechanism notebooks while excluding projects and reserved slots."""
    projects = registered_project_paths()
    return sorted(
        path
        for path in PART02_ROOT.glob("*.ipynb")
        if path.resolve() not in projects and "Reserved" not in path.name
    )


def expand_paths(paths: list[Path]) -> list[Path]:
    """Expand explicit notebook and directory arguments in stable order."""
    notebooks: set[Path] = set()
    for path in paths:
        if path.is_dir():
            notebooks.update(path.glob("*.ipynb"))
        elif path.suffix == ".ipynb":
            notebooks.add(path)
    return sorted(notebooks)


def step4_markdown(notebook: dict) -> str | None:
    """Return markdown starting at the first mechanism Step 4 heading."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", []))
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("### Step 4"):
                return "\n".join(lines[index:])
    return None


def split_row(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def iter_tables(markdown: str) -> list[list[str]]:
    """Return every contiguous Markdown table in the text, in order."""
    tables: list[list[str]] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("|"):
            current.append(line)
            continue
        if current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def todo_table_lines(markdown: str) -> list[str]:
    """Return the first table whose body rows all start with a TODO number."""
    for table in iter_tables(markdown):
        body = table[2:]
        if body and all(TODO_ROW_RE.match(split_row(row)[0]) for row in body):
            return table
    return []


def audit_with_warnings(path: Path) -> tuple[list[str], list[str]]:
    """Return (blocking issues, soft warnings) for one notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    markdown = step4_markdown(notebook)
    if markdown is None:
        return ["缺少以 `### Step 4` 开头的 Markdown 单元"], []

    table = todo_table_lines(markdown)
    if not table:
        return ["Step 4 缺少 TODO 任务表（每行第一列以 TODO n 开头）"], []

    issues: list[str] = []
    warnings: list[str] = []
    width = len(split_row(table[0]))
    table_todos: list[str] = []

    for row_number, row in enumerate(table[2:], start=1):
        cells = split_row(row)
        if len(cells) != width:
            issues.append(f"第 {row_number} 行列数与表头不一致：{row!r}")
            continue
        if any(not cell for cell in cells):
            issues.append(f"第 {row_number} 行存在空字段：{row!r}")
        first_cell = cells[0]
        match = TODO_ROW_RE.match(first_cell)
        if match:
            table_todos.append(match.group(1))
        if "`" not in first_cell:
            warnings.append(f"第 {row_number} 行第一列建议用反引号标出函数或方法名：{first_cell!r}")

    question_cells = [
        cell for cell in notebook.get("cells", [])
        if "tutorial-question" in cell.get("metadata", {}).get("tags", [])
    ]
    if len(question_cells) != 1 or question_cells[0].get("cell_type") != "code":
        issues.append("必须恰有一个带 tutorial-question 标签的代码单元；旧 Notebook 请先补齐标签")
    question_source = "\n".join("".join(cell.get("source", [])) for cell in question_cells)
    question_todos = set(TODO_RE.findall(question_source))
    if not question_todos:
        issues.append("题目区未找到 TODO 编号，无法核对任务表")
    table_set = set(table_todos)
    missing = sorted(question_todos - table_set, key=int)
    extra = sorted(table_set - question_todos, key=int)
    if missing:
        issues.append(f"题目区有 TODO {', '.join(missing)}，但 Step 4 表格缺少对应行")
    if extra:
        issues.append(f"Step 4 表格有 TODO {', '.join(extra)}，但题目区没有对应 TODO")
    if len(table_todos) != len(table_set):
        issues.append("Step 4 表格存在重复的 TODO 编号")
    return issues, warnings


def audit(path: Path) -> list[str]:
    """Return blocking teaching-template violations for one notebook."""
    return audit_with_warnings(path)[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Step 4 TODO tables in Part 02 mechanism notebooks."
    )
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument(
        "--all-part02",
        action="store_true",
        help="自动扫描 Part 02 中未登记为项目且不是保留位的机制 Notebook。",
    )
    args = parser.parse_args()

    notebooks = discover_part02_mechanisms() if args.all_part02 else expand_paths(args.notebooks)
    if not notebooks:
        parser.error("请传入机制 Notebook，或使用 --all-part02")

    project_paths = registered_project_paths()
    wrong_type = [path for path in notebooks if path.resolve() in project_paths]
    if wrong_type:
        names = ", ".join(path.as_posix() for path in wrong_type)
        parser.error(f"以下文件已登记为项目页，不能使用机制页 Step 4 检查器：{names}")

    failed = False
    for path in notebooks:
        issues, warnings = audit_with_warnings(path)
        if not issues:
            print(f"PASS {path}")
        else:
            failed = True
            print(f"FAIL {path}")
            for issue in issues:
                print(f"  - {issue}")
        for warning in warnings:
            print(f"  ~ {warning}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
