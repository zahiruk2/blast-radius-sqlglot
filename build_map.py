#!/usr/bin/env python3
"""Fast 4-worker sys.monitoring coverage + fixture + AST scope mapper for sqlglot."""

from __future__ import annotations

import ast
import builtins
import importlib
import io
import json
import os
import sys
import time
import unittest
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SKIP_SLOW_METHODS: set[str] = set()


def collect_unittest_cases(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    cases: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            cases.extend(collect_unittest_cases(item))
        else:
            cases.append(item)
    return cases


def extract_ast_scopes(repo_root: Path) -> dict[str, list[dict]]:
    scopes_by_file: dict[str, list[dict]] = {}
    for py_path in sorted(repo_root.rglob("*.py")):
        rel = py_path.relative_to(repo_root).as_posix()
        if not (rel.startswith("sqlglot/") or rel.startswith("tests/")):
            continue
        try:
            source = py_path.read_text("utf-8")
            tree = ast.parse(source, filename=rel)
        except Exception:
            continue
        blocks = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if getattr(node, "decorator_list", None):
                    dec_starts = [d.lineno for d in node.decorator_list if hasattr(d, "lineno")]
                    if dec_starts:
                        start = min(start, min(dec_starts))
                if start is not None and end is not None:
                    kind = "class" if isinstance(node, ast.ClassDef) else "func"
                    blocks.append({"kind": kind, "name": node.name, "start": start, "end": end})
        blocks.sort(key=lambda b: (b["end"] - b["start"], b["start"]))
        scopes_by_file[rel] = blocks
    return scopes_by_file


def compress_lines(lines: set[int]) -> list[list[int]]:
    if not lines:
        return []
    ordered = sorted(lines)
    ranges: list[list[int]] = []
    s = e = ordered[0]
    for ln in ordered[1:]:
        if ln == e + 1:
            e = ln
        else:
            ranges.append([s, e])
            s = e = ln
    ranges.append([s, e])
    return ranges


def case_to_pytest_id(case: unittest.TestCase, repo_root: Path) -> str:
    cls = case.__class__
    mod = sys.modules.get(cls.__module__)
    if mod and getattr(mod, "__file__", None):
        try:
            rel = Path(mod.__file__).resolve().relative_to(repo_root).as_posix()
            return f"{rel}::{cls.__qualname__}::{case._testMethodName}"
        except ValueError:
            pass
    return case.id()


def class_to_pytest_prefix(cls: type, repo_root: Path) -> str:
    mod = sys.modules.get(cls.__module__)
    if mod and getattr(mod, "__file__", None):
        try:
            rel = Path(mod.__file__).resolve().relative_to(repo_root).as_posix()
            return f"{rel}::{cls.__qualname__}::"
        except ValueError:
            pass
    return f"{cls.__module__}.{cls.__qualname__}."


def trace_test_files(args: tuple[str, list[str]]) -> dict:
    repo_str, test_rel_paths = args
    os.environ.pop("SKIP_INTEGRATION", None)
    repo_root = Path(repo_str).resolve()
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    repo_prefix = str(repo_root) + os.sep
    repo_prefix_len = len(repo_prefix)

    active_lines: dict[str, set[int]] = defaultdict(set)
    active_fixtures: set[str] = set()

    orig_builtins_open = builtins.open
    orig_io_open = io.open
    orig_path_open = Path.open

    def record_path(file_arg) -> None:
        try:
            if isinstance(file_arg, (str, os.PathLike)):
                p = os.path.abspath(os.fspath(file_arg))
                if p.startswith(repo_prefix):
                    rel = p[repo_prefix_len:].replace(os.sep, "/")
                    if not rel.endswith(".py"):
                        active_fixtures.add(rel)
        except Exception:
            pass

    def hooked_builtins_open(file, *args, **kwargs):
        record_path(file)
        return orig_builtins_open(file, *args, **kwargs)

    def hooked_io_open(file, *args, **kwargs):
        record_path(file)
        return orig_io_open(file, *args, **kwargs)

    def hooked_path_open(self, *args, **kwargs):
        record_path(self)
        return orig_path_open(self, *args, **kwargs)

    builtins.open = hooked_builtins_open
    io.open = hooked_io_open
    Path.open = hooked_path_open

    TOOL_ID = sys.monitoring.COVERAGE_ID
    sys.monitoring.use_tool_id(TOOL_ID, "blast_mapper")
    code_to_rel: dict[object, str | None] = {}
    seen_lines_this_phase: set[tuple[object, int]] = set()

    def on_line(code, line_number):
        rel = code_to_rel.get(code)
        if rel is None:
            if code in code_to_rel:
                return None
            fn = code.co_filename
            if fn.startswith(repo_prefix):
                rel = fn[repo_prefix_len:].replace(os.sep, "/")
                if rel.startswith("sqlglot/"):
                    code_to_rel[code] = rel
                else:
                    code_to_rel[code] = None
                    return None
            else:
                code_to_rel[code] = None
                return None
        key = (code, line_number)
        if key in seen_lines_this_phase:
            return sys.monitoring.DISABLE
        seen_lines_this_phase.add(key)
        active_lines[rel].add(line_number)
        return None

    sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.LINE)
    sys.monitoring.register_callback(TOOL_ID, sys.monitoring.events.LINE, on_line)

    import_lines: dict[str, set[int]] = defaultdict(set)
    import_fixtures: set[str] = set()
    file_import_lines: dict[str, dict[str, list[list[int]]]] = {}

    for rel_test in test_rel_paths:
        mod_name = rel_test[:-3].replace("/", ".")
        mod_lines: dict[str, set[int]] = defaultdict(set)
        active_lines = mod_lines
        active_fixtures = import_fixtures
        seen_lines_this_phase.clear()
        sys.monitoring.restart_events()
        try:
            importlib.import_module(mod_name)
        except Exception:
            pass
        for k, v in mod_lines.items():
            import_lines[k].update(v)
        if mod_lines:
            file_import_lines[rel_test] = {
                k: compress_lines(v) for k, v in sorted(mod_lines.items())
            }

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for rel_test in test_rel_paths:
        mod_name = rel_test[:-3].replace("/", ".")
        if mod_name in sys.modules:
            suite.addTests(loader.loadTestsFromModule(sys.modules[mod_name]))

    cases = collect_unittest_cases(suite)
    test_coverage: dict[str, dict] = {}
    class_setup_map: dict[str, dict[str, list[list[int]]]] = {}
    seen_classes: set[type] = set()

    for case in cases:
        if case._testMethodName in SKIP_SLOW_METHODS:
            continue
        pytest_id = case_to_pytest_id(case, repo_root)
        cls = case.__class__

        if cls not in seen_classes:
            seen_classes.add(cls)
            cls_lines: dict[str, set[int]] = defaultdict(set)
            cls_fixtures: set[str] = set()
            active_lines = cls_lines
            active_fixtures = cls_fixtures
            seen_lines_this_phase.clear()
            sys.monitoring.restart_events()
            if hasattr(cls, "setUpClass"):
                try:
                    cls.setUpClass()
                except Exception:
                    pass
            if cls_lines:
                cls_prefix = class_to_pytest_prefix(cls, repo_root)
                class_setup_map[cls_prefix] = {
                    k: compress_lines(v) for k, v in sorted(cls_lines.items())
                }

        t_lines: dict[str, set[int]] = defaultdict(set)
        t_fixtures: set[str] = set()
        active_lines = t_lines
        active_fixtures = t_fixtures
        seen_lines_this_phase.clear()
        sys.monitoring.restart_events()

        try:
            case.setUp()
            getattr(case, case._testMethodName)()
            case.tearDown()
        except Exception:
            pass

        test_coverage[pytest_id] = {
            "lines": {k: compress_lines(v) for k, v in sorted(t_lines.items()) if v},
            "fixtures": sorted(t_fixtures),
        }

    for cls in seen_classes:
        if hasattr(cls, "tearDownClass"):
            try:
                cls.tearDownClass()
            except Exception:
                pass

    sys.monitoring.set_events(TOOL_ID, 0)
    sys.monitoring.free_tool_id(TOOL_ID)
    builtins.open = orig_builtins_open
    io.open = orig_io_open
    Path.open = orig_path_open

    return {
        "import_lines": {k: list(v) for k, v in import_lines.items()},
        "file_import_lines": file_import_lines,
        "class_setup_lines": class_setup_map,
        "import_fixtures": list(import_fixtures),
        "tests": test_coverage,
    }


def main() -> None:
    t0 = time.time()
    repo_root = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()

    all_test_files = sorted(
        p.relative_to(repo_root).as_posix() for p in (repo_root / "tests").rglob("test_*.py")
    )
    all_test_files.sort(key=lambda r: -(repo_root / r).stat().st_size)

    workers = min(4, os.cpu_count() or 2)
    buckets: list[list[str]] = [[] for _ in range(workers)]
    for i, rel_test in enumerate(all_test_files):
        buckets[i % workers].append(rel_test)

    print(f"Tracing {len(all_test_files)} test files across {workers} workers...", file=sys.stderr, flush=True)

    merged_import_lines: dict[str, set[int]] = defaultdict(set)
    merged_file_imports: dict[str, dict[str, list[list[int]]]] = {}
    merged_class_setup: dict[str, dict[str, list[list[int]]]] = {}
    merged_import_fixtures: set[str] = set()
    merged_tests: dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(trace_test_files, [(str(repo_root), b) for b in buckets if b]):
            for k, lns in res["import_lines"].items():
                merged_import_lines[k].update(lns)
            merged_file_imports.update(res["file_import_lines"])
            merged_class_setup.update(res["class_setup_lines"])
            merged_import_fixtures.update(res["import_fixtures"])
            merged_tests.update(res["tests"])

    payload = {
        "import_lines": {k: compress_lines(v) for k, v in sorted(merged_import_lines.items())},
        "file_import_lines": merged_file_imports,
        "class_setup_lines": merged_class_setup,
        "import_fixtures": sorted(merged_import_fixtures),
        "ast_scopes": extract_ast_scopes(repo_root),
        "tests": merged_tests,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Mapped {len(merged_tests)} tests in {time.time() - t0:.1f}s -> {out_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()