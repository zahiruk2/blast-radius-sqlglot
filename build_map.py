#!/usr/bin/env python3
"""Ultra-fast build-time dynamic coverage + fixture + AST scope mapper for sqlglot."""

from __future__ import annotations

import ast
import builtins
import io
import json
import os
import sys
import threading
import time
import unittest
from collections import defaultdict
from pathlib import Path


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


def main() -> None:
    t0 = time.time()
    repo_root = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    repo_prefix = str(repo_root) + os.sep
    repo_prefix_len = len(repo_prefix)

    import_lines: dict[str, set[int]] = defaultdict(set)
    import_fixtures: set[str] = set()

    active_lines: dict[str, set[int]] = import_lines
    active_fixtures: set[str] = import_fixtures

    orig_builtins_open = builtins.open
    orig_io_open = io.open

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

    builtins.open = hooked_builtins_open
    io.open = hooked_io_open

    TOOL_ID = sys.monitoring.COVERAGE_ID
    sys.monitoring.use_tool_id(TOOL_ID, "blast_mapper")
    code_to_rel: dict[object, str | None] = {}
    seen_lines_this_phase: set[tuple[object, int]] = set()

    def on_line(code, line_number):
        rel = code_to_rel.get(code)
        if rel is None:
            if code in code_to_rel:
                return sys.monitoring.DISABLE
            fn = code.co_filename
            if fn.startswith(repo_prefix):
                rel = fn[repo_prefix_len:].replace(os.sep, "/")
                if rel.startswith("sqlglot/"):
                    code_to_rel[code] = rel
                else:
                    code_to_rel[code] = None
                    return sys.monitoring.DISABLE
            else:
                code_to_rel[code] = None
                return sys.monitoring.DISABLE
        key = (code, line_number)
        if key in seen_lines_this_phase:
            return sys.monitoring.DISABLE
        seen_lines_this_phase.add(key)
        active_lines[rel].add(line_number)
        return None

    sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.LINE)
    sys.monitoring.register_callback(TOOL_ID, sys.monitoring.events.LINE, on_line)

    print("Discovering tests...", file=sys.stderr, flush=True)
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(repo_root / "tests"), top_level_dir=str(repo_root))
    cases = collect_unittest_cases(suite)
    total = len(cases)
    print(f"Found {total} tests. Tracing...", file=sys.stderr, flush=True)

    current_test_name = "init"
    test_start_time = time.time()

    def watchdog() -> None:
        while True:
            time.sleep(2.0)
            elapsed = time.time() - test_start_time
            if elapsed > 4.0 and current_test_name != "done":
                print(f"  [slow >{elapsed:.0f}s] {current_test_name}", file=sys.stderr, flush=True)

    threading.Thread(target=watchdog, daemon=True).start()

    test_coverage: dict[str, dict] = {}
    class_setup_cache: dict[type, tuple[dict[str, set[int]], set[str]]] = {}
    seen_classes: set[type] = set()

    for idx, case in enumerate(cases, start=1):
        if idx % 300 == 0 or idx == total:
            print(f"  [{idx}/{total}] ({time.time() - t0:.1f}s)", file=sys.stderr, flush=True)
        test_id = case.id()
        current_test_name = test_id
        test_start_time = time.time()
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
            class_setup_cache[cls] = (cls_lines, cls_fixtures)

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

        cls_lines, cls_fixtures = class_setup_cache[cls]
        merged_lines: dict[str, list[list[int]]] = {}
        all_files = set(t_lines.keys()) | set(cls_lines.keys())
        for rel in sorted(all_files):
            combined = t_lines.get(rel, set()) | cls_lines.get(rel, set())
            if combined:
                merged_lines[rel] = compress_lines(combined)

        merged_fixtures = sorted(t_fixtures | cls_fixtures)
        test_coverage[test_id] = {
            "lines": merged_lines,
            "fixtures": merged_fixtures,
        }

    current_test_name = "done"
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

    payload = {
        "import_lines": {k: compress_lines(v) for k, v in sorted(import_lines.items())},
        "import_fixtures": sorted(import_fixtures),
        "ast_scopes": extract_ast_scopes(repo_root),
        "tests": test_coverage,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Mapped {len(test_coverage)} tests in {time.time() - t0:.1f}s -> {out_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()