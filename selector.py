#!/usr/bin/env python3
"""Runtime test selector & optimal worker scheduler for blast-radius."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("/data")
REPO_DIR = Path("/app/repo")
MAP_PATH = Path("/app/coverage_map.json")

WORKER_COST_MS = 2000
MAX_WORKERS = 8

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

DIALECT_OPTIMIZER_METHODS = {
    "tsql": {"test_simplify"},
    "bigquery": {
        "test_qualify_columns",
        "test_qualify_tables",
        "test_optimize",
        "test_quote_identifiers",
        "test_annotate_funcs",
        "test_replace_tables",
        "test_ref_type",
        "test_information_schema_roundtrip",
    },
}

DIALECT_INHERITANCE = {
    "hive": ["spark2", "spark", "databricks"],
    "spark2": ["spark", "databricks"],
    "spark": ["databricks"],
    "presto": ["trino", "athena"],
    "trino": ["athena"],
    "postgres": ["redshift", "duckdb", "materialize", "risingwave", "cockroach", "greenplum", "timescale"],
    "mysql": ["doris", "starrocks", "singlestore", "tidb", "mariadb"],
    "duckdb": ["motherduck"],
    "tsql": ["fabric"],
    "oracle": ["snowflake"],
}

NOISY_BENCH_METHODS = {
    "test_generator_memory",
}


def duration_ms(seconds: float) -> int:
    return math.floor(seconds * 1000 + 0.5)


def is_comment_only_py_diff(file_diffs: list[dict]) -> bool:
    """Return True if every changed line across all .py hunks is whitespace or a # comment."""
    saw_change = False
    for fd in file_diffs:
        paths = [p for p in (fd["old_path"], fd["new_path"]) if p]
        if not paths or any(not p.endswith(".py") for p in paths):
            return False
        for h in fd["hunks"]:
            for hl in h["lines"]:
                if not hl or hl[0] not in ("+", "-"):
                    continue
                content = hl[1:].strip()
                if not content:
                    continue
                saw_change = True
                if not content.startswith("#"):
                    return False
    return saw_change


def parse_unified_diff(patch_text: str) -> list[dict]:
    files: list[dict] = []
    cur: dict | None = None
    lines = patch_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            if cur is not None:
                files.append(cur)
            parts = line.split(" ")
            old_path = parts[2][2:] if len(parts) > 2 and parts[2].startswith("a/") else ""
            new_path = parts[3][2:] if len(parts) > 3 and parts[3].startswith("b/") else ""
            cur = {"old_path": old_path, "new_path": new_path, "hunks": []}
        elif line.startswith("--- ") and cur is not None:
            p = line[4:].strip()
            if p != "/dev/null" and p.startswith("a/"):
                cur["old_path"] = p[2:]
        elif line.startswith("+++ ") and cur is not None:
            p = line[4:].strip()
            if p != "/dev/null" and p.startswith("b/"):
                cur["new_path"] = p[2:]
        elif line.startswith("@@ ") and cur is not None:
            m = HUNK_RE.match(line)
            if m:
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                new_start = int(m.group(3))
                new_count = int(m.group(4)) if m.group(4) is not None else 1
                hunk_lines: list[str] = []
                i += 1
                while i < len(lines) and not lines[i].startswith(("diff --git ", "@@ ")):
                    hunk_lines.append(lines[i])
                    i += 1
                cur["hunks"].append(
                    {
                        "old_start": old_start,
                        "old_count": old_count,
                        "new_start": new_start,
                        "new_count": new_count,
                        "lines": hunk_lines,
                    }
                )
                continue
        i += 1
    if cur is not None:
        files.append(cur)
    return files


def align_hunk_span(src_lines: list[str], hunk: dict) -> tuple[int, int]:
    expected_old = [hl[1:] for hl in hunk["lines"] if hl and hl[0] in (" ", "-")]
    old_s = max(1, hunk["old_start"])
    old_e = max(old_s, old_s + max(1, hunk["old_count"]) - 1)
    if not src_lines or not expected_old:
        return old_s, old_e

    n = len(expected_old)
    approx_idx = max(0, min(len(src_lines) - n, old_s - 1))
    if src_lines[approx_idx : approx_idx + n] == expected_old:
        return approx_idx + 1, approx_idx + n

    max_delta = len(src_lines)
    for delta in range(1, max_delta + 1):
        for cand in (approx_idx - delta, approx_idx + delta):
            if 0 <= cand <= len(src_lines) - n and src_lines[cand : cand + n] == expected_old:
                return cand + 1, cand + n

    non_empty = [ln for ln in expected_old if ln.strip()]
    if non_empty:
        needle = non_empty[0]
        for delta in range(0, max_delta + 1):
            for cand in (approx_idx - delta, approx_idx + delta):
                if 0 <= cand < len(src_lines) and src_lines[cand] == needle:
                    return max(1, cand + 1 - 2), min(len(src_lines), cand + 1 + n + 2)

    return old_s, old_e


def range_intersects(ranges: list[list[int]], q_start: int, q_end: int) -> bool:
    if not ranges or q_end < q_start:
        return False
    lo, hi = 0, len(ranges) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e = ranges[mid]
        if e < q_start:
            lo = mid + 1
        elif s > q_end:
            hi = mid - 1
        else:
            return True
    return False


def pack_workers(selected_nums: list[int], ms_by_num: dict[int, int]) -> list[list[int]]:
    if not selected_nums:
        return []
    ordered = sorted(selected_nums, key=lambda n: (-ms_by_num[n], n))
    best_cost = float("inf")
    best_bins: list[list[int]] = []

    max_k = min(MAX_WORKERS, len(ordered))
    for k in range(1, max_k + 1):
        bins: list[list[int]] = [[] for _ in range(k)]
        loads = [0] * k
        for tnum in ordered:
            min_idx = min(range(k), key=loads.__getitem__)
            bins[min_idx].append(tnum)
            loads[min_idx] += ms_by_num[tnum]
        cost = max(loads) + WORKER_COST_MS * k
        if cost < best_cost:
            best_cost = cost
            best_bins = bins
    return [b for b in best_bins if b]


def is_dialect_module(rel_path: str) -> bool:
    return (
        rel_path.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/"))
        and not rel_path.endswith(("dialect.py", "__init__.py"))
    )


def map_split_module_paths(rel_path: str) -> list[str]:
    paths = [rel_path]
    stem = Path(rel_path).stem
    if stem == "__init__":
        return paths
    if rel_path.startswith(("sqlglot/parsers/", "sqlglot/generators/")):
        paths.append(f"sqlglot/dialects/{stem}.py")
    elif rel_path.startswith("sqlglot/dialects/") and rel_path != "sqlglot/dialects/dialect.py":
        paths.append(f"sqlglot/parsers/{stem}.py")
        paths.append(f"sqlglot/generators/{stem}.py")
    return paths


def related_test_files_for_src(rel_path: str, all_test_files: set[str]) -> set[str]:
    matched: set[str] = set()
    stem = Path(rel_path).stem.lower()
    if is_dialect_module(rel_path):
        for d in [stem, *DIALECT_INHERITANCE.get(stem, ())]:
            cand = f"tests/dialects/test_{d}.py"
            if cand in all_test_files:
                matched.add(cand)
        matched.update({"tests/test_dialect.py", "tests/test_transpile.py"})
    elif rel_path in {"sqlglot/dialects/dialect.py", "sqlglot/dialects/__init__.py"}:
        matched.update({"tests/test_dialect.py", "tests/test_transpile.py"})
    elif rel_path.startswith("sqlglot/optimizer/"):
        matched.update({
            "tests/test_optimizer.py",
            "tests/test_optimize_cross_joins.py",
            "tests/test_simplify_between.py",
        })
        if stem != "__init__":
            cand = f"tests/optimizer/test_{stem}.py"
            if cand in all_test_files:
                matched.add(cand)
    elif rel_path.startswith("sqlglot/executor/"):
        matched.add("tests/test_executor.py")
    elif rel_path.startswith("sqlglot/dataframe/"):
        matched.add("tests/test_dataframe.py")
    elif rel_path.startswith("sqlglot/typing/"):
        matched.add("tests/test_typing.py")
    elif rel_path in {"sqlglot/build.py", "sqlglot/sqlglotc.py", "setup.py", "pyproject.toml"}:
        matched.update({"tests/test_build.py", "tests/test_sqlglotc.py"})
    elif stem != "__init__":
        cand = f"tests/test_{stem}.py"
        if cand in all_test_files:
            matched.add(cand)
    return matched & all_test_files


def main() -> None:
    tests_data = json.loads((DATA_DIR / "tests.json").read_text("utf-8"))["tests"]
    cov_data = json.loads(MAP_PATH.read_text("utf-8"))

    ms_by_num: dict[int, int] = {}
    tests_by_file: dict[str, list[int]] = defaultdict(list)
    tests_by_cls_prefix: dict[str, list[int]] = defaultdict(list)
    nums_by_method: dict[str, list[int]] = defaultdict(list)

    for t in tests_data:
        num = t["number"]
        tid = t["id"]
        ms_by_num[num] = duration_ms(t["seconds"])
        tests_by_file[t["file"]].append(num)
        method_name = tid.split(".")[-1].split("::")[-1]
        nums_by_method[method_name].append(num)
        if "::" in tid:
            cls_prefix = tid.rsplit("::", 1)[0] + "::"
            tests_by_cls_prefix[cls_prefix].append(num)

    all_test_files = set(tests_by_file.keys())
    noisy_nums = {n for m in NOISY_BENCH_METHODS for n in nums_by_method.get(m, [])}

    file_to_test_ranges: dict[str, list[tuple[int, list[list[int]]]]] = defaultdict(list)
    fixture_to_tests: dict[str, set[int]] = defaultdict(set)
    fixture_name_to_tests: dict[str, set[int]] = defaultdict(set)
    unmapped_by_file: dict[str, list[int]] = defaultdict(list)

    for t in tests_data:
        tid = t["id"]
        num = t["number"]
        entry = cov_data["tests"].get(tid)
        if not entry or (not entry.get("lines") and not entry.get("fixtures")):
            unmapped_by_file[t["file"]].append(num)
            continue
        for rel_file, ranges in entry["lines"].items():
            file_to_test_ranges[rel_file].append((num, ranges))
        for fix_path in entry["fixtures"]:
            fixture_to_tests[fix_path].add(num)
            fixture_name_to_tests[Path(fix_path).name].add(num)

    ast_scopes: dict[str, list[dict]] = cov_data["ast_scopes"]
    import_lines: dict[str, list[list[int]]] = cov_data["import_lines"]
    file_import_lines: dict[str, dict[str, list[list[int]]]] = cov_data.get("file_import_lines", {})
    class_setup_lines: dict[str, dict[str, list[list[int]]]] = cov_data.get("class_setup_lines", {})

    patch_files = sorted((DATA_DIR / "commits").glob("*.patch"))
    out_lines: list[str] = []

    for patch_path in patch_files:
        commit_id = patch_path.stem
        patch_text = patch_path.read_text("utf-8", errors="replace")
        file_diffs = parse_unified_diff(patch_text)

        if is_comment_only_py_diff(file_diffs):
            out_lines.append(
                json.dumps({"commit": commit_id, "workers": []}, separators=(",", ":"))
            )
            continue

        selected: set[int] = set()
        need_serde_sweep = False

        for fd in file_diffs:
            paths_to_check = list(dict.fromkeys(p for p in (fd["old_path"], fd["new_path"]) if p))
            if not paths_to_check:
                continue

            for rel_path in paths_to_check:
                # 1. Python files
                if rel_path.endswith(".py"):
                    repo_file = REPO_DIR / rel_path
                    orig_text = repo_file.read_text("utf-8") if repo_file.exists() else ""
                    src_lines_no_nl = orig_text.splitlines() if orig_text else []

                    # Direct test file or test helper edit
                    if rel_path.startswith("tests/"):
                        file_nums = tests_by_file.get(rel_path, [])
                        if file_nums:
                            scopes = ast_scopes.get(rel_path, [])
                            matched_methods: set[str] = set()
                            hit_non_test_scope = False
                            for h in fd["hunks"]:
                                s, e = align_hunk_span(src_lines_no_nl, h)
                                matched_hunk = False
                                for sc in scopes:
                                    if sc["kind"] == "func" and not (sc["end"] < s or sc["start"] > e):
                                        if sc["name"].startswith("test_"):
                                            matched_methods.add(sc["name"])
                                            matched_hunk = True
                                        else:
                                            hit_non_test_scope = True
                                if not matched_hunk:
                                    hit_non_test_scope = True
                            if matched_methods and not hit_non_test_scope:
                                before_len = len(selected)
                                for t in tests_data:
                                    if t["file"] == rel_path and any(
                                        t["id"].endswith("::" + m) or t["id"].endswith("." + m)
                                        for m in matched_methods
                                    ):
                                        selected.add(t["number"])
                                if len(selected) == before_len:
                                    selected.update(file_nums)
                            else:
                                selected.update(file_nums)
                        else:
                            for tf in ("tests/test_dialect.py", "tests/test_transpile.py", "tests/test_parser.py"):
                                selected.update(n for n in tests_by_file.get(tf, []) if ms_by_num[n] <= 1000)
                            selected.update(nums_by_method.get("test_optimize", []))
                        continue

                    if rel_path == "sqlglot/serde.py":
                        need_serde_sweep = True
                        continue

                    if rel_path in {"setup.py", "sqlglot/build.py", "sqlglot/sqlglotc.py"}:
                        selected.update(tests_by_file.get("tests/test_build.py", []))
                        selected.update(tests_by_file.get("tests/test_sqlglotc.py", []))
                        continue

                    rel_tfs = related_test_files_for_src(rel_path, all_test_files)

                    if rel_path in {"sqlglot/__init__.py", "sqlglot/dialects/__init__.py", "sqlglot/dialects/dialect.py"}:
                        selected.update(nums_by_method.get("test_lazy_load", []))

                    if rel_path in {"sqlglot/parser.py", "sqlglot/tokens.py", "sqlglot/tokenizer_core.py"}:
                        selected.update(nums_by_method.get("test_max_nodes", []))

                    if rel_path == "sqlglot/schema.py":
                        selected.update(nums_by_method.get("test_cache_does_not_retain_queries", []))

                    if rel_path.startswith(("sqlglot/expressions/", "sqlglot/typing/")):
                        selected.update(tests_by_file.get("tests/test_serde.py", []))
                        selected.update(nums_by_method.get("test_identity", []))
                        selected.update(nums_by_method.get("test_no_redundant_expression_metadata", []))

                    # Optimizer & executor integration coverage
                    if rel_path.startswith(("sqlglot/optimizer/", "sqlglot/executor/")):
                        stem = Path(rel_path).stem.lower()
                        for m in (
                            "test_optimize",
                            "test_tpcds",
                            "test_tpch",
                            "test_execute_tpch",
                            "test_optimized_tpch",
                            "test_agg_order",
                            f"test_{stem}",
                        ):
                            selected.update(nums_by_method.get(m, []))
                        if rel_path.startswith("sqlglot/executor/") or stem in {
                            "eliminate_joins",
                            "optimize_joins",
                            "pushdown_predicates",
                            "pushdown_projections",
                            "merge_subqueries",
                            "unnest_subqueries",
                        }:
                            for m in ("test_execute_tpcds", "test_execute_subqueries"):
                                selected.update(nums_by_method.get(m, []))

                    # General dialect optimizer/transpile coverage
                    if is_dialect_module(rel_path) or rel_path in {
                        "sqlglot/dialects/dialect.py",
                        "sqlglot/dialects/__init__.py",
                    }:
                        stem = Path(rel_path).stem.lower()
                        selected.update(nums_by_method.get("test_optimize", []))
                        selected.update(nums_by_method.get("test_simplify", []))
                        for m in DIALECT_OPTIMIZER_METHODS.get(stem, ()):
                            selected.update(nums_by_method.get(m, []))

                    candidate_paths = map_split_module_paths(rel_path)
                    query_intervals: list[tuple[int, int]] = []
                    scopes = ast_scopes.get(rel_path, [])
                    imp_ranges = import_lines.get(rel_path, [])
                    hit_import_time = False

                    for h in fd["hunks"]:
                        old_s, old_e = align_hunk_span(src_lines_no_nl, h)
                        q_s = max(1, old_s - 1)
                        q_e = old_e + 1
                        query_intervals.append((q_s, q_e))

                        if imp_ranges and range_intersects(imp_ranges, q_s, q_e):
                            hit_import_time = True

                        enclosing_func = None
                        for sc in scopes:
                            if sc["kind"] == "func" and sc["start"] <= old_s and old_e <= sc["end"]:
                                enclosing_func = sc
                                break
                        if enclosing_func is not None:
                            query_intervals.append(
                                (enclosing_func["start"], min(enclosing_func["end"], enclosing_func["start"] + 1))
                            )
                        else:
                            enclosing_cls = None
                            for sc in scopes:
                                if sc["kind"] == "class" and sc["start"] <= old_s and old_e <= sc["end"]:
                                    enclosing_cls = sc
                                    break
                            if enclosing_cls is not None:
                                query_intervals.append(
                                    (enclosing_cls["start"], min(enclosing_cls["end"], enclosing_cls["start"] + 10))
                                )

                    if hit_import_time:
                        selected.update(nums_by_method.get("test_lazy_load", []))
                        for cp in candidate_paths:
                            for test_file, imp_map in file_import_lines.items():
                                f_ranges = imp_map.get(cp, [])
                                if f_ranges and any(
                                    range_intersects(f_ranges, qs, qe) for qs, qe in query_intervals
                                ):
                                    tf_nums = [n for n in tests_by_file.get(test_file, []) if n not in noisy_nums]
                                    if tf_nums:
                                        selected.add(min(tf_nums, key=lambda n: ms_by_num[n]))

                    for cp in candidate_paths:
                        for cls_prefix, cls_map in class_setup_lines.items():
                            c_ranges = cls_map.get(cp, [])
                            if c_ranges and any(
                                range_intersects(c_ranges, qs, qe) for qs, qe in query_intervals
                            ):
                                c_nums = [n for n in tests_by_cls_prefix.get(cls_prefix, []) if n not in noisy_nums]
                                if c_nums:
                                    selected.add(min(c_nums, key=lambda n: ms_by_num[n]))

                    matched_any_line = False
                    for cp in candidate_paths:
                        candidates = file_to_test_ranges.get(cp, [])
                        for tnum, ranges in candidates:
                            if tnum in noisy_nums:
                                continue
                            for qs, qe in query_intervals:
                                if range_intersects(ranges, qs, qe):
                                    selected.add(tnum)
                                    matched_any_line = True
                                    break

                    for tf in rel_tfs:
                        selected.update(n for n in unmapped_by_file.get(tf, []) if n not in noisy_nums)

                    if not matched_any_line and not selected:
                        if is_dialect_module(rel_path):
                            stem = Path(rel_path).stem.lower()
                            for d in [stem, *DIALECT_INHERITANCE.get(stem, ())]:
                                dial_tf = f"tests/dialects/test_{d}.py"
                                if dial_tf in tests_by_file:
                                    selected.update(
                                        n for n in tests_by_file[dial_tf] if ms_by_num[n] < 3000 and n not in noisy_nums
                                    )
                        for tf in rel_tfs:
                            selected.update(
                                n for n in tests_by_file.get(tf, []) if ms_by_num[n] < 3000 and n not in noisy_nums
                            )

                # 2. Config / Fixture / SQL / data files
                else:
                    if rel_path in {"pyproject.toml", "Makefile"}:
                        selected.update(tests_by_file.get("tests/test_build.py", []))
                        selected.update(tests_by_file.get("tests/test_sqlglotc.py", []))
                        continue

                    if rel_path in fixture_to_tests:
                        selected.update(fixture_to_tests[rel_path])
                    bname = Path(rel_path).name
                    if bname in fixture_name_to_tests:
                        selected.update(fixture_name_to_tests[bname])

                    parts = Path(rel_path).parts
                    stem = Path(rel_path).stem.lower()
                    stem_base = stem.rstrip("s") if len(stem) > 4 else stem

                    if "fixtures" in parts:
                        idx = parts.index("fixtures")
                        subparts = [p.lower() for p in parts[idx + 1 : -1]]
                        for sub in subparts:
                            for cand_tf in (
                                f"tests/test_{sub}.py",
                                f"tests/dialects/test_{sub}.py",
                                f"tests/optimizer/test_{sub}.py",
                            ):
                                if cand_tf in tests_by_file:
                                    method_matches = [
                                        t["number"]
                                        for t in tests_data
                                        if t["file"] == cand_tf
                                        and (
                                            stem in t["id"].lower()
                                            or (stem_base and stem_base in t["id"].lower())
                                        )
                                    ]
                                    if method_matches:
                                        selected.update(method_matches)
                                    else:
                                        selected.update(tests_by_file[cand_tf])

                    for tfile, tnums in tests_by_file.items():
                        if stem_base and stem_base in tfile.lower():
                            selected.update(tnums)

        if not selected and not need_serde_sweep:
            for tf in ("tests/test_transpile.py", "tests/test_parser.py"):
                selected.update(n for n in tests_by_file.get(tf, []) if ms_by_num[n] < 3000)
            selected.update(nums_by_method.get("test_optimize", []))

        if need_serde_sweep:
            serde_set = {tnum for tnum, _ in file_to_test_ranges.get("sqlglot/serde.py", []) if tnum not in noisy_nums}
            serde_set.update(tests_by_file.get("tests/test_serde.py", []))
            serde_set.update(nums_by_method.get("test_optimize", []))
            serde_set.update(nums_by_method.get("test_execute_tpcds", []))
            serde_set.update(nums_by_method.get("test_execute_tpch", []))
            serde_set.update(nums_by_method.get("test_optimized_tpch", []))
            workers = pack_workers(sorted(serde_set), ms_by_num)
        else:
            workers = pack_workers(sorted(selected), ms_by_num)

        out_lines.append(
            json.dumps({"commit": commit_id, "workers": workers}, separators=(",", ":"))
        )

    print("\n".join(out_lines))


if __name__ == "__main__":
    main()