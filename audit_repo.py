#!/usr/bin/env python3
"""
Repository auditor for Python projects.

Usage:
  python audit_repo.py /path/to/repo

Outputs:
  - audit_report.json
  - audit_report.md
"""

from __future__ import annotations
import ast, hashlib, io, json, os, re, sys, textwrap
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Set

# --------- Helpers ---------
PY_EXT = (".py",)
DATA_EXT = (".csv", ".parquet", ".feather", ".json", ".ndjson")
NB_EXT = (".ipynb",)
CFG_FILES = ("pyproject.toml", "requirements.txt", "requirements-dev.txt",
             "setup.cfg", "setup.py", ".env", ".env.example")
IGNORE_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "build", "dist", ".idea", ".vscode"}

@dataclass
class FunctionInfo:
    name: str
    lineno: int
    args: List[str]
    hash: str

@dataclass
class ClassInfo:
    name: str
    lineno: int
    methods: List[FunctionInfo]

@dataclass
class ModuleInfo:
    path: str
    rel: str
    loc: int
    doc: Optional[str]
    imports: List[str]
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    has_cli: bool
    has_fastapi: bool
    has_typer: bool
    has_argparse: bool
    models_found: List[str]

@dataclass
class RepoReport:
    root: str
    python: List[ModuleInfo]
    data_files: List[str]
    notebooks: List[str]
    config_files: Dict[str, bool]
    deps: Dict[str, List[str]]   # from pyproject/requirements
    duplicates: Dict[str, List[str]]  # func_hash -> [module:function@lineno]
    maybe_dead: List[str]        # def names with few/no references
    multiple_mains: List[str]    # files with if __name__ == '__main__'
    notes: List[str]

# --------- Parsing ---------
def normalize_fn_ast(fn: ast.AST) -> str:
    """Create a structure-hash for a function (ignore names, focus on shape)."""
    class StripNames(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):
            return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)
        def visit_arg(self, node: ast.arg):
            node.arg = "_"
            return node
        def visit_Attribute(self, node: ast.Attribute):
            node.attr = "_"
            return self.generic_visit(node)
        def visit_FunctionDef(self, node: ast.FunctionDef):
            node.name = "_"
            return self.generic_visit(node)
        visit_AsyncFunctionDef = visit_FunctionDef
    tree = StripNames().visit(ast.fix_missing_locations(fn))
    dumped = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha1(dumped.encode()).hexdigest()

MODEL_IMPORT_MARKERS = (
    "sklearn", "xgboost", "lightgbm", "catboost",
    "torch", "tensorflow", "statsmodels", "prophet"
)

def analyze_python_file(path: str, rel: str) -> ModuleInfo:
    src = open(path, "r", encoding="utf-8", errors="ignore").read()
    loc = sum(1 for _ in src.splitlines())
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ModuleInfo(path, rel, loc, None, [], [], [], False, False, False, False, [])

    doc = ast.get_docstring(tree)

    imports_set: Set[str] = set()
    models_set: Set[str] = set()
    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []

    has_cli = "if __name__ == '__main__':" in src
    has_typer = bool(re.search(r"\bTyper\(", src)) or "import typer" in src
    has_argparse = "import argparse" in src
    has_fastapi = "from fastapi" in src or "import fastapi" in src

    # --- imports & ML markers ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # import x, y.z as z2
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top:
                    imports_set.add(top)
                    if top in MODEL_IMPORT_MARKERS:
                        models_set.add(top)
        elif isinstance(node, ast.ImportFrom):
            # from a.b import c, d as e
            base = (node.module or "").split(".")[0] if getattr(node, "module", None) else ""
            if base:
                imports_set.add(base)
                if base in MODEL_IMPORT_MARKERS:
                    models_set.add(base)
            # (Optional) also consider aliases in case of "from sklearn.linear_model import LinearRegression"
            for alias in node.names:
                # Record the base already; no need to add per-alias package unless absolute
                # But if it's a top-level import (no base), treat alias as potential package
                if not base:
                    top = alias.name.split(".")[0]
                    if top:
                        imports_set.add(top)
                        if top in MODEL_IMPORT_MARKERS:
                            models_set.add(top)

    # --- defs/classes ---
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            functions.append(FunctionInfo(node.name, node.lineno, args, normalize_fn_ast(node)))
        elif isinstance(node, ast.ClassDef):
            methods: List[FunctionInfo] = []
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in b.args.args]
                    methods.append(FunctionInfo(b.name, b.lineno, args, normalize_fn_ast(b)))
            classes.append(ClassInfo(node.name, node.lineno, methods))

    return ModuleInfo(
        path=path,
        rel=rel,
        loc=loc,
        doc=doc,
        imports=sorted(imports_set),
        functions=functions,
        classes=classes,
        has_cli=has_cli,
        has_fastapi=has_fastapi,
        has_typer=has_typer,
        has_argparse=has_argparse,
        models_found=sorted(models_set),
    )


def walk_repo(root: str) -> Tuple[List[ModuleInfo], List[str], List[str], Dict[str, bool]]:
    py: List[ModuleInfo] = []
    data_files: List[str] = []
    notebooks: List[str] = []
    cfg: Dict[str, bool] = {c: False for c in CFG_FILES}
    for base, dirs, files in os.walk(root):
        # prune
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            p = os.path.join(base, f)
            rel = os.path.relpath(p, root)
            lf = f.lower()
            if lf.endswith(PY_EXT):
                py.append(analyze_python_file(p, rel))
            elif lf.endswith(DATA_EXT):
                data_files.append(rel)
            elif lf.endswith(NB_EXT):
                notebooks.append(rel)
            if f in cfg:
                cfg[f] = True
    return py, sorted(data_files), sorted(notebooks), cfg

def parse_requirements(root: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    # requirements*.txt
    for fn in ("requirements.txt", "requirements-dev.txt"):
        p = os.path.join(root, fn)
        if os.path.exists(p):
            pkgs = []
            for line in open(p, "r", encoding="utf-8", errors="ignore"):
                s = line.strip()
                if not s or s.startswith("#"): continue
                pkgs.append(s)
            out[fn] = pkgs
    # pyproject
    pyp = os.path.join(root, "pyproject.toml")
    if os.path.exists(pyp):
        try:
            import tomllib
            data = tomllib.load(open(pyp, "rb"))
            deps = data.get("project", {}).get("dependencies", []) or data.get("tool", {}).get("poetry", {}).get("dependencies", {})
            if isinstance(deps, dict):
                deps_list = [f"{k}{'' if isinstance(v, str) else ''}" for k,v in deps.items()]
            else:
                deps_list = deps
            out["pyproject.toml:project.dependencies"] = deps_list
        except Exception:
            out["pyproject.toml:project.dependencies"] = []
    return out

def find_duplicates(py: List[ModuleInfo]) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for m in py:
        for f in m.functions:
            buckets[f.hash].append(f"{m.rel}:{f.name}@{f.lineno}")
        for c in m.classes:
            for f in c.methods:
                buckets[f.hash].append(f"{m.rel}:{c.name}.{f.name}@{f.lineno}")
    return {h: locs for h,locs in buckets.items() if len(locs) > 1}

def estimate_dead_defs(py: List[ModuleInfo]) -> List[str]:
    # Best-effort: count references of function/class names across repo (excluding their own def line).
    code_map = {}
    name_locations = {}
    for m in py:
        try:
            with open(m.path, "r", encoding="utf-8", errors="ignore") as fh:
                code_map[m.rel] = fh.read()
        except Exception:
            code_map[m.rel] = ""
        for f in m.functions:
            name_locations.setdefault(f.name, []).append(m.rel)
        for c in m.classes:
            name_locations.setdefault(c.name, []).append(m.rel)
    maybe = []
    for name, files in name_locations.items():
        refcount = 0
        for rel, code in code_map.items():
            # skip actual definition lines by using a rough negative lookbehind
            hits = len(re.findall(rf"\b{name}\s*\(", code))
            refcount += hits
        if refcount <= 1:  # only its own definition likely
            maybe.append(name)
    return sorted(set(maybe))

def detect_multiple_mains(py: List[ModuleInfo]) -> List[str]:
    return sorted([m.rel for m in py if m.has_cli])

def gather_notes(report: RepoReport) -> List[str]:
    notes = []
    if not report.deps:
        notes.append("No deps found (missing requirements/pyproject).")
    if report.notebooks:
        notes.append("Jupyter notebooks present—pin kernels or export to scripts for CI.")
    if report.multiple_mains and len(report.multiple_mains) > 1:
        notes.append("Multiple CLI entrypoints detected; consolidate with Typer.")
    if any(m.has_fastapi for m in report.python) and not report.config_files.get(".env", False):
        notes.append("FastAPI detected but .env missing—add config management.")
    return notes

def generate_markdown(report: RepoReport) -> str:
    lines = []
    lines.append(f"# Audit summary for `{os.path.basename(report.root)}`\n")
    lines.append("## Inventory\n")
    lines.append(f"- Python modules: **{len(report.python)}**")
    lines.append(f"- Data files: **{len(report.data_files)}**")
    lines.append(f"- Notebooks: **{len(report.notebooks)}**")
    lines.append(f"- Config present: " + ", ".join([k for k,v in report.config_files.items() if v]) or "None")
    lines.append("\n## Dependencies\n")
    for k, v in report.deps.items():
        lines.append(f"- **{k}**:")
        for pkg in v:
            lines.append(f"  - {pkg}")
    lines.append("\n## Models/ML libs detected\n")
    counter = Counter([lib for m in report.python for lib in m.models_found])
    for lib, n in counter.most_common():
        lines.append(f"- {lib}: {n} files")
    if not counter:
        lines.append("- None detected")

    lines.append("\n## Potential issues\n")
    if report.duplicates:
        lines.append(f"- Duplicate function/method bodies: **{len(report.duplicates)}** groups")
    if report.maybe_dead:
        lines.append(f"- Possibly dead/unused defs: **{len(report.maybe_dead)}**")
    if report.multiple_mains:
        lines.append(f"- Files with `if __name__ == '__main__'`: {len(report.multiple_mains)}")
    lines.append("\n## Notes\n")
    for n in report.notes:
        lines.append(f"- {n}")

    lines.append("\n## Files\n")
    for m in sorted(report.python, key=lambda x: x.rel):
        flags = []
        if m.has_typer: flags.append("typer")
        if m.has_argparse: flags.append("argparse")
        if m.has_fastapi: flags.append("fastapi")
        lines.append(f"- `{m.rel}` (loc={m.loc}) [{' '.join(flags) if flags else ''}]")
        if m.functions:
            lines.append("  - functions: " + ", ".join(f.name for f in m.functions[:12]) + (" ..." if len(m.functions) > 12 else ""))
        if m.classes:
            lines.append("  - classes: " + ", ".join(c.name for c in m.classes[:12]) + (" ..." if len(m.classes) > 12 else ""))
    if report.data_files:
        lines.append("\n## Data files")
        for d in report.data_files[:200]:
            lines.append(f"- {d}")
        if len(report.data_files) > 200:
            lines.append(f"- ... (+{len(report.data_files)-200} more)")

    if report.duplicates:
        lines.append("\n## Duplicate bodies (same AST hash)")
        for h, locs in report.duplicates.items():
            lines.append(f"- {h[:10]}… :")
            for loc in locs:
                lines.append(f"  - {loc}")

    if report.maybe_dead:
        lines.append("\n## Possibly unused definitions")
        for n in report.maybe_dead[:300]:
            lines.append(f"- {n}")
        if len(report.maybe_dead) > 300:
            lines.append(f"- ... (+{len(report.maybe_dead)-300} more)")
    return "\n".join(lines)

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    root = os.path.abspath(root)
    py, data_files, notebooks, cfg = walk_repo(root)
    deps = parse_requirements(root)
    duplicates = find_duplicates(py)
    maybe_dead = estimate_dead_defs(py)
    multiple_mains = detect_multiple_mains(py)
    report = RepoReport(
        root=root,
        python=py,
        data_files=data_files,
        notebooks=notebooks,
        config_files=cfg,
        deps=deps,
        duplicates=duplicates,
        maybe_dead=maybe_dead,
        multiple_mains=multiple_mains,
        notes=[]
    )
    report.notes = gather_notes(report)
    with open(os.path.join(root, "audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    with open(os.path.join(root, "audit_report.md"), "w", encoding="utf-8") as f:
        f.write(generate_markdown(report))
    print("Wrote audit_report.json and audit_report.md")

if __name__ == "__main__":
    main()
