"""AST Codemap tool: extract code structure for documentation and context.

Parses source files using AST (Python) or regex patterns (other languages)
to extract a structural overview: classes, functions, exports, routes, models.

This gives agents project-awareness without reading every file line-by-line.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path
from typing import Any

from tools.base import ToolDef
from tools.injection import _is_path_allowed

logger = logging.getLogger(__name__)

# Max files to scan per invocation
_MAX_FILES = 100
_MAX_FILE_SIZE = 200_000  # 200KB


def _parse_python(path: Path, content: str) -> dict:
    """Parse Python file using AST."""
    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError:
        return {"error": "SyntaxError"}

    result: dict[str, list] = {
        "classes": [],
        "functions": [],
        "imports": [],
        "exports": [],  # __all__ if defined
    }

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not n.name.startswith("_")
            ]
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.unparse(base))
            cls_info = {"name": node.name, "line": node.lineno}
            if bases:
                cls_info["bases"] = bases
            if methods:
                cls_info["methods"] = methods
            # Check for decorators (route, model, etc.)
            for dec in node.decorator_list:
                dec_str = ast.unparse(dec) if hasattr(ast, "unparse") else ""
                if dec_str:
                    cls_info.setdefault("decorators", []).append(dec_str)
            result["classes"].append(cls_info)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_info: dict[str, Any] = {
                "name": node.name,
                "line": node.lineno,
                "async": isinstance(node, ast.AsyncFunctionDef),
            }
            # Extract parameters
            args = []
            for arg in node.args.args:
                if arg.arg != "self":
                    args.append(arg.arg)
            if args:
                fn_info["params"] = args[:6]  # Limit to 6 params
            # Check decorators for routes
            for dec in node.decorator_list:
                dec_str = ast.unparse(dec) if hasattr(ast, "unparse") else ""
                if dec_str:
                    fn_info.setdefault("decorators", []).append(dec_str)
            result["functions"].append(fn_info)

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                result["imports"].append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append(alias.name)

        elif isinstance(node, ast.Assign):
            # Check for __all__
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        result["exports"] = [
                            elt.value for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]

    # Clean up empty lists
    return {k: v for k, v in result.items() if v}


def _parse_typescript(content: str) -> dict:
    """Parse TypeScript/JavaScript using regex patterns."""
    result: dict[str, list] = {
        "exports": [],
        "classes": [],
        "functions": [],
        "interfaces": [],
        "routes": [],
    }

    # Exports
    for m in re.finditer(r"export\s+(default\s+)?(class|function|const|interface|type|enum)\s+(\w+)", content):
        kind = m.group(2)
        name = m.group(3)
        is_default = bool(m.group(1))
        result["exports"].append({"name": name, "kind": kind, "default": is_default})

    # Classes
    for m in re.finditer(r"class\s+(\w+)(?:\s+extends\s+(\w+))?", content):
        cls = {"name": m.group(1)}
        if m.group(2):
            cls["extends"] = m.group(2)
        result["classes"].append(cls)

    # Functions (top-level)
    for m in re.finditer(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", content):
        result["functions"].append({"name": m.group(1)})

    # Arrow function exports
    for m in re.finditer(r"export\s+const\s+(\w+)\s*=\s*(?:async\s+)?\(", content):
        result["functions"].append({"name": m.group(1), "arrow": True})

    # Interfaces
    for m in re.finditer(r"interface\s+(\w+)", content):
        result["interfaces"].append({"name": m.group(1)})

    # Express/Fastify routes
    for m in re.finditer(r"(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)", content):
        result["routes"].append({"method": m.group(1).upper(), "path": m.group(2)})

    # Next.js API routes (from file structure)
    for m in re.finditer(r"export\s+(?:async\s+)?function\s+(GET|POST|PUT|DELETE|PATCH)\s*\(", content):
        result["routes"].append({"method": m.group(1), "handler": True})

    return {k: v for k, v in result.items() if v}


def _parse_go(content: str) -> dict:
    """Parse Go using regex patterns."""
    result: dict[str, list] = {
        "package": "",
        "structs": [],
        "functions": [],
        "interfaces": [],
    }

    # Package
    m = re.search(r"^package\s+(\w+)", content, re.MULTILINE)
    if m:
        result["package"] = m.group(1)

    # Structs
    for m in re.finditer(r"type\s+(\w+)\s+struct\s*\{", content):
        result["structs"].append({"name": m.group(1)})

    # Interfaces
    for m in re.finditer(r"type\s+(\w+)\s+interface\s*\{", content):
        result["interfaces"].append({"name": m.group(1)})

    # Functions (exported = capitalized)
    for m in re.finditer(r"func\s+(?:\((\w+)\s+\*?(\w+)\)\s+)?(\w+)\s*\(", content):
        fn: dict[str, Any] = {"name": m.group(3)}
        if m.group(2):
            fn["receiver"] = m.group(2)
        fn["exported"] = m.group(3)[0].isupper()
        result["functions"].append(fn)

    # HTTP handlers (net/http, gin, echo)
    for m in re.finditer(r"\.(?:GET|POST|PUT|DELETE|PATCH|Handle|HandleFunc)\s*\(\s*\"([^\"]+)\"", content):
        result.setdefault("routes", []).append({"path": m.group(1)})

    return {k: v for k, v in result.items() if v}


def _parse_generic(content: str, ext: str) -> dict:
    """Generic parsing for other languages using common patterns."""
    result: dict[str, list] = {
        "classes": [],
        "functions": [],
    }

    # Classes
    for m in re.finditer(r"(?:public\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?", content):
        cls: dict[str, Any] = {"name": m.group(1)}
        if m.group(2):
            cls["extends"] = m.group(2)
        result["classes"].append(cls)

    # Functions/methods
    for m in re.finditer(r"(?:pub(?:lic)?\s+)?(?:async\s+)?(?:fn|func|fun|def)\s+(\w+)\s*\(", content):
        result["functions"].append({"name": m.group(1)})

    return {k: v for k, v in result.items() if v}


_LANG_MAP = {
    ".py": ("python", _parse_python),
    ".ts": ("typescript", _parse_typescript),
    ".tsx": ("typescript", _parse_typescript),
    ".js": ("javascript", _parse_typescript),
    ".jsx": ("javascript", _parse_typescript),
    ".go": ("go", _parse_go),
}


def _scan_file(path: Path) -> dict | None:
    """Scan a single file and return its structure."""
    if path.stat().st_size > _MAX_FILE_SIZE:
        return None

    try:
        content = path.read_text(errors="replace")
    except Exception:
        return None

    ext = path.suffix.lower()
    parser = _LANG_MAP.get(ext)

    if parser:
        lang, parse_fn = parser
        if lang == "python":
            structure = parse_fn(path, content)
        else:
            structure = parse_fn(content)
    else:
        structure = _parse_generic(content, ext)

    if not structure or structure.get("error"):
        return None

    return structure


async def _codemap(params: dict[str, Any], **deps: Any) -> str:
    """Generate a structural codemap for a directory or file."""
    target = params.get("path", "")
    if not target:
        return "Error: path is required"

    path = Path(target).expanduser()
    if not _is_path_allowed(path):
        return f"Error: access denied for {path}"

    if not path.exists():
        return f"Error: path not found: {path}"

    # Supported extensions
    supported_exts = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt", ".cpp", ".c", ".h"}

    results: dict[str, dict] = {}

    if path.is_file():
        if path.suffix.lower() in supported_exts:
            structure = _scan_file(path)
            if structure:
                results[str(path)] = structure
    else:
        # Directory scan
        file_count = 0
        for root, dirs, files in os.walk(path):
            # Skip hidden dirs, node_modules, __pycache__, .git, vendor
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d not in (
                    "node_modules", "__pycache__", "vendor", "dist",
                    "build", ".git", "venv", "env", ".venv",
                )
            ]
            for fname in sorted(files):
                fpath = Path(root) / fname
                if fpath.suffix.lower() not in supported_exts:
                    continue
                if file_count >= _MAX_FILES:
                    break
                structure = _scan_file(fpath)
                if structure:
                    # Use relative path for readability
                    rel = str(fpath.relative_to(path))
                    results[rel] = structure
                    file_count += 1
            if file_count >= _MAX_FILES:
                break

    if not results:
        return f"No parseable source files found in {path}"

    # Format output
    lines = [f"# Codemap: {path}\n"]
    lines.append(f"Files scanned: {len(results)}\n")

    for filepath, structure in sorted(results.items()):
        lines.append(f"\n## {filepath}")

        if "package" in structure:
            lines.append(f"Package: {structure['package']}")

        if "exports" in structure:
            exports = structure["exports"]
            if isinstance(exports, list) and exports:
                if isinstance(exports[0], str):
                    lines.append(f"__all__: {exports}")
                else:
                    for exp in exports[:10]:
                        default = " (default)" if exp.get("default") else ""
                        lines.append(f"  export {exp.get('kind', '')} {exp['name']}{default}")

        if "classes" in structure:
            for cls in structure["classes"]:
                bases = f"({', '.join(cls['bases'])})" if cls.get("bases") else ""
                extends = f" extends {cls['extends']}" if cls.get("extends") else ""
                line_info = f" :L{cls['line']}" if cls.get("line") else ""
                lines.append(f"  class {cls['name']}{bases}{extends}{line_info}")
                if cls.get("decorators"):
                    lines.append(f"    decorators: {', '.join(cls['decorators'][:3])}")
                if cls.get("methods"):
                    lines.append(f"    methods: {', '.join(cls['methods'][:8])}")

        if "interfaces" in structure:
            for iface in structure["interfaces"]:
                lines.append(f"  interface {iface['name']}")

        if "structs" in structure:
            for s in structure["structs"]:
                lines.append(f"  struct {s['name']}")

        if "functions" in structure:
            for fn in structure["functions"]:
                async_tag = "async " if fn.get("async") else ""
                receiver = f"({fn['receiver']}) " if fn.get("receiver") else ""
                params_str = f"({', '.join(fn.get('params', []))})" if fn.get("params") else "()"
                line_info = f" :L{fn['line']}" if fn.get("line") else ""
                exported = " [exported]" if fn.get("exported") else ""
                lines.append(f"  {async_tag}fn {receiver}{fn['name']}{params_str}{line_info}{exported}")
                if fn.get("decorators"):
                    lines.append(f"    decorators: {', '.join(fn['decorators'][:3])}")

        if "routes" in structure:
            for r in structure["routes"]:
                method = r.get("method", "?")
                route_path = r.get("path", "handler")
                lines.append(f"  route: {method} {route_path}")

        if "imports" in structure:
            # Show only unique external imports (not stdlib)
            ext_imports = sorted(set(
                imp for imp in structure["imports"]
                if "." in imp or not imp.startswith("_")
            ))[:8]
            if ext_imports:
                lines.append(f"  imports: {', '.join(ext_imports)}")

    output = "\n".join(lines)
    if len(output) > 8000:
        output = output[:8000] + "\n... (truncated)"
    return output


CODEMAP = ToolDef(
    name="codemap",
    description=(
        "Generate a structural map of a codebase or file. Extracts classes, functions, "
        "exports, routes, and models using AST analysis. Use this to understand project "
        "structure before making changes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to a file or directory to scan",
            },
        },
        "required": ["path"],
    },
    fn=_codemap,
    category="search",
)
