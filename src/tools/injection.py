"""Injection tools: bring external content into the context.

- web_fetch: Fetch content from a URL
- read_file: Read a local file
- glob_search: Find files by glob pattern
- grep_search: Search file contents by regex
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from tools.base import ToolDef

logger = logging.getLogger(__name__)

# ── URL cache: avoid re-fetching the same URL within a short window ──
_url_cache: dict[str, tuple[str, float]] = {}  # url → (content, timestamp)
_URL_CACHE_TTL = 300  # 5 minutes
_URL_CACHE_MAX = 50


def _get_cached(url: str) -> str | None:
    """Return cached content if fresh, else None."""
    entry = _url_cache.get(url)
    if entry and (time.monotonic() - entry[1]) < _URL_CACHE_TTL:
        logger.debug("URL cache hit: %s", url[:80])
        return entry[0]
    return None


def _set_cache(url: str, content: str) -> None:
    """Cache fetched content."""
    _url_cache[url] = (content, time.monotonic())
    # Evict oldest if over limit
    if len(_url_cache) > _URL_CACHE_MAX:
        oldest_key = min(_url_cache, key=lambda k: _url_cache[k][1])
        _url_cache.pop(oldest_key, None)


# Safety: restrict file reading to certain directories
import tempfile as _tempfile
_ALLOWED_ROOTS = [
    Path.home(),
    Path(_tempfile.gettempdir()),
]


def _is_path_allowed(path: Path) -> bool:
    """Check if a file path is within allowed directories."""
    resolved = path.resolve()
    return any(resolved.is_relative_to(root) for root in _ALLOWED_ROOTS)


async def _web_fetch(params: dict[str, Any], **deps: Any) -> str:
    """Fetch content from a URL using mcporter or curl fallback.

    For JSON API responses (like GitHub API), extracts key fields instead
    of dumping raw JSON to avoid truncated/unparseable output.
    """
    import json
    import shutil

    url = params.get("url", "")
    if not url:
        return "Error: url is required"

    # Check cache first
    cached = _get_cached(url)
    if cached is not None:
        return cached

    raw_content = None

    # Try mcporter first
    if shutil.which("mcporter"):
        try:
            proc = await asyncio.create_subprocess_exec(
                "mcporter", "call", "WebFetch.fetch_url",
                f"url={url}", "--output", "text",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0:
                raw_content = stdout.decode(errors="replace")
        except Exception:
            pass  # Fall through to curl

    # Fallback: curl
    if raw_content is None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "curl", "-sL", "--max-time", "20", "-A",
                "Mozilla/5.0 (compatible; MulAgent/1.0)", url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=25)
            if proc.returncode != 0:
                return f"Fetch failed: {stderr.decode(errors='replace')[:500]}"
            raw_content = stdout.decode(errors="replace")
        except asyncio.TimeoutError:
            return "Fetch timed out (25s)."
        except Exception as e:
            return f"Fetch error: {e}"

    if not raw_content:
        return "Empty response."

    # Extract readable content from HTML before compression
    raw_content = _extract_readable(raw_content)

    # Smart compression: if it's JSON, extract key fields
    content = _smart_compress(raw_content, url)
    _set_cache(url, content)
    return content


def _extract_readable(raw: str) -> str:
    """Extract readable text content from HTML.

    Tries trafilatura first (if installed), then falls back to regex-based
    tag stripping. Only processes content that looks like HTML.
    """
    import re as _re

    # Quick check: is this HTML?
    if not ("<html" in raw[:500].lower() or "<body" in raw[:500].lower()
            or "<!doctype" in raw[:500].lower()):
        return raw  # Not HTML, return as-is

    # Try trafilatura (best quality)
    try:
        import trafilatura
        extracted = trafilatura.extract(raw, include_links=True, include_tables=True)
        if extracted and len(extracted) > 100:
            return extracted
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: regex-based extraction
    # Remove script and style tags with content
    text = _re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=_re.DOTALL | _re.IGNORECASE)
    text = _re.sub(r"<style[^>]*>.*?</style>", "", text, flags=_re.DOTALL | _re.IGNORECASE)
    text = _re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=_re.DOTALL | _re.IGNORECASE)
    text = _re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=_re.DOTALL | _re.IGNORECASE)
    text = _re.sub(r"<header[^>]*>.*?</header>", "", text, flags=_re.DOTALL | _re.IGNORECASE)

    # Replace block elements with newlines
    text = _re.sub(r"<(?:p|div|br|li|h[1-6]|tr)[^>]*>", "\n", text, flags=_re.IGNORECASE)
    # Remove remaining tags
    text = _re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&nbsp;", " ").replace("&#39;", "'")
    # Clean up whitespace
    text = _re.sub(r"\n\s*\n+", "\n\n", text)
    text = _re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    return text if len(text) > 50 else raw


def _smart_compress(raw: str, url: str, max_chars: int = 5000) -> str:
    """Intelligently compress fetched content.

    Strategies (in priority order):
    1. GitHub Search API: extract repo summaries
    2. Paginated API ({data/results/items: [...], total/count/...})
    3. Generic JSON list: compact first N items
    4. Generic JSON object: compact dump
    5. HTML/text: plain truncation
    """
    import json

    if len(raw) <= max_chars:
        return raw

    # Try to parse as JSON
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Not JSON — truncate at line boundary for readability
        lines = raw.split("\n")
        kept, total = [], 0
        for line in lines:
            if total + len(line) > max_chars:
                break
            kept.append(line)
            total += len(line) + 1
        if len(kept) < len(lines):
            kept.append(f"... (truncated, {len(raw)} chars total)")
        return "\n".join(kept)

    # ── Strategy 1: GitHub Search API ──
    if isinstance(data, dict) and "items" in data and "total_count" in data:
        return _compress_github_search(data)

    # ── Strategy 2: Paginated API with common wrapper keys ──
    if isinstance(data, dict):
        # Detect common paginated formats: {data:[...]}, {results:[...]}, etc.
        list_key = None
        for key in ("data", "results", "items", "records", "list", "entries", "hits"):
            if key in data and isinstance(data[key], list):
                list_key = key
                break

        if list_key:
            items = data[list_key]
            meta_parts = []
            # Capture pagination metadata
            for mk in ("total", "total_count", "count", "page", "pages",
                        "page_size", "per_page", "next", "has_more"):
                if mk in data:
                    meta_parts.append(f"{mk}: {data[mk]}")
            meta = ", ".join(meta_parts) if meta_parts else f"{len(items)} items"
            return _compress_json_list(items, meta, max_chars)

    # ── Strategy 3: Generic JSON list ──
    if isinstance(data, list):
        return _compress_json_list(data, f"{len(data)} items total", max_chars)

    # ── Strategy 4: Generic JSON object ──
    compact = json.dumps(data, ensure_ascii=False, indent=1)
    if len(compact) > max_chars:
        compact = compact[:max_chars] + "\n... (truncated)"
    return compact


def _compress_github_search(data: dict) -> str:
    """Extract structured summaries from GitHub Search API response."""
    items = data["items"]
    total = data["total_count"]
    parts = [f"Total results: {total}\n"]
    for item in items[:15]:
        name = item.get("full_name", item.get("name", ""))
        stars = item.get("stargazers_count", 0)
        desc = item.get("description", "") or ""
        lang = item.get("language", "") or ""
        html_url = item.get("html_url", "")
        created = item.get("created_at", "")[:10]
        parts.append(
            f"- **{name}** | Stars: {stars} | Language: {lang} | Created: {created}\n"
            f"  {desc[:200]}\n"
            f"  {html_url}"
        )
    return "\n".join(parts)


def _compress_json_list(items: list, meta: str, max_chars: int) -> str:
    """Compress a JSON list by keeping first N items within budget."""
    import json

    parts = [f"[{meta}]\n"]
    total_len = len(parts[0])
    kept = 0

    for item in items:
        if isinstance(item, dict):
            # Try to extract a one-line summary from common fields
            summary = _summarize_dict(item)
        else:
            summary = json.dumps(item, ensure_ascii=False)
        if total_len + len(summary) + 2 > max_chars:
            parts.append(f"... ({len(items) - kept} more items)")
            break
        parts.append(f"- {summary}")
        total_len += len(summary) + 3
        kept += 1

    return "\n".join(parts)


def _summarize_dict(d: dict, max_len: int = 300) -> str:
    """Extract a readable one-line summary from a dict.

    Prioritizes common "useful" keys like name, title, description, url, etc.
    """
    import json

    priority_keys = [
        "name", "full_name", "title", "label", "id",
        "description", "summary", "content", "text",
        "url", "html_url", "link",
        "status", "state", "type", "category",
        "created_at", "updated_at", "date",
        "score", "stars", "stargazers_count", "count",
    ]

    parts = []
    used_keys = set()
    for key in priority_keys:
        if key in d:
            val = d[key]
            if val is None or val == "":
                continue
            s = str(val)
            if len(s) > 100:
                s = s[:100] + "..."
            parts.append(f"{key}: {s}")
            used_keys.add(key)
            if sum(len(p) for p in parts) > max_len:
                break

    # If we captured nothing useful, fall back to compact JSON
    if not parts:
        compact = json.dumps(d, ensure_ascii=False)
        return compact[:max_len] + ("..." if len(compact) > max_len else "")

    return " | ".join(parts)


async def _read_file(params: dict[str, Any], **deps: Any) -> str:
    """Read a local file's content."""
    file_path = params.get("path", "")
    if not file_path:
        return "Error: path is required"

    path = Path(file_path).expanduser()

    if not _is_path_allowed(path):
        return f"Error: access denied. Path must be under {[str(r) for r in _ALLOWED_ROOTS]}"

    if not path.exists():
        return f"Error: file not found: {path}"

    if not path.is_file():
        return f"Error: not a file: {path}"

    # Size check
    size = path.stat().st_size
    if size > 500_000:  # 500KB
        return f"Error: file too large ({size} bytes). Max 500KB."

    try:
        content = path.read_text(errors="replace")
        offset = params.get("offset", 0)
        limit = params.get("limit", 200)

        lines = content.split("\n")
        total_lines = len(lines)
        selected = lines[offset:offset + limit]

        # Header: show file stats so LLM can plan reads for large files
        header = f"[{path.name}: {total_lines} lines, {size} bytes"
        if total_lines > limit:
            header += f", showing lines {offset + 1}-{min(offset + limit, total_lines)}"
        header += "]\n"

        result = header + "\n".join(f"{i + offset + 1:4d}| {line}" for i, line in enumerate(selected))
        if offset + limit < total_lines:
            result += f"\n... ({total_lines - offset - limit} more lines, use offset={offset + limit} to continue)"
        return result
    except Exception as e:
        return f"Read error: {e}"


async def _list_dir(params: dict[str, Any], **deps: Any) -> str:
    """List directory contents with file info."""
    dir_path = params.get("path", "")
    if not dir_path:
        return "Error: path is required"

    path = Path(dir_path).expanduser()

    if not _is_path_allowed(path):
        return f"Error: access denied for {path}"
    if not path.exists():
        return f"Error: directory not found: {path}"
    if not path.is_dir():
        return f"Error: not a directory: {path}"

    recursive = params.get("recursive", False)
    max_entries = params.get("max_entries", 200)

    entries = []

    if recursive:
        import os
        count = 0
        for root, dirs, files in os.walk(path):
            # Skip hidden and common non-essential dirs
            dirs[:] = sorted([
                d for d in dirs
                if not d.startswith(".") and d not in (
                    "node_modules", "__pycache__", "vendor", "dist",
                    "build", "venv", "env", ".venv",
                )
            ])
            rel_root = Path(root).relative_to(path)
            for name in sorted(dirs):
                entries.append(f"  {rel_root / name}/")
                count += 1
                if count >= max_entries:
                    break
            for name in sorted(files):
                fpath = Path(root) / name
                try:
                    size = fpath.stat().st_size
                    size_str = _format_size(size)
                    entries.append(f"  {rel_root / name}  ({size_str})")
                except OSError:
                    entries.append(f"  {rel_root / name}  (?)")
                count += 1
                if count >= max_entries:
                    break
            if count >= max_entries:
                entries.append(f"  ... (truncated at {max_entries} entries)")
                break
    else:
        items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        for item in items[:max_entries]:
            if item.is_dir():
                entries.append(f"  {item.name}/")
            else:
                try:
                    size_str = _format_size(item.stat().st_size)
                    entries.append(f"  {item.name}  ({size_str})")
                except OSError:
                    entries.append(f"  {item.name}  (?)")
        if len(list(path.iterdir())) > max_entries:
            entries.append(f"  ... (truncated at {max_entries} entries)")

    header = f"{path}/  ({len(entries)} entries)"
    return header + "\n" + "\n".join(entries)


def _format_size(size: int) -> str:
    """Format file size in human-readable form."""
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    else:
        return f"{size / (1024 * 1024):.1f}MB"



# ── Glob: file pattern search ──────────────────────────────────

async def _glob_search(params: dict[str, Any], **deps: Any) -> str:
    """Find files matching a glob pattern."""
    import fnmatch
    import os

    pattern = params.get("pattern", "")
    if not pattern:
        return "Error: pattern is required (e.g. '**/*.py', 'src/**/*.ts')"

    path = params.get("path", "")
    if not path:
        path = str(Path.cwd())
    base = Path(path).expanduser().resolve()

    if not _is_path_allowed(base):
        return f"Error: access denied for {base}"
    if not base.is_dir():
        return f"Error: not a directory: {base}"

    max_results = params.get("max_results", 200)

    # Use Path.glob for recursive patterns
    try:
        matches = []
        for p in sorted(base.glob(pattern)):
            # Skip hidden dirs and common junk
            parts = p.relative_to(base).parts
            if any(
                part.startswith(".") or part in (
                    "node_modules", "__pycache__", "vendor", "dist",
                    "build", "venv", "env", ".venv",
                )
                for part in parts
            ):
                continue
            rel = str(p.relative_to(base))
            if p.is_dir():
                rel += "/"
            else:
                try:
                    size_str = _format_size(p.stat().st_size)
                    rel += f"  ({size_str})"
                except OSError:
                    pass
            matches.append(rel)
            if len(matches) >= max_results:
                break

        if not matches:
            return f"No files matching '{pattern}' in {base}"

        header = f"[{len(matches)} matches for '{pattern}' in {base}]\n"
        result = header + "\n".join(matches)
        if len(matches) >= max_results:
            result += f"\n... (limited to {max_results}, use a more specific pattern)"
        return result
    except Exception as e:
        return f"Glob error: {e}"


# ── Grep: content search ───────────────────────────────────────

async def _grep_search(params: dict[str, Any], **deps: Any) -> str:
    """Search file contents by regex pattern."""
    import re

    pattern = params.get("pattern", "")
    if not pattern:
        return "Error: pattern is required"

    path = params.get("path", "")
    if not path:
        path = str(Path.cwd())
    base = Path(path).expanduser().resolve()

    if not _is_path_allowed(base):
        return f"Error: access denied for {base}"

    file_glob = params.get("file_glob", "")  # e.g. "*.py", "*.ts"
    max_results = params.get("max_results", 50)
    context_lines = params.get("context", 0)
    case_insensitive = params.get("case_insensitive", False)

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Invalid regex: {e}"

    # Collect target files
    skip_dirs = {
        "node_modules", "__pycache__", "vendor", "dist", "build",
        "venv", "env", ".venv", ".git", ".tox",
    }
    binary_exts = {
        ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll", ".exe",
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip",
        ".gz", ".tar", ".whl", ".egg", ".bin", ".dat", ".db", ".sqlite",
    }

    files_to_search = []
    if base.is_file():
        files_to_search = [base]
    else:
        glob_pat = file_glob if file_glob else "**/*"
        for p in base.glob(glob_pat):
            if not p.is_file():
                continue
            if p.suffix.lower() in binary_exts:
                continue
            parts = p.relative_to(base).parts
            if any(part.startswith(".") or part in skip_dirs for part in parts):
                continue
            files_to_search.append(p)
            if len(files_to_search) > 5000:  # safety cap
                break

    matches = []
    files_with_matches = set()

    for fpath in files_to_search:
        if len(matches) >= max_results:
            break
        try:
            text = fpath.read_text(errors="replace")
            lines = text.split("\n")
            for line_num, line in enumerate(lines, 1):
                if regex.search(line):
                    rel = str(fpath.relative_to(base)) if base.is_dir() else fpath.name
                    files_with_matches.add(rel)

                    # Build match entry with optional context
                    entry_lines = []
                    if context_lines > 0:
                        start = max(0, line_num - 1 - context_lines)
                        end = min(len(lines), line_num + context_lines)
                        for ctx_num in range(start, end):
                            prefix = ">" if ctx_num == line_num - 1 else " "
                            entry_lines.append(f"{prefix} {ctx_num + 1:4d}| {lines[ctx_num]}")
                        matches.append(f"{rel}:\n" + "\n".join(entry_lines))
                    else:
                        matches.append(f"{rel}:{line_num}: {line.rstrip()}")

                    if len(matches) >= max_results:
                        break
        except (UnicodeDecodeError, PermissionError, OSError):
            continue

    if not matches:
        scope = f"in {base}" + (f" ({file_glob})" if file_glob else "")
        return f"No matches for /{pattern}/ {scope}"

    header = f"[{len(matches)} matches in {len(files_with_matches)} files for /{pattern}/]\n"
    result = header + "\n".join(matches)
    if len(matches) >= max_results:
        result += f"\n... (limited to {max_results}, refine pattern for more)"
    return result


WEB_FETCH = ToolDef(
    name="web_fetch",
    description="Fetch the content of a URL (webpage, API endpoint, etc.).",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
        },
        "required": ["url"],
    },
    fn=_web_fetch,
    category="search",
)

READ_FILE = ToolDef(
    name="read_file",
    description="Read content from a local file. Output includes total line count. For large files (>200 lines), use offset/limit to read in sections rather than reading the whole file multiple times.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute file path",
            },
            "offset": {
                "type": "integer",
                "description": "Starting line number (0-based, default 0)",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to read (default 200)",
            },
        },
        "required": ["path"],
    },
    fn=_read_file,
    category="search",
)

LIST_DIR = ToolDef(
    name="list_dir",
    description=(
        "List directory contents with file sizes. Use to understand project structure "
        "before reading specific files. Supports recursive listing."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute directory path",
            },
            "recursive": {
                "type": "boolean",
                "description": "List recursively (default: false)",
            },
            "max_entries": {
                "type": "integer",
                "description": "Max entries to return (default: 200)",
            },
        },
        "required": ["path"],
    },
    fn=_list_dir,
    category="search",
)

GLOB_SEARCH = ToolDef(
    name="glob_search",
    description=(
        "Find files matching a glob pattern. Fast file discovery by name/extension. "
        "Use this instead of execute_shell with find/ls. "
        "Examples: '**/*.py', 'src/**/*.ts', '**/test_*.py', '*.yaml'"
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g. '**/*.py', 'src/**/*.ts')",
            },
            "path": {
                "type": "string",
                "description": "Base directory to search in (default: current working directory)",
            },
            "max_results": {
                "type": "integer",
                "description": "Max files to return (default: 200)",
            },
        },
        "required": ["pattern"],
    },
    fn=_glob_search,
    category="search",
)

GREP_SEARCH = ToolDef(
    name="grep_search",
    description=(
        "Search file contents by regex pattern. Use this instead of execute_shell with grep/rg. "
        "Returns matching lines with file paths and line numbers. "
        "Supports file type filtering, context lines, and case-insensitive search."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for (e.g. 'def main', 'import.*os', 'TODO')",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search in (default: current working directory)",
            },
            "file_glob": {
                "type": "string",
                "description": "Only search files matching this glob (e.g. '*.py', '**/*.ts')",
            },
            "max_results": {
                "type": "integer",
                "description": "Max matches to return (default: 50)",
            },
            "context": {
                "type": "integer",
                "description": "Number of context lines before and after each match (default: 0)",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Case-insensitive search (default: false)",
            },
        },
        "required": ["pattern"],
    },
    fn=_grep_search,
    category="search",
)
