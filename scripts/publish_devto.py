#!/usr/bin/env python3
"""
scripts/publish_devto.py — Local Dev.to publisher, no GitHub Actions required.

Preprocessing per post:
  - Rewrites relative image paths → GitHub raw URLs
  - Strips Hugo shortcodes ({{< icon "link" >}} → 🔗, rest stripped)
  - Renders LaTeX via matplotlib (white-background PNG data URI)
    Falls back to CodeCogs if matplotlib or TeX math fails
  - Fixes CommonMark hard line breaks (consecutive inline lines joined with <br>)
  - Strips <div> wrappers that block CommonMark markdown inside HTML blocks

Publishing:
  - Reads existing article id from _devto/<path>.md frontmatter
  - Creates (POST) or updates (PUT) via Dev.to API
  - Writes returned id back to _devto/<path>.md for future runs

Requirements:
    pip install requests pyyaml matplotlib

Usage:
    export DEVTO_TOKEN=<your_api_key>        # or pass --token
    python scripts/publish_devto.py          # publish all eligible posts
    python scripts/publish_devto.py --dry-run       # preprocess only, no API calls
    python scripts/publish_devto.py --force-new     # always create, ignore stored ids
    python scripts/publish_devto.py --file content/posts/foo/index.md  # single post
"""
import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

import requests
import yaml

# Load .env from repo root if present (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_DIR = Path("content/posts")
OUTPUT_DIR = Path("_devto")
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/PsyDak-Meng/DakDevDiary/main"
DEVTO_API = "https://dev.to/api/articles"

HUGO_ONLY_FIELDS = {"math", "draft", "date"}

# Block-level markdown elements — lines matching these are NOT joined with <br>
_BLOCK_RE = re.compile(
    r"^(\s{0,3}#{1,6}\s"   # ATX headings
    r"|\s*>"                # blockquotes (any indent)
    r"|\s*[-*+]\s"          # unordered list items
    r"|\s*\d+\.\s"          # ordered list items
    r"|```|~~~"             # fenced code fences
    r"|</?[a-zA-Z]"         # HTML tags
    r"|---+$|===+$"         # thematic breaks / setext underlines
    r"|\|"                  # tables
    r"|\{%"                 # Liquid tags ({% katex %}, {% endkatex %}, etc.)
    r")"
)

# ── Frontmatter ────────────────────────────────────────────────────────────────

def extract_frontmatter(content: str) -> tuple[dict, str]:
    # TOML frontmatter (+++) is Hugo-only — return empty dict so caller skips
    if not content.startswith("---"):
        return {}, content
    end = content.index("---", 3)
    fm = yaml.safe_load(content[3:end]) or {}
    return fm, content[end + 3:].lstrip("\n")


def render_frontmatter(fm: dict) -> str:
    return "---\n" + yaml.dump(fm, allow_unicode=True, sort_keys=False) + "---\n\n"


# ── Math rendering ─────────────────────────────────────────────────────────────

def latex_to_katex(latex: str, display: bool = False) -> str:
    """Convert LaTeX to Dev.to KaTeX liquid tag (renders identically to markdown preview)."""
    inner = latex.strip()
    if display:
        return f"{{% katex %}}\n{inner}\n{{% endkatex %}}"
    return f"{{% katex inline %}}{inner}{{% endkatex %}}"


# ── Body transforms ────────────────────────────────────────────────────────────

def rewrite_images(body: str, src: Path) -> str:
    """Rewrite relative image paths to absolute GitHub raw URLs."""
    src_dir = src.parent

    def replace(m: re.Match) -> str:
        alt, path = m.group(1), m.group(2)
        if path.startswith(("http", "data:")):
            return m.group(0)
        abs_path = (src_dir / path).resolve().relative_to(Path(".").resolve())
        url = f"{GITHUB_RAW_BASE}/{quote(str(abs_path), safe='/')}"
        return f"![{alt}]({url})"

    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace, body)


def strip_html_wrappers(body: str) -> str:
    """Remove <div> wrappers that prevent CommonMark from rendering markdown inside."""
    body = re.sub(r'<div[^>]*>\n?', '', body)
    body = re.sub(r'\n?</div>', '', body)
    return body


_IMAGE_RE = re.compile(r'^\s*!\[')


def add_hard_breaks(body: str) -> str:
    """Join consecutive inline lines with <br> for CommonMark hard line breaks.

    Rules:
    - Image lines get their own paragraph (blank lines around them) so they render.
    - Non-block inline lines are joined with <br>, preserving the indent of the
      first line in the run so list-item continuations stay inside the list.
    - Blockquote lines + their lazy-continuation inline lines are buffered together,
      dedented to the block level (dev.to doesn't parse indented blockquotes inside
      list items), and emitted with backslash hard breaks between items.
    """
    lines = body.split('\n')
    in_code = False
    result: list[str] = []
    run: list[str] = []
    run_prefix: str = ''
    bq_buf: list[str] = []

    def flush_run() -> None:
        nonlocal run_prefix
        if run:
            result.append(run_prefix + '<br>'.join(run))
            run.clear()
            run_prefix = ''

    def flush_bq() -> None:
        if not bq_buf:
            return
        if result and result[-1] != '':
            result.append('')
        for i, ln in enumerate(bq_buf):
            suffix = '<br>' if i < len(bq_buf) - 1 else ''
            result.append(ln.lstrip().rstrip() + suffix)
        bq_buf.clear()

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('```') or stripped.startswith('~~~'):
            flush_run()
            flush_bq()
            in_code = not in_code
            result.append(line)
            continue

        if in_code:
            result.append(line)
            continue

        is_bq = bool(re.match(r'\s*>', line))

        # Images need special handling:
        # - Inside an active blockquote run: prefix with > so the image stays in the quote.
        # - Indented (in a list item): convert to <img> tag and join the <br> run.
        # - Top-level: own paragraph with blank lines so it renders as a block.
        if _IMAGE_RE.match(line) and stripped:
            indent = line[: len(line) - len(line.lstrip())]
            img_m = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', stripped)
            if bq_buf and img_m:
                img_tag = f'<img src="{img_m.group(2)}" alt="{img_m.group(1)}">'
                bq_prefix = re.match(r'(\s*>)', bq_buf[0]).group(1)
                bq_buf.append(bq_prefix + ' ' + img_tag)
            elif indent and img_m:
                img_tag = f'<img src="{img_m.group(2)}" alt="{img_m.group(1)}">'
                if not run:
                    run_prefix = indent
                run.append(img_tag)
            else:
                flush_run()
                flush_bq()
                if result and result[-1] != '':
                    result.append('')
                result.append(stripped)
                result.append('')
            continue

        if stripped and not is_bq and not _BLOCK_RE.match(line):
            if bq_buf:
                # Lazy continuation of the active blockquote
                bq_prefix = re.match(r'(\s*>)', bq_buf[0]).group(1)
                bq_buf.append(bq_prefix + ' ' + stripped)
            else:
                if not run:
                    run_prefix = line[: len(line) - len(line.lstrip())]
                run.append(stripped)
        elif is_bq:
            flush_run()
            bq_buf.append(line)
        else:
            flush_run()
            flush_bq()
            result.append(line)

    flush_run()
    flush_bq()
    return '\n'.join(result)


def preprocess_body(body: str, src: Path) -> str:
    body = rewrite_images(body, src)
    body = re.sub(r'\{\{<\s*icon\s+"link"\s*>\}\}', "🔗", body)
    body = re.sub(r'\{\{<[^>]+>\}\}', "", body)
    # Display math $$...$$ and inline $...$ → Dev.to KaTeX liquid tags
    body = re.sub(
        r'\$\$(.*?)\$\$',
        lambda m: latex_to_katex(m.group(1), display=True),
        body, flags=re.DOTALL,
    )
    body = re.sub(
        r'(?<!\$)\$([^\$\n]+?)\$(?!\$)',
        lambda m: latex_to_katex(m.group(1)),
        body,
    )
    body = strip_html_wrappers(body)
    body = add_hard_breaks(body)
    return body


# ── ID persistence (via _devto/ files) ────────────────────────────────────────

def get_existing_id(out_path: Path) -> int | None:
    if not out_path.exists():
        return None
    try:
        fm, _ = extract_frontmatter(out_path.read_text(encoding="utf-8"))
        return fm.get("id")
    except Exception:
        return None


# ── Dev.to API ─────────────────────────────────────────────────────────────────

def devto_publish(
    title: str,
    body_markdown: str,
    tags: list[str],
    token: str,
    existing_id: int | None,
) -> int | None:
    headers = {"api-key": token, "content-type": "application/json"}
    payload = {
        "article": {
            "title": title,
            "body_markdown": body_markdown,
            "published": True,
            "tags": tags,
        }
    }
    if existing_id:
        resp = requests.put(f"{DEVTO_API}/{existing_id}", json=payload, headers=headers)
        action = "updated"
    else:
        resp = requests.post(DEVTO_API, json=payload, headers=headers)
        action = "created"

    if resp.status_code in (200, 201):
        article_id = resp.json().get("id")
        url = resp.json().get("url", "")
        print(f"    {action}: {url}")
        return article_id
    else:
        print(f"    ERROR {resp.status_code}: {resp.text[:400]}", file=sys.stderr)
        return None


# ── Per-file processing ────────────────────────────────────────────────────────

def process_file(
    src: Path,
    token: str | None,
    dry_run: bool,
    force_new: bool,
) -> None:
    content = src.read_text(encoding="utf-8")
    fm, body = extract_frontmatter(content)

    if (
        src.name == "_index.md"
        or not fm.get("title")
        or fm.get("draft") is True
        or fm.get("devto_sync") is False
    ):
        return

    title = fm["title"]
    tags = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in fm.get("tags", [])[:4]]

    rel = src.relative_to(INPUT_DIR)
    out_path = OUTPUT_DIR / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_id = None if force_new else get_existing_id(out_path)
    processed_body = preprocess_body(body, src)

    print(f"  {rel}")

    if dry_run or token is None:
        # Write preprocessed output for inspection
        out_fm: dict = {}
        if existing_id:
            out_fm["id"] = existing_id
        out_fm["published"] = True
        if tags:
            out_fm["tags"] = tags
        out_path.write_text(render_frontmatter(out_fm) + processed_body, encoding="utf-8")
        print("    (dry-run — skipping API call)")
        return

    article_id = devto_publish(title, processed_body, tags, token, existing_id)
    if article_id is None:
        return

    # Persist id for future update runs
    out_fm = {}
    if article_id:
        out_fm["id"] = article_id
    out_fm["published"] = True
    if tags:
        out_fm["tags"] = tags
    out_path.write_text(render_frontmatter(out_fm) + processed_body, encoding="utf-8")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", default=os.environ.get("DEVTO_TOKEN"),
                        help="Dev.to API token (or set DEVTO_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preprocess only; write _devto/ files but skip API calls")
    parser.add_argument("--force-new", action="store_true",
                        help="Ignore stored ids — always create new articles")
    parser.add_argument("--file", type=Path, metavar="PATH",
                        help="Process a single markdown file instead of all posts")
    args = parser.parse_args()

    if not args.dry_run and not args.token:
        print("Error: provide --token or set DEVTO_TOKEN env var", file=sys.stderr)
        sys.exit(1)

    if not INPUT_DIR.exists():
        print(f"Input dir not found: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.file:
        files = [args.file]
        print(f"Processing 1 file → {OUTPUT_DIR}/")
    else:
        files = sorted(INPUT_DIR.rglob("*.md"))
        print(f"Processing {len(files)} files → {OUTPUT_DIR}/")

    for f in files:
        process_file(f, args.token, args.dry_run, args.force_new)


if __name__ == "__main__":
    main()
