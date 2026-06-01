#!/usr/bin/env python3
"""
scripts/publish_medium.py — Render Hugo posts to HTML for Medium copy-paste.

Outputs a fully rendered .html file per post to _medium/ that you can open
in a browser, select-all, and paste directly into Medium's editor.

Handles:
  - Relative image paths → absolute GitHub raw URLs
  - Hugo shortcodes stripped
  - LaTeX → CodeCogs PNG images (Medium has no KaTeX)
  - Blockquote lazy-continuation lines (lines without > inside indented blockquotes)
  - Code blocks with syntax highlighting via <pre><code>

Usage:
    python scripts/publish_medium.py                         # all posts
    python scripts/publish_medium.py --file content/posts/foo/index.md
"""
import argparse
import re
import sys
from pathlib import Path
from urllib.parse import quote

import markdown
import yaml

INPUT_DIR = Path("content/posts")
OUTPUT_DIR = Path("_medium")
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/PsyDak-Meng/DakDevDiary/main"
CODECOGS_BASE = "https://latex.codecogs.com/png.latex?%5Cbg_white%20"

_BLOCK_RE = re.compile(
    r"^(\s{0,3}#{1,6}\s"
    r"|\s*>"
    r"|\s*[-*+]\s"
    r"|\s*\d+\.\s"
    r"|```|~~~"
    r"|</?[a-zA-Z]"
    r"|---+$|===+$"
    r"|\|"
    r")"
)

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Georgia, serif; max-width: 740px; margin: 40px auto; line-height: 1.7; color: #333; }}
  blockquote {{ border-left: 3px solid #ccc; margin: 1.5em 0; padding: 0.5em 1em; color: #555; }}
  pre {{ background: #f4f4f4; padding: 1em; overflow-x: auto; border-radius: 4px; }}
  code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 0.9em; }}
  pre code {{ background: none; padding: 0; }}
  img {{ max-width: 100%; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


# ── Frontmatter ────────────────────────────────────────────────────────────────

def extract_frontmatter(content: str) -> tuple[dict, str]:
    if not content.startswith("---"):
        return {}, content
    end = content.index("---", 3)
    fm = yaml.safe_load(content[3:end]) or {}
    return fm, content[end + 3:].lstrip("\n")


# ── Preprocessing ──────────────────────────────────────────────────────────────

def rewrite_images(body: str, src: Path) -> str:
    src_dir = src.parent

    def replace(m: re.Match) -> str:
        alt, path = m.group(1), m.group(2)
        if path.startswith(("http", "data:")):
            return m.group(0)
        abs_path = (src_dir / path).resolve().relative_to(Path(".").resolve())
        url = f"{GITHUB_RAW_BASE}/{quote(str(abs_path), safe='/')}"
        return f"![{alt}]({url})"

    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace, body)


def latex_to_img(latex: str, display: bool = False) -> str:
    encoded = quote(latex.strip())
    url = f"{CODECOGS_BASE}{encoded}"
    if display:
        return f'\n<p align="center"><img alt="equation" src="{url}"></p>\n'
    # Medium doesn't support inline images — render as inline code with LaTeX source
    return f"<code>{latex.strip()}</code>"


def normalize_blockquotes(body: str) -> str:
    """Add > prefix to lazy-continuation lines and ensure blank lines around blockquotes.

    Hugo/CommonMark allows indented blockquotes inside list items where subsequent
    lines omit the > marker. The markdown library needs explicit > on every line.
    """
    lines = body.split('\n')
    in_code = False
    result: list[str] = []
    bq_buf: list[str] = []

    def flush_bq() -> None:
        if not bq_buf:
            return
        if result and result[-1] != '':
            result.append('')
        for ln in bq_buf:
            result.append(ln.lstrip())  # dedent so markdown parser sees top-level >
        result.append('')
        bq_buf.clear()

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('```') or stripped.startswith('~~~'):
            flush_bq()
            in_code = not in_code
            result.append(line)
            continue

        if in_code:
            result.append(line)
            continue

        is_bq = bool(re.match(r'\s*>', line))
        is_block = bool(_BLOCK_RE.match(line)) and not is_bq

        if is_bq:
            bq_buf.append(line)
        elif stripped and not is_block and bq_buf:
            bq_prefix = re.match(r'(\s*>)', bq_buf[0]).group(1).lstrip()
            bq_buf.append(bq_prefix + ' ' + stripped)
        else:
            flush_bq()
            result.append(line)

    flush_bq()
    return '\n'.join(result)


def preprocess_body(body: str, src: Path) -> str:
    body = rewrite_images(body, src)
    body = re.sub(r'\{\{<\s*icon\s+"link"\s*>\}\}', "🔗", body)
    body = re.sub(r'\{\{<[^>]+>\}\}', "", body)
    body = re.sub(
        r'\$\$(.*?)\$\$',
        lambda m: latex_to_img(m.group(1), display=True),
        body, flags=re.DOTALL,
    )
    body = re.sub(
        r'(?<!\$)\$([^\$\n]+?)\$(?!\$)',
        lambda m: latex_to_img(m.group(1)),
        body,
    )
    body = re.sub(r'<div[^>]*>\n?', '', body)
    body = re.sub(r'\n?</div>', '', body)
    body = normalize_blockquotes(body)
    return body


# ── Per-file processing ────────────────────────────────────────────────────────

def process_file(src: Path) -> None:
    content = src.read_text(encoding="utf-8")
    fm, body = extract_frontmatter(content)

    if (
        src.name == "_index.md"
        or not fm.get("title")
        or fm.get("draft") is True
        or fm.get("devto_sync") is False
    ):
        return

    rel = src.relative_to(INPUT_DIR)
    out_path = (OUTPUT_DIR / rel).with_suffix(".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    body = preprocess_body(body, src)
    html_body = markdown.markdown(
        body,
        extensions=["fenced_code", "tables", "nl2br", "sane_lists"],
    )
    title = fm["title"]
    full_html = HTML_TEMPLATE.format(body=f"<h1>{title}</h1>\n{html_body}")
    out_path.write_text(full_html, encoding="utf-8")
    print(f"  rendered: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, metavar="PATH",
                        help="Process a single markdown file instead of all posts")
    args = parser.parse_args()

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
        process_file(f)


if __name__ == "__main__":
    main()
