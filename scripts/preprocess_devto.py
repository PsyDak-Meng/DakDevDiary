"""Preprocesses Hugo markdown posts for Dev.to publishing.

Reads content/posts/**/*.md, converts LaTeX → Codecogs PNG images,
strips Hugo shortcodes, fixes line breaks and HTML blocks for CommonMark,
and writes to _devto/ preserving existing id: fields.

Skips files where draft: true, devto_sync: false, or no title (e.g. TOML frontmatter).
"""
import re
import sys
from pathlib import Path
from urllib.parse import quote

import yaml

INPUT_DIR = Path("content/posts")
OUTPUT_DIR = Path("_devto")
# PNG is more reliable than SVG on Dev.to (CSP/proxy compatibility)
CODECOGS_BASE = "https://latex.codecogs.com/png.latex?%5Cbg_white%20"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/PsyDak-Meng/DakDevDiary/main"

HUGO_ONLY_FIELDS = {"math", "draft", "date"}

# Lines starting with these are Markdown block elements — don't add hard breaks after them.
# Note: intentionally excludes \s{4,} (indented code) so list-continuation lines
# (4-space indent in list items) still get hard breaks. Use fenced code blocks instead.
_BLOCK_RE = re.compile(
    r"^(\s{0,3}#{1,6}\s"   # ATX headings
    r"|\s*>"                 # blockquotes (any indent, e.g. inside list items)
    r"|\s*[-*+]\s"          # unordered list items
    r"|\s*\d+\.\s"          # ordered list items
    r"|```|~~~"             # fenced code
    r"|</?[a-zA-Z]"         # HTML tags
    r"|---+$|===+$"         # thematic breaks / setext underlines
    r"|\|"                  # tables
    r")"
)


def extract_frontmatter(content: str) -> tuple[dict, str]:
    # TOML frontmatter (+++) is Hugo-only — return empty dict so caller skips the file
    if not content.startswith("---"):
        return {}, content
    end = content.index("---", 3)
    fm = yaml.safe_load(content[3:end]) or {}
    return fm, content[end + 3:].lstrip("\n")


def render_frontmatter(fm: dict) -> str:
    return "---\n" + yaml.dump(fm, allow_unicode=True, sort_keys=False) + "---\n\n"


def latex_to_img(latex: str, display: bool = False) -> str:
    encoded = quote(latex.strip())
    alt = "equation" if display else latex.strip()[:40]
    return f"![{alt}]({CODECOGS_BASE}{encoded})"


def rewrite_images(body: str, src: Path) -> str:
    """Rewrite relative image paths to absolute GitHub raw URLs."""
    src_dir = src.parent

    def replace(m: re.Match) -> str:
        alt, path = m.group(1), m.group(2)
        if path.startswith("http"):
            return m.group(0)
        abs_path = (src_dir / path).resolve().relative_to(Path(".").resolve())
        url = f"{GITHUB_RAW_BASE}/{quote(str(abs_path), safe='/')}"
        return f"![{alt}]({url})"

    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace, body)


def strip_html_wrappers(body: str) -> str:
    """Remove <div> wrappers that block CommonMark from rendering markdown inside them.

    In CommonMark, a block-level <div> starts an HTML block where markdown is not parsed.
    Stripping the tags lets image syntax (from LaTeX conversion) render normally.
    """
    body = re.sub(r'<div[^>]*>\n?', '', body)
    body = re.sub(r'\n?</div>', '', body)
    return body


def add_hard_breaks(body: str) -> str:
    """Join consecutive inline text lines with <br> so each renders on its own line.

    Dev.to uses CommonMark where a single newline within a paragraph does not produce
    a <br>. Consecutive non-block lines are collapsed into one line joined by <br>
    so Dev.to renders them with visible hard line breaks.

    Blockquote lines and their lazy-continuation inline lines are buffered together,
    dedented to the block level (dev.to ignores indented blockquotes inside list items),
    and emitted with <br> hard breaks between items so Dev.to renders each line visibly.
    """
    lines = body.split('\n')
    in_code = False
    result: list[str] = []
    run: list[str] = []
    bq_buf: list[str] = []  # blockquote + lazy-continuation lines, dedented on flush

    def flush_run() -> None:
        if run:
            result.append('<br>'.join(l.rstrip() for l in run))
            run.clear()

    def flush_bq() -> None:
        """Emit buffered blockquote group dedented to block level with <br> hard breaks."""
        if not bq_buf:
            return
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
        is_block = bool(_BLOCK_RE.match(line)) and not is_bq

        if stripped and not is_bq and not is_block:
            if bq_buf:
                # Lazy continuation: re-prefix with the blockquote marker
                bq_prefix = re.match(r'(\s*>)', bq_buf[0]).group(1)
                bq_buf.append(bq_prefix + ' ' + stripped)
            else:
                run.append(line)
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
    # Hugo link icon shortcode → emoji
    body = re.sub(r'\{\{<\s*icon\s+"link"\s*>\}\}', "🔗", body)
    # Strip all remaining Hugo shortcodes
    body = re.sub(r'\{\{<[^>]+>\}\}', "", body)
    # Display math $$...$$ — must run before inline to avoid double-match
    body = re.sub(
        r'\$\$(.*?)\$\$',
        lambda m: latex_to_img(m.group(1), display=True),
        body,
        flags=re.DOTALL,
    )
    # Inline math $...$
    body = re.sub(
        r'(?<!\$)\$([^\$\n]+?)\$(?!\$)',
        lambda m: latex_to_img(m.group(1)),
        body,
    )
    # Remove <div> wrappers so converted math images render in CommonMark
    body = strip_html_wrappers(body)
    # Fix hard line breaks for consecutive inline lines
    body = add_hard_breaks(body)
    return body


def get_existing_id(out_path: Path) -> int | None:
    """Preserve the Dev.to article id so updates don't create duplicates."""
    if not out_path.exists():
        return None
    try:
        fm, _ = extract_frontmatter(out_path.read_text(encoding="utf-8"))
        return fm.get("id")
    except Exception:
        return None


def process_file(src: Path) -> None:
    content = src.read_text(encoding="utf-8")
    fm, body = extract_frontmatter(content)

    # Skip section indexes, TOML files (no title after failed parse), drafts, opted-out
    if (
        src.name == "_index.md"
        or not fm.get("title")
        or fm.get("draft") is True
        or fm.get("devto_sync") is False
    ):
        return

    rel = src.relative_to(INPUT_DIR)
    out_path = OUTPUT_DIR / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_id = get_existing_id(out_path)
    if existing_id is not None:
        fm["id"] = existing_id

    # Strip Hugo-specific fields Dev.to doesn't understand
    for field in HUGO_ONLY_FIELDS:
        fm.pop(field, None)

    # Sanitize tags: Dev.to requires lowercase alphanumeric only, max 4
    if "tags" in fm:
        fm["tags"] = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in fm["tags"][:4]]

    # Inject published: true so Hugo never sees it in source files
    fm["published"] = True

    out_path.write_text(render_frontmatter(fm) + preprocess_body(body, src), encoding="utf-8")
    print(f"  processed: {rel}")


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Input dir not found: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    files = list(INPUT_DIR.rglob("*.md"))
    print(f"Processing {len(files)} files → {OUTPUT_DIR}/")
    for f in files:
        process_file(f)


if __name__ == "__main__":
    main()
