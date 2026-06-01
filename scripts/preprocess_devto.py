"""Preprocesses Hugo markdown posts for Dev.to publishing.

Reads content/posts/**/*.md, converts LaTeX → Codecogs SVG images,
strips Hugo shortcodes, and writes to _devto/ preserving existing id: fields.

Skips files where draft: true or devto_sync: false.
"""
import re
import sys
from pathlib import Path
from urllib.parse import quote

import yaml

INPUT_DIR = Path("content/posts")
OUTPUT_DIR = Path("_devto")
CODECOGS_BASE = "https://latex.codecogs.com/svg.image?"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/PsyDak-Meng/DakDevDiary/main"


def extract_frontmatter(content: str) -> tuple[dict, str]:
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

    # Skip Hugo drafts and files explicitly opted out
    if fm.get("draft") is True or fm.get("devto_sync") is False:
        return

    rel = src.relative_to(INPUT_DIR)
    out_path = OUTPUT_DIR / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_id = get_existing_id(out_path)
    if existing_id is not None:
        fm["id"] = existing_id

    # Sanitize tags: Dev.to requires lowercase, no spaces, max 4
    if "tags" in fm:
        fm["tags"] = [re.sub(r"\s+", "", t).lower() for t in fm["tags"][:4]]

    # Inject published: true here so Hugo never sees it in source files
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
