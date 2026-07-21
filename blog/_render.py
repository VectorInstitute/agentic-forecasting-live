#!/usr/bin/env python3
"""Post-process a pandoc/markdown HTML fragment into a publication-ready page.

Not meant to be run standalone by hand — invoked by render.sh, which passes
the fragment HTML on stdin and options as CLI flags, and prints the finished
document to stdout. Kept as a separate file because the image/caption
merge and base64 embedding are easier to get right in real Python than in
shell+sed.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
import sys
from pathlib import Path


# Matches a lone image paragraph immediately followed by a lone-em caption
# paragraph -- the "italic paragraph right after the image" convention used
# throughout post.md. Pandoc renders each markdown paragraph as its own
# top-level <p>, so this pattern is exact, not a heuristic over free text.
FIG_PATTERN = re.compile(
    r"<p><img\s+([^>]*)/?>\s*</p>\s*<p><em>(.*?)</em></p>",
    re.DOTALL,
)
SRC_RE = re.compile(r'src="([^"]*)"')
ALT_RE = re.compile(r'alt="([^"]*)"')


def _embed_data_uri(img_path: Path) -> str:
    data = img_path.read_bytes()
    mime = mimetypes.guess_type(img_path.name)[0] or "application/octet-stream"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_figure(match: re.Match, *, source_dir: Path, mode: str, out_dir: Path) -> str:
    attrs, caption_html = match.group(1), match.group(2)
    src_m = SRC_RE.search(attrs)
    alt_m = ALT_RE.search(attrs)
    if not src_m:
        return match.group(0)  # not an image we recognize; leave untouched
    src = src_m.group(1)
    alt = alt_m.group(1) if alt_m else ""

    if mode == "embed":
        img_path = (source_dir / src).resolve()
        new_src = _embed_data_uri(img_path)
    else:  # linked: rewrite relative to the output directory
        img_path = (source_dir / src).resolve()
        new_src = str(Path("..") / img_path.relative_to(source_dir.parent.resolve()))

    img_tag = f'<img src="{new_src}" alt="{alt}" loading="lazy" decoding="async">'
    return (
        '<figure class="fig">'
        '<span class="fig-frame">'
        f"{img_tag}"
        '<span class="fig-hint">&#10021; Click to enlarge</span>'
        "</span>"
        f"<figcaption><em>{caption_html}</em></figcaption>"
        "</figure>"
    )


def _check_orphan_images(body: str) -> None:
    """Fail on any <img> paragraph that did not merge into a <figure>.

    Every image in post.md must be immediately followed by a one-paragraph
    italic caption (the FIG_PATTERN contract). An image that survives the
    substitution as a bare ``<p><img …></p>`` has a missing, blank-line-split,
    or nested-italic-broken caption — and would silently render captionless.
    """
    orphans = re.findall(r"<p><img\s+([^>]*)/?>\s*</p>", body)
    if orphans:
        srcs = [SRC_RE.search(a).group(1) if SRC_RE.search(a) else "<unknown src>" for a in orphans]
        sys.exit(
            "_render.py: image(s) without a merged caption: "
            + ", ".join(srcs)
            + "\nEach image paragraph must be immediately followed by a single"
            " *italic* caption paragraph (no blank line inside, no intervening"
            " paragraph, no nested *italics* within the caption)."
        )


def render(
    fragment: str,
    *,
    title: str,
    source_dir: Path,
    out_dir: Path,
    mode: str,
    css: str,
    js: str,
) -> str:
    # Split off the H1 + immediate byline paragraphs into a styled header,
    # then merge every image+caption pair into a lightbox-ready <figure>.
    body = FIG_PATTERN.sub(
        lambda m: build_figure(m, source_dir=source_dir, mode=mode, out_dir=out_dir),
        fragment,
    )
    _check_orphan_images(body)

    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>\s*", body, re.DOTALL)
    header_html = ""
    if h1_match:
        rest = body[h1_match.end() :]
        # Byline + kicker are the two short paragraphs pandoc emits right
        # after the title in post.md ("**By ...**" and "*Part N of two.*").
        byline_pat = re.compile(r"^\s*(<p><strong>.*?</strong></p>)\s*(<p><em>.*?</em></p>)?", re.DOTALL)
        byline_match = byline_pat.match(rest)
        byline_html = ""
        if byline_match:
            byline = byline_match.group(1) or ""
            kicker = byline_match.group(2) or ""
            byline = byline.replace("<p><strong>", '<p class="byline"><strong>')
            kicker = kicker.replace("<p><em>", '<p class="byline kicker"><em>')
            byline_html = byline + kicker
            body = rest[byline_match.end() :]
        else:
            body = rest
        header_html = f'<header class="post-header"><h1>{h1_match.group(1)}</h1>{byline_html}</header>\n'
    else:
        header_html = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{title}">
<style>
{css}
</style>
</head>
<body>
{header_html}<article>
{body}
</article>
<footer class="post-footer">
<p>Agentic Forecasting series &mdash; Vector AI Engineering.</p>
</footer>
<div class="lightbox-overlay" id="lightbox">
<button class="lightbox-close" type="button" aria-label="Close">&times;</button>
<img alt="">
</div>
<div class="lightbox-hint" aria-hidden="true">Esc or click outside to close</div>
<script>
{js}
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--source-dir", required=True, type=Path, help="dir containing post.md")
    ap.add_argument("--out-dir", required=True, type=Path, help="dir the output file lives in")
    ap.add_argument("--mode", choices=["embed", "linked"], required=True)
    ap.add_argument("--css", required=True, type=Path)
    ap.add_argument("--js", required=True, type=Path)
    args = ap.parse_args()

    fragment = sys.stdin.read()
    css = args.css.read_text()
    js = args.js.read_text()
    html = render(
        fragment,
        title=args.title,
        source_dir=args.source_dir.resolve(),
        out_dir=args.out_dir.resolve(),
        mode=args.mode,
        css=css,
        js=js,
    )
    sys.stdout.write(html)


if __name__ == "__main__":
    main()
