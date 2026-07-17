#!/usr/bin/env bash
# Render blog/part-N/post.md to publication-ready HTML.
#
#   ./render.sh                  # render every part-*/post.md (self-contained)
#   ./render.sh part-1           # render just blog/part-1/post.md
#   ./render.sh --assets-linked  # also emit a linked-images variant for CMS handoff
#
# Self-contained mode (default): CSS and JS are inlined and every figure is
# embedded as a base64 data URI, so blog/dist/part-N.html is a single file
# that can be emailed or opened anywhere with no external dependencies.
#
# --assets-linked mode: same CSS/JS/markup, but <img> src stays a relative
# path into part-N/assets/ instead of a data URI -- for handing to a CMS
# that wants to manage images itself. Written to
# blog/dist/part-N.assets-linked.html so it never clobbers the canonical file.
#
# Requires pandoc (`brew install pandoc` / `apt install pandoc`) for the
# markdown -> HTML conversion. If pandoc isn't on PATH, falls back to
# Python's `markdown` package (`pip install markdown`) with a note printed
# to stderr -- the fallback covers the same markdown subset post.md uses
# (headings, paragraphs, emphasis, links, images) but has not been battle
# tested on edge-case markdown the way pandoc has.
set -euo pipefail
cd "$(dirname "$0")"

CSS="assets/blog.css"
JS="assets/lightbox.js"
PY="$(command -v python3 || command -v python)"

MODE_LINKED=0
TARGETS=()
for arg in "$@"; do
  case "$arg" in
    --assets-linked) MODE_LINKED=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    part-*) TARGETS+=("${arg%/}") ;;
    *) echo "render.sh: unrecognized argument '$arg'" >&2; exit 1 ;;
  esac
done

if [ ${#TARGETS[@]} -eq 0 ]; then
  for d in part-*/; do
    [ -f "${d}post.md" ] && TARGETS+=("${d%/}")
  done
fi

if [ ${#TARGETS[@]} -eq 0 ]; then
  echo "render.sh: no part-*/post.md found under $(pwd)" >&2
  exit 1
fi

mkdir -p dist

HAVE_PANDOC=0
if command -v pandoc >/dev/null 2>&1; then
  HAVE_PANDOC=1
  echo "using pandoc ($(pandoc --version | head -1))"
else
  echo "pandoc not found on PATH -- falling back to Python's 'markdown' package." >&2
  echo "Install pandoc for best fidelity: https://pandoc.org/installing.html" >&2
  "$PY" -c "import markdown" 2>/dev/null || {
    echo "render.sh: neither pandoc nor the Python 'markdown' package is available." >&2
    echo "Install one of: 'brew install pandoc' or 'pip install markdown'." >&2
    exit 1
  }
fi

markdown_to_fragment() {
  # $1 = path to post.md; fragment HTML on stdout.
  local md="$1"
  if [ "$HAVE_PANDOC" -eq 1 ]; then
    # -implicit_figures: keep the image and its following italic paragraph
    # as two separate <p> tags (post.md's caption convention), rather than
    # pandoc auto-building a <figure> from the alt text.
    pandoc "$md" -f markdown-implicit_figures -t html5
  else
    "$PY" -c '
import sys, markdown
print(markdown.markdown(open(sys.argv[1]).read(), extensions=["extra"]))
' "$md"
  fi
}

render_one() {
  local dir="$1"                      # e.g. part-1
  local md="${dir}/post.md"
  local title
  title="$(sed -n "s/^# //p" "$md" | head -1)"
  [ -n "$title" ] || title="$dir"

  local out="dist/${dir}.html"
  markdown_to_fragment "$md" \
    | "$PY" _render.py \
        --title "$title" \
        --source-dir "$dir" \
        --out-dir "dist" \
        --mode embed \
        --css "$CSS" \
        --js "$JS" \
    > "$out"
  echo "rendered  $out  ($(du -h "$out" | cut -f1), self-contained)"

  if [ "$MODE_LINKED" -eq 1 ]; then
    local out_linked="dist/${dir}.assets-linked.html"
    markdown_to_fragment "$md" \
      | "$PY" _render.py \
          --title "$title" \
          --source-dir "$dir" \
          --out-dir "dist" \
          --mode linked \
          --css "$CSS" \
          --js "$JS" \
      > "$out_linked"
    echo "rendered  $out_linked  ($(du -h "$out_linked" | cut -f1), linked images)"
  fi
}

for t in "${TARGETS[@]}"; do
  render_one "$t"
done
