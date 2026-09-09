#!/usr/bin/env python3
"""Render a Markdown document as plain text for pasting into Google Docs.

- Headings lose their '#' markers (level kept as a blank line before them).
- **bold** / *italic* / `code` markers are stripped.
- Images become [filename] placeholders on their own line.
- Tables become tab-separated rows (paste into Docs, then Format > Table, or
  paste via Sheets), separator rows are dropped.
- Wrapped paragraph and list-item lines are re-joined into single lines.

Usage:
    python tools/md_to_gdoc_txt.py docs/proposal/MLPerf_Training_proposal_Qwen3-VL.md
    (writes the .txt next to the .md, or pass --out)
"""

import argparse
import re
from pathlib import Path

IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
BOLD_ITALIC = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
CODE = re.compile(r"`([^`]*)`")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")
LIST_ITEM = re.compile(r"^(\s*)([*+-]|\d+\.)\s+")


def inline(text):
    text = IMAGE.sub(lambda m: f"[{Path(m.group(1)).name}]", text)
    text = LINK.sub(r"\1 (\2)", text)
    text = CODE.sub(r"\1", text)
    for _ in range(2):  # nested emphasis
        text = BOLD_ITALIC.sub(r"\2", text)
    return text


def table_row(line):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return "\t".join(inline(c) for c in cells)


def convert(md):
    out = []
    para = []  # raw wrapped lines of the current paragraph / list item / image

    def flush():
        # join wrapped lines first, then strip inline markup, so emphasis or
        # image syntax spanning a line break is handled; keep the first line's
        # indentation for nested list items
        if para:
            indent = para[0][: len(para[0]) - len(para[0].lstrip())]
            out.append(indent + inline(" ".join(s.strip() for s in para)))
            para.clear()

    for raw in md.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            out.append("")
            continue
        if stripped.startswith("#"):
            flush()
            out.append(inline(stripped.lstrip("#").strip()))
            continue
        if stripped.startswith("|"):
            flush()
            if not TABLE_SEP.match(stripped):
                out.append(table_row(stripped))
            continue
        match = LIST_ITEM.match(line)
        if match:
            flush()
            indent, marker = match.groups()
            bullet = marker if marker[0].isdigit() else "*"
            para.append(f"{indent}{bullet} {LIST_ITEM.sub('', line)}")
            continue
        para.append(line)  # paragraph text, continuation line, or (wrapped) image

    flush()
    # collapse runs of blank lines to at most one
    text, blank = [], False
    for line in out:
        if line == "":
            if not blank:
                text.append("")
            blank = True
        else:
            text.append(line)
            blank = False
    return "\n".join(text).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", help="Input .md file")
    parser.add_argument("--out", help="Output .txt path (default: alongside the input)")
    args = parser.parse_args()
    src = Path(args.markdown)
    dst = Path(args.out) if args.out else src.with_suffix(".txt")
    dst.write_text(convert(src.read_text()))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
