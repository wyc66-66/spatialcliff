#!/usr/bin/env python3
"""Render the SpatialCliff technical report Markdown -> styled HTML -> PDF.

Uses the Python `markdown` package for HTML and headless Edge for PDF
(no external network deps). Outputs under docs/paper/spatialcliff/:
    spatialcliff_paper.html   styled standalone HTML
    spatialcliff_paper.pdf    print-to-pdf via headless Edge
"""
from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path

import markdown  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper" / "spatialcliff"
SRC = PAPER / "spatialcliff_paper.md"
FIG = PAPER / "figures"
OUT_HTML = PAPER / "spatialcliff_paper.html"
OUT_PDF = PAPER / "spatialcliff_paper.pdf"

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
:root{--ink:#e6edf3;--mut:#8b949e;--line:#2d3744;--acc:#58a6ff;--bg:#0d1117;}
*{box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;color:var(--ink);
  background:var(--bg);line-height:1.62;max-width:820px;margin:0 auto;padding:40px 48px 72px;font-size:14.5px}
h1{font-size:26px;line-height:1.25;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:19px;margin:30px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:15.5px;margin:20px 0 6px}
p{margin:10px 0}
code{font-family:Consolas,'Cascadia Mono',monospace;font-size:.9em;background:#1c2430;
  padding:1px 4px;border-radius:3px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left}
th{background:#1c2430;font-weight:600}
tr:nth-child(even) td{background:#131a24}
img{max-width:100%;border:1px solid var(--line);border-radius:6px;margin:8px 0}
figure{margin:16px 0}
figcaption{font-size:12px;color:var(--mut);margin-top:2px}
blockquote{border-left:3px solid var(--acc);margin:12px 0;padding:2px 16px;
  background:#131a24;color:#c9d4e0}
a{color:var(--acc)}
hr{border:0;border-top:1px solid var(--line);margin:28px 0}
@media print{
  body{padding:0 4mm;background:#fff;color:#111318}
  h2{page-break-before:auto}
  figure{page-break-inside:avoid}
  a{color:inherit;text-decoration:none}
}
"""


def embed_figures(html: str) -> str:
    """Replace markdown-rendered <img src="figures/figX.png"> with inline base64
    data URIs so the HTML is fully standalone (works from file://)."""

    def repl(m: re.Match) -> str:
        cap, fname = m.group(1), m.group(2)
        path = FIG / fname
        if not path.is_file():
            return m.group(0)
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<figure><img alt="{cap}" src="data:image/png;base64,{b64}"/><figcaption>{cap}</figcaption></figure>'

    return re.sub(
        r'<img alt="([^"]*)" src="figures/(fig\d+_\w+\.png)"\s*/?>',
        repl,
        html,
    )


def main() -> int:
    md = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    body = embed_figures(body)
    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<title>SpatialCliff</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[paper] HTML -> {OUT_HTML}")

    edge = next((p for p in EDGE_CANDIDATES if Path(p).is_file()), None)
    if edge is None:
        print("[paper] no Edge/Chrome found; PDF skipped (HTML is standalone).")
        return 0
    cmd = [
        edge,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUT_PDF}",
        OUT_HTML.as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if OUT_PDF.is_file():
        print(f"[paper] PDF -> {OUT_PDF} ({OUT_PDF.stat().st_size} bytes)")
        return 0
    print("[paper] PDF failed; HTML remains usable.")
    print(proc.stderr[-500:] if proc.stderr else "")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
