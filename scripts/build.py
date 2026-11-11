"""Build the site into _site/.

- Copies every folder in pages/ as-is.
- Renders pages/<slug>/index.md to index.html if there is no index.html.
- Generates a root index.html listing all pages.
"""
import html
import json
import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "pages"
OUT = ROOT / "_site"

STYLE = """
body{font-family:system-ui,sans-serif;max-width:46rem;margin:3rem auto;padding:0 1rem;line-height:1.6;color:#222;background:#fdfdfb}
a{color:#2457c5}h1{font-size:1.8rem}
ul.pages{list-style:none;padding:0}ul.pages li{padding:.7rem 0;border-bottom:1px solid #eee}
code,pre{background:#f3f3f0;border-radius:4px}pre{padding:1rem;overflow:auto}
@media(prefers-color-scheme:dark){body{background:#151515;color:#ddd}a{color:#7fa8ff}code,pre{background:#222}ul.pages li{border-color:#333}}
"""


def page_title(folder: Path) -> str:
    for name in ("index.html", "index.md"):
        f = folder / name
        if f.exists():
            text = f.read_text(encoding="utf-8")
            m = re.search(r"<title>(.*?)</title>", text, re.S | re.I) or re.search(r"^#\s+(.+)$", text, re.M)
            if m:
                return m.group(1).strip()
    return folder.name.replace("-", " ").title()


def wrap(title: str, body: str) -> str:
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
            f"<body>{body}</body></html>")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "pages").mkdir(parents=True)
    # Folders listed under their own heading instead of "Workbooks"
    SPECIAL_SECTIONS = {
        "encrypted": "Encrypted pages",
        "interview-prep": "Encrypted pages",
        "trips": "Encrypted pages",
        "hello-workbook": "Tools",
    }
    entries = []
    # Group special pages by heading (multiple folders can share a heading)
    from collections import OrderedDict
    section_pages: dict[str, list[tuple[str, str]]] = OrderedDict()
    for heading in dict.fromkeys(SPECIAL_SECTIONS.values()):
        section_pages[heading] = []
    for folder in sorted(p for p in PAGES.iterdir() if p.is_dir()) if PAGES.exists() else []:
        dest = OUT / "pages" / folder.name
        shutil.copytree(folder, dest)
        md = dest / "index.md"
        if md.exists() and not (dest / "index.html").exists():
            title = page_title(folder)
            body = markdown.markdown(md.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
            (dest / "index.html").write_text(wrap(title, "<p><a href='../../'>← All workbooks</a></p>" + body), encoding="utf-8")
        if folder.name in SPECIAL_SECTIONS:
            section_pages[SPECIAL_SECTIONS[folder.name]].append((folder.name, page_title(folder)))
        else:
            entries.append((folder.name, page_title(folder)))
    items = "".join(f"<li><a href='pages/{s}/'>{html.escape(t)}</a></li>" for s, t in entries) or "<li>No pages yet.</li>"
    links_file = ROOT / "links.json"
    links = json.loads(links_file.read_text(encoding="utf-8")) if links_file.exists() else []
    tool_links = [l for l in links if l.get("section") == "Tools"]
    links = [l for l in links if l.get("section") != "Tools"]
    more = ""
    if links:
        li = "".join(f"<li><a href='{html.escape(l['url'], quote=True)}'>{html.escape(l['title'])}</a></li>" for l in links)
        more = f"<h2>More sites</h2><ul class='pages'>{li}</ul>"
    # Build special sections (grouped by heading)
    special_html = ""
    for heading, pages in section_pages.items():
        if pages:
            li = "".join(f"<li><a href='pages/{s}/'>{html.escape(t)}</a></li>" for s, t in pages)
            if heading == "Tools":
                li += "".join(f"<li><a href='{html.escape(l['url'], quote=True)}'>{html.escape(l['title'])}</a></li>" for l in tool_links)
            special_html += f"<h2>{html.escape(heading)}</h2><ul class='pages'>{li}</ul>"
    (OUT / "index.html").write_text(wrap("Workbooks", f"<h1>Workbooks</h1><ul class='pages'>{items}</ul>{special_html}{more}"), encoding="utf-8")
    (OUT / ".nojekyll").touch()
    print(f"Built {len(entries)} page(s) into {OUT}")


if __name__ == "__main__":
    main()
