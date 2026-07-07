#!/usr/bin/env python
"""Render manual.html -> the distributed PDF, then add a navigable outline.

Headless Chrome renders the print-styled HTML; PyMuPDF then reads toc.json and
attaches a nested bookmark outline by locating each heading in the rendered
pages (Chrome's print-to-pdf does not emit a document outline itself).

Usage:  python render_pdf.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import fitz  # PyMuPDF

HERE = Path(__file__).resolve().parent
HTML = HERE / "manual.html"
TOC = HERE / "toc.json"
OUT = HERE.parent / "GeotechStaffEngineer_User_Manual_v5.3.pdf"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    raise SystemExit("No Chrome/Edge found for PDF rendering; edit CHROME_CANDIDATES.")


def render():
    chrome = find_chrome()
    url = HTML.resolve().as_uri()
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={OUT}", url]
    print("Rendering with:", Path(chrome).name)
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def add_outline():
    doc = fitz.open(OUT)
    entries = json.loads(TOC.read_text(encoding="utf-8"))

    # Start the search past the front matter (cover / disclaimer / contents).
    # The Contents pages list every title, so a naive title search would collide
    # with them. The body Chapter-1 heading is the first page whose kicker reads
    # "CHAPTER 1" (rendered letter-spaced, so match on the space-squished text).
    cursor = 0
    for i in range(doc.page_count):
        if "CHAPTER1" in doc[i].get_text().replace(" ", ""):
            cursor = i
            break

    toc = []
    for level, outline_title, search_title in entries:
        page = cursor
        needle = search_title[:38]
        for i in range(cursor, doc.page_count):
            if needle and needle in doc[i].get_text():
                page = i
                break
        cursor = max(cursor, page)
        toc.append([int(level), outline_title, page + 1])  # 1-based page

    doc.set_toc(toc)
    doc.saveIncr()
    doc.close()
    print(f"Outline: {len(toc)} entries attached to {OUT.name}")


def verify():
    doc = fitz.open(OUT)
    n = doc.page_count
    blank = sum(1 for i in range(n) if len(doc[i].get_text().strip()) < 5)
    print(f"Pages: {n}   blank/near-blank: {blank}   outline: {len(doc.get_toc())}")
    doc.close()


if __name__ == "__main__":
    render()
    add_outline()
    verify()
