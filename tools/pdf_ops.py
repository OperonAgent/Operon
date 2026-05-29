"""
Operon PDF Operations Tool.

Full PDF manipulation powered by pypdf + reportlab.
Handles: creation, merging, splitting, text extraction, watermarking,
         encryption, rotation, metadata editing, page reordering.

Install:
    pip install pypdf reportlab

All functions return a dict: {success, output, error}
"""

from __future__ import annotations

import io
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ── lazy imports so the module loads even without the optional packages ────────

def _pypdf():
    try:
        from pypdf import PdfReader, PdfWriter
        return PdfReader, PdfWriter
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")

def _reportlab():
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
            Preformatted,
        )
        from reportlab.lib.colors import HexColor
        return {
            "letter": letter, "A4": A4,
            "getSampleStyleSheet": getSampleStyleSheet,
            "inch": inch,
            "SimpleDocTemplate": SimpleDocTemplate,
            "Paragraph": Paragraph,
            "Spacer": Spacer,
            "PageBreak": PageBreak,
            "Table": Table,
            "TableStyle": TableStyle,
            "Preformatted": Preformatted,
            "HexColor": HexColor,
        }
    except ImportError:
        raise ImportError("reportlab not installed. Run: pip install reportlab")

def _ok(output: Any) -> dict:
    return {"success": True,  "output": output, "error": None}

def _err(msg: str) -> dict:
    return {"success": False, "output": None,   "error": msg}


# ── PDF Reading & Extraction ──────────────────────────────────────────────────

def pdf_extract_text(
    path: str,
    pages: Optional[str] = None,  # e.g. "1-3,5,7-9"  (1-indexed), None = all
) -> dict:
    """
    Extract text from a PDF file.

    Args:
        path:  Path to the PDF file.
        pages: Page range string, e.g. "1-3,5,7" (1-indexed). None = all pages.

    Returns:
        {success, output: str (extracted text), error}
    """
    PdfReader, _ = _pypdf()
    try:
        reader = PdfReader(path)
        total  = len(reader.pages)
        indices = _parse_page_range(pages, total) if pages else list(range(total))
        chunks  = []
        for i in indices:
            page_text = reader.pages[i].extract_text() or ""
            chunks.append(f"[Page {i+1}]\n{page_text}")
        return _ok("\n\n".join(chunks))
    except FileNotFoundError:
        return _err(f"File not found: {path}")
    except Exception as e:
        return _err(f"pdf_extract_text failed: {e}")


def pdf_info(path: str) -> dict:
    """
    Return metadata and page count for a PDF.

    Args:
        path: Path to the PDF file.

    Returns:
        {success, output: dict with title/author/pages/encrypted/size, error}
    """
    PdfReader, _ = _pypdf()
    try:
        reader = PdfReader(path)
        meta   = reader.metadata or {}
        size   = os.path.getsize(path)
        return _ok({
            "pages":     len(reader.pages),
            "encrypted": reader.is_encrypted,
            "size_bytes": size,
            "size_kb":   round(size / 1024, 1),
            "title":     meta.get("/Title",    ""),
            "author":    meta.get("/Author",   ""),
            "subject":   meta.get("/Subject",  ""),
            "creator":   meta.get("/Creator",  ""),
            "producer":  meta.get("/Producer", ""),
        })
    except FileNotFoundError:
        return _err(f"File not found: {path}")
    except Exception as e:
        return _err(f"pdf_info failed: {e}")


# ── PDF Manipulation ──────────────────────────────────────────────────────────

def pdf_merge(
    paths: List[str],
    output: str,
) -> dict:
    """
    Merge multiple PDFs into one file.

    Args:
        paths:  List of PDF file paths to merge (in order).
        output: Output file path.

    Returns:
        {success, output: str (output path), error}
    """
    PdfReader, PdfWriter = _pypdf()
    try:
        writer = PdfWriter()
        for p in paths:
            reader = PdfReader(p)
            for page in reader.pages:
                writer.add_page(page)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "wb") as f:
            writer.write(f)
        return _ok(f"Merged {len(paths)} PDFs → {output}  ({_pagecount(output)} pages)")
    except FileNotFoundError as e:
        return _err(f"File not found: {e}")
    except Exception as e:
        return _err(f"pdf_merge failed: {e}")


def pdf_split(
    path: str,
    output_dir: str,
    pages_per_file: int = 1,
) -> dict:
    """
    Split a PDF into multiple files.

    Args:
        path:           Input PDF path.
        output_dir:     Directory to write output files.
        pages_per_file: Pages per output file (default 1 = one file per page).

    Returns:
        {success, output: list of created file paths, error}
    """
    PdfReader, PdfWriter = _pypdf()
    try:
        reader  = PdfReader(path)
        total   = len(reader.pages)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem    = Path(path).stem
        created = []
        chunk   = 0
        i       = 0
        while i < total:
            writer = PdfWriter()
            end    = min(i + pages_per_file, total)
            for j in range(i, end):
                writer.add_page(reader.pages[j])
            out_path = out_dir / f"{stem}_part{chunk+1:03d}.pdf"
            with open(out_path, "wb") as f:
                writer.write(f)
            created.append(str(out_path))
            chunk += 1
            i      = end
        return _ok(created)
    except FileNotFoundError:
        return _err(f"File not found: {path}")
    except Exception as e:
        return _err(f"pdf_split failed: {e}")


def pdf_rotate(
    path: str,
    output: str,
    degrees: int = 90,
    pages: Optional[str] = None,  # None = all pages
) -> dict:
    """
    Rotate pages in a PDF.

    Args:
        path:    Input PDF path.
        output:  Output PDF path.
        degrees: Rotation in degrees (90, 180, 270).
        pages:   Page range string (1-indexed). None = all pages.

    Returns:
        {success, output: str, error}
    """
    PdfReader, PdfWriter = _pypdf()
    if degrees not in (90, 180, 270):
        return _err("degrees must be 90, 180, or 270")
    try:
        reader  = PdfReader(path)
        writer  = PdfWriter()
        total   = len(reader.pages)
        targets = set(_parse_page_range(pages, total) if pages else range(total))
        for i, page in enumerate(reader.pages):
            if i in targets:
                page.rotate(degrees)
            writer.add_page(page)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "wb") as f:
            writer.write(f)
        return _ok(f"Rotated {len(targets)} pages by {degrees}° → {output}")
    except FileNotFoundError:
        return _err(f"File not found: {path}")
    except Exception as e:
        return _err(f"pdf_rotate failed: {e}")


def pdf_watermark(
    path: str,
    output: str,
    text: str,
    opacity: float = 0.3,
    font_size: int = 48,
    color: str = "gray",
) -> dict:
    """
    Add a diagonal text watermark to every page of a PDF.

    Args:
        path:      Input PDF path.
        output:    Output PDF path.
        text:      Watermark text (e.g. "CONFIDENTIAL").
        opacity:   Opacity 0.0–1.0 (default 0.3).
        font_size: Font size (default 48).
        color:     Color name or hex (default "gray").

    Returns:
        {success, output: str, error}
    """
    PdfReader, PdfWriter = _pypdf()
    rl = _reportlab()
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.colors import Color, HexColor
        import math

        # Determine color
        try:
            if color.startswith("#"):
                wm_color = HexColor(color)
            else:
                _color_map = {
                    "gray": Color(0.5, 0.5, 0.5, opacity),
                    "red":  Color(1.0, 0.0, 0.0, opacity),
                    "blue": Color(0.0, 0.0, 1.0, opacity),
                    "black": Color(0.0, 0.0, 0.0, opacity),
                }
                wm_color = _color_map.get(color.lower(), Color(0.5, 0.5, 0.5, opacity))
        except Exception:
            wm_color = Color(0.5, 0.5, 0.5, opacity)

        # Build watermark on a BytesIO buffer
        buf = io.BytesIO()
        reader = PdfReader(path)
        first  = reader.pages[0]
        # Try to get page dimensions
        w = float(first.mediabox.width)
        h = float(first.mediabox.height)

        c = rl_canvas.Canvas(buf, pagesize=(w, h))
        c.setFillColor(wm_color)
        c.setFont("Helvetica-Bold", font_size)
        c.saveState()
        c.translate(w / 2, h / 2)
        c.rotate(45)
        c.drawCentredString(0, 0, text)
        c.restoreState()
        c.save()

        buf.seek(0)
        wm_reader = PdfReader(buf)
        wm_page   = wm_reader.pages[0]

        writer = PdfWriter()
        for page in reader.pages:
            page.merge_page(wm_page)
            writer.add_page(page)

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "wb") as f:
            writer.write(f)
        return _ok(f"Watermark '{text}' applied → {output}")
    except FileNotFoundError:
        return _err(f"File not found: {path}")
    except Exception as e:
        return _err(f"pdf_watermark failed: {e}")


def pdf_encrypt(
    path: str,
    output: str,
    user_password: str,
    owner_password: Optional[str] = None,
) -> dict:
    """
    Password-protect a PDF.

    Args:
        path:           Input PDF path.
        output:         Output PDF path.
        user_password:  Password required to open the PDF.
        owner_password: Owner password (defaults to user_password).

    Returns:
        {success, output: str, error}
    """
    PdfReader, PdfWriter = _pypdf()
    try:
        reader = PdfReader(path)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.encrypt(user_password, owner_password or user_password)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "wb") as f:
            writer.write(f)
        return _ok(f"Encrypted PDF saved → {output}")
    except FileNotFoundError:
        return _err(f"File not found: {path}")
    except Exception as e:
        return _err(f"pdf_encrypt failed: {e}")


def pdf_decrypt(
    path: str,
    output: str,
    password: str,
) -> dict:
    """
    Remove password protection from a PDF.

    Args:
        path:     Encrypted PDF path.
        output:   Output (decrypted) PDF path.
        password: The PDF password.

    Returns:
        {success, output: str, error}
    """
    PdfReader, PdfWriter = _pypdf()
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            reader.decrypt(password)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "wb") as f:
            writer.write(f)
        return _ok(f"Decrypted PDF saved → {output}")
    except FileNotFoundError:
        return _err(f"File not found: {path}")
    except Exception as e:
        return _err(f"pdf_decrypt failed: {e}")


def pdf_extract_pages(
    path: str,
    output: str,
    pages: str,
) -> dict:
    """
    Extract specific pages from a PDF into a new file.

    Args:
        path:   Input PDF path.
        output: Output PDF path.
        pages:  Page range, e.g. "1-3,5,8-10" (1-indexed).

    Returns:
        {success, output: str, error}
    """
    PdfReader, PdfWriter = _pypdf()
    try:
        reader  = PdfReader(path)
        total   = len(reader.pages)
        indices = _parse_page_range(pages, total)
        writer  = PdfWriter()
        for i in indices:
            writer.add_page(reader.pages[i])
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "wb") as f:
            writer.write(f)
        return _ok(f"Extracted {len(indices)} pages → {output}")
    except FileNotFoundError:
        return _err(f"File not found: {path}")
    except Exception as e:
        return _err(f"pdf_extract_pages failed: {e}")


# ── PDF Creation ──────────────────────────────────────────────────────────────

def pdf_create(
    output: str,
    content: str,
    title: str = "Document",
    author: str = "Operon",
    font_size: int = 11,
    page_size: str = "letter",
    margins: float = 1.0,
) -> dict:
    """
    Create a new PDF from plain text or Markdown-like content.

    Args:
        output:    Output PDF path.
        content:   Text content. Headings: lines starting with # / ## / ###.
                   Bold: **text**. Code: ```blocks```.
        title:     PDF metadata title.
        author:    PDF metadata author.
        font_size: Body font size (default 11).
        page_size: "letter" or "a4".
        margins:   Margin in inches (default 1.0).

    Returns:
        {success, output: str, error}
    """
    rl = _reportlab()
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
        from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY

        ps    = letter if page_size.lower() == "letter" else A4
        m     = margins * inch
        styles = getSampleStyleSheet()

        h1_style = ParagraphStyle("h1", fontSize=font_size+6, leading=font_size+10,
                                  fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=8)
        h2_style = ParagraphStyle("h2", fontSize=font_size+3, leading=font_size+7,
                                  fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
        h3_style = ParagraphStyle("h3", fontSize=font_size+1, leading=font_size+5,
                                  fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
        body_style = ParagraphStyle("body", fontSize=font_size, leading=font_size+4,
                                    fontName="Helvetica", alignment=TA_JUSTIFY,
                                    spaceBefore=3, spaceAfter=3)
        code_style = ParagraphStyle("code", fontSize=font_size-1.5, leading=font_size+2,
                                    fontName="Courier", spaceBefore=6, spaceAfter=6)

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            output, pagesize=ps,
            leftMargin=m, rightMargin=m, topMargin=m, bottomMargin=m,
            title=title, author=author,
        )

        story = []
        in_code = False
        code_buf: List[str] = []

        for line in content.splitlines():
            # Code block toggle
            if line.strip().startswith("```"):
                if in_code:
                    # flush code block
                    code_text = "\n".join(code_buf)
                    story.append(Preformatted(
                        code_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"),
                        code_style
                    ))
                    code_buf = []
                    in_code = False
                else:
                    in_code = True
                continue

            if in_code:
                code_buf.append(line)
                continue

            # Headings
            if line.startswith("### "):
                story.append(Paragraph(line[4:], h3_style))
            elif line.startswith("## "):
                story.append(Paragraph(line[3:], h2_style))
            elif line.startswith("# "):
                story.append(Paragraph(line[2:], h1_style))
            elif line.strip() == "":
                story.append(Spacer(1, 8))
            else:
                # Inline bold: **text**
                safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
                story.append(Paragraph(safe, body_style))

        if in_code and code_buf:
            story.append(Preformatted("\n".join(code_buf), code_style))

        doc.build(story)
        size_kb = round(os.path.getsize(output) / 1024, 1)
        return _ok(f"PDF created → {output}  ({size_kb} KB)")
    except Exception as e:
        return _err(f"pdf_create failed: {e}")


def pdf_create_from_html(
    output: str,
    html: str,
) -> dict:
    """
    Create a PDF from an HTML string (requires weasyprint).

    Args:
        output: Output PDF path.
        html:   HTML content string.

    Returns:
        {success, output: str, error}
    """
    try:
        from weasyprint import HTML
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html).write_pdf(output)
        return _ok(f"PDF from HTML saved → {output}")
    except ImportError:
        return _err("weasyprint not installed. Run: pip install weasyprint")
    except Exception as e:
        return _err(f"pdf_create_from_html failed: {e}")


# ── Utility ───────────────────────────────────────────────────────────────────

def _parse_page_range(spec: str, total: int) -> List[int]:
    """Parse "1-3,5,7-9" into 0-indexed list. Clamps to valid range."""
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            start = max(0, int(a.strip()) - 1)
            end   = min(total - 1, int(b.strip()) - 1)
            indices.extend(range(start, end + 1))
        else:
            idx = int(part.strip()) - 1
            if 0 <= idx < total:
                indices.append(idx)
    return sorted(set(indices))


def _pagecount(path: str) -> int:
    PdfReader, _ = _pypdf()
    return len(PdfReader(path).pages)


import re  # used in pdf_create — placed here to avoid top-level import order issues
