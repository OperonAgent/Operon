#!/usr/bin/env python3
"""
Generate Operon AI Terminal Cockpit — Full PDF Documentation
Output: ~/Desktop/Operon_Documentation.pdf
"""

import os
import sys
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image, KeepTogether,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.colors import HexColor

# ── Brand colours ─────────────────────────────────────────────────────────────
BLACK        = HexColor("#0A0A0F")   # page background
DEEP_PURPLE  = HexColor("#1A0A2E")
PURPLE_BASE  = HexColor("#7B2FBE")
PURPLE_LIGHT = HexColor("#9D5CE5")
PURPLE_NEON  = HexColor("#C084FC")
CYAN_GLOW    = HexColor("#22D3EE")
WHITE_BRIGHT = HexColor("#F8FAFC")
GRAY_TEXT    = HexColor("#94A3B8")
GRAY_LIGHT   = HexColor("#CBD5E1")
GRAY_DARK    = HexColor("#141B2D")   # slightly brighter than BLACK so rows contrast
GRAY_MID     = HexColor("#1E2D45")   # table headers / mid-tone
GRAY_ROW     = HexColor("#0F1623")   # alternate table row (darker stripe)
GREEN_OK     = HexColor("#22C55E")
ORANGE_WARN  = HexColor("#F97316")

PAGE_W, PAGE_H = A4   # 595.28 x 841.89

OUTPUT_PATH = Path(__file__).resolve().parent / "Operon_Documentation.pdf"
# Prefer the newer triangle logo; fall back to the original
_LOGO_CANDIDATES = [
    Path.home() / "Downloads" / "Operon Logo.png",
    Path.home() / "Downloads" / "Operon Logo 1.png",
    Path.home() / "Downloads" / "operon_logo.png",
    Path.home() / "Downloads" / "operon logo.png",
    Path.home() / "Desktop"   / "Operon Logo.png",
    Path.home() / "Downloads" / "ChatGPT Image May 20, 2026, 09_04_08 PM.png",
    Path.home() / "Downloads" / "ChatGPT Image May 20, 2026, 08_54_55 PM.png",
]
LOGO_PATH = next((p for p in _LOGO_CANDIDATES if p.exists()), _LOGO_CANDIDATES[0])


# ── Custom canvas (page numbers + header/footer) ───────────────────────────────

class OperonCanvas(pdf_canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        page_count = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_frame(page_count)
            super().showPage()
        super().save()

    def _draw_page_frame(self, page_count):
        page_num = self._pageNumber

        # Background is NOT drawn here — it is painted by OperonDocTemplate
        # .handle_pageBegin() BEFORE the platypus content, so it sits under
        # the text rather than covering it.

        # ── Cover-page overlay (minimal — just a watermark badge) ─────────────
        if page_num == 1:
            self.saveState()
            # "v2.0" badge — bottom-right corner inside the dark panel
            self.setFillColor(PURPLE_BASE)
            bx, by, bw, bh = PAGE_W - 38*mm, 5*mm, 26*mm, 9*mm
            self.roundRect(bx, by, bw, bh, 2, fill=1, stroke=0)
            self.setFillColor(WHITE_BRIGHT)
            self.setFont("Helvetica-Bold", 7.5)
            badge_txt = "v3.1.0  RELEASE"
            btw = self.stringWidth(badge_txt, "Helvetica-Bold", 7.5)
            self.drawString(bx + (bw - btw)/2, by + 1.8*mm, badge_txt)
            self.restoreState()
            return

        # ── Footer bar (all other pages) ─────────────────────────────────────
        self.saveState()

        # Footer background
        self.setFillColor(HexColor("#0D1117"))
        self.rect(0, 0, PAGE_W, 18*mm, fill=1, stroke=0)

        # Purple separator line above footer
        self.setStrokeColor(PURPLE_BASE)
        self.setLineWidth(0.8)
        self.line(0, 18*mm, PAGE_W, 18*mm)

        # Left: product name
        self.setFillColor(PURPLE_NEON)
        self.setFont("Helvetica-Bold", 7)
        self.drawString(15*mm, 6.5*mm, "OPERON  AI TERMINAL COCKPIT")

        # Centre: page X of Y
        self.setFillColor(GRAY_TEXT)
        self.setFont("Helvetica", 7)
        txt = f"Page {page_num} of {page_count}"
        tw  = self.stringWidth(txt, "Helvetica", 7)
        self.drawString((PAGE_W - tw) / 2, 6.5*mm, txt)

        # Right: version
        self.setFillColor(CYAN_GLOW)
        self.setFont("Helvetica-Bold", 7)
        ver = "v3.1.0  •  2026"
        vw  = self.stringWidth(ver, "Helvetica-Bold", 7)
        self.drawString(PAGE_W - 15*mm - vw, 6.5*mm, ver)

        # Top-of-page accent line
        self.setStrokeColor(HexColor("#2A1060"))
        self.setLineWidth(0.4)
        self.line(0, PAGE_H - 5*mm, PAGE_W, PAGE_H - 5*mm)

        self.restoreState()


# ── DocTemplate ───────────────────────────────────────────────────────────────

class OperonDocTemplate(SimpleDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=18*mm,
            rightMargin=18*mm,
            topMargin=18*mm,
            bottomMargin=25*mm,
            title="Operon AI Terminal Cockpit — Full Documentation",
            author="Operon Project",
            subject="AI Terminal Cockpit v3.1.0",
            creator="Operon generate_docs.py",
        )

    def handle_pageBegin(self):
        """Paint the full-page dark background BEFORE any platypus content."""
        super().handle_pageBegin()
        c = self.canv
        c.saveState()

        # Full dark background for every page
        c.setFillColor(BLACK)
        c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

        # ── Cover-page decorative layer (page 1 only) ─────────────────────────
        if getattr(c, '_pageNumber', 0) == 1:
            # Deep-purple halo behind the logo area (top-centre)
            cx = PAGE_W / 2
            cy = PAGE_H - 105 * mm   # approximate vertical centre of the logo

            # Outer soft haze
            c.setFillColor(HexColor("#0D0520"))
            c.circle(cx, cy, 72 * mm, fill=1, stroke=0)
            # Inner reset to background
            c.setFillColor(BLACK)
            c.circle(cx, cy, 60 * mm, fill=1, stroke=0)

            # Concentric ring accents
            for radius, alpha_hex in [
                (68 * mm, "#3A1870"),
                (74 * mm, "#2A0F55"),
                (80 * mm, "#1A0838"),
            ]:
                c.setStrokeColor(HexColor(alpha_hex))
                c.setLineWidth(0.4)
                c.circle(cx, cy, radius, fill=0, stroke=1)

            # Subtle accent line across the page (horizontal divider hint)
            c.setStrokeColor(HexColor("#2A1060"))
            c.setLineWidth(0.5)
            divider_y = PAGE_H - 210 * mm
            c.line(15 * mm, divider_y, PAGE_W - 15 * mm, divider_y)

            # Dark bottom panel for version/meta area
            c.setFillColor(HexColor("#060410"))
            c.rect(0, 0, PAGE_W, 42 * mm, fill=1, stroke=0)

            # Purple top accent bar (3 mm tall)
            c.setFillColor(PURPLE_BASE)
            c.rect(0, PAGE_H - 3 * mm, PAGE_W, 3 * mm, fill=1, stroke=0)
            # Neon highlight on top of the bar
            c.setFillColor(PURPLE_NEON)
            c.rect(0, PAGE_H - 0.8 * mm, PAGE_W, 0.8 * mm, fill=1, stroke=0)

            # Left and right side border accents (thin purple lines)
            c.setStrokeColor(HexColor("#3A1870"))
            c.setLineWidth(1)
            c.line(8 * mm, 42 * mm, 8 * mm, PAGE_H - 12 * mm)
            c.line(PAGE_W - 8 * mm, 42 * mm, PAGE_W - 8 * mm, PAGE_H - 12 * mm)

        c.restoreState()


# ── Styles ────────────────────────────────────────────────────────────────────

def make_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title",
        fontName="Helvetica-Bold",
        fontSize=46,
        textColor=CYAN_GLOW,
        alignment=TA_CENTER,
        spaceAfter=4,
        leading=54,
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub",
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=PURPLE_NEON,
        alignment=TA_CENTER,
        spaceAfter=3,
        leading=18,
    )
    styles["cover_tagline"] = ParagraphStyle(
        "cover_tagline",
        fontName="Helvetica-Oblique",
        fontSize=10,
        textColor=GRAY_TEXT,
        alignment=TA_CENTER,
        spaceAfter=3,
    )
    styles["cover_feat"] = ParagraphStyle(
        "cover_feat",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=GRAY_LIGHT,
        alignment=TA_LEFT,
        spaceAfter=0,
        leading=14,
    )
    styles["cover_version"] = ParagraphStyle(
        "cover_version",
        fontName="Helvetica",
        fontSize=9,
        textColor=GRAY_TEXT,
        alignment=TA_CENTER,
        spaceAfter=0,
        leading=14,
    )

    styles["h1"] = ParagraphStyle(
        "h1",
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=PURPLE_NEON,
        spaceBefore=14,
        spaceAfter=5,
        borderPad=4,
    )
    styles["h2"] = ParagraphStyle(
        "h2",
        fontName="Helvetica-Bold",
        fontSize=15,
        textColor=CYAN_GLOW,
        spaceBefore=10,
        spaceAfter=4,
    )
    styles["h3"] = ParagraphStyle(
        "h3",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=PURPLE_LIGHT,
        spaceBefore=7,
        spaceAfter=3,
    )
    styles["body"] = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=10,
        textColor=WHITE_BRIGHT,
        leading=15,
        spaceAfter=5,
        alignment=TA_JUSTIFY,
    )
    styles["body_dim"] = ParagraphStyle(
        "body_dim",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=GRAY_LIGHT,
        leading=14,
        spaceAfter=4,
        alignment=TA_JUSTIFY,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet",
        fontName="Helvetica",
        fontSize=10,
        textColor=WHITE_BRIGHT,
        leading=14,
        spaceAfter=3,
        leftIndent=14,
        bulletIndent=0,
    )
    styles["code"] = ParagraphStyle(
        "code",
        fontName="Courier",
        fontSize=8.5,
        textColor=CYAN_GLOW,
        backColor=HexColor("#141B2D"),   # deep navy — clearly distinct from #0A0A0F page bg
        leading=13,
        spaceAfter=3,
        leftIndent=8,
        rightIndent=8,
        borderPad=4,
    )
    styles["code_label"] = ParagraphStyle(
        "code_label",
        fontName="Courier-Bold",
        fontSize=9,
        textColor=PURPLE_NEON,
        backColor=HexColor("#1E2D45"),   # lighter navy header band
        leading=13,
        spaceAfter=0,
        leftIndent=8,
        rightIndent=8,
        borderPad=3,
    )
    styles["table_header"] = ParagraphStyle(
        "table_header",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=WHITE_BRIGHT,
    )
    styles["table_cell"] = ParagraphStyle(
        "table_cell",
        fontName="Courier",
        fontSize=8.5,
        textColor=CYAN_GLOW,
        leading=12,
    )
    styles["table_desc"] = ParagraphStyle(
        "table_desc",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=GRAY_LIGHT,
        leading=12,
    )
    styles["note"] = ParagraphStyle(
        "note",
        fontName="Helvetica-Oblique",
        fontSize=9,
        textColor=ORANGE_WARN,
        leading=13,
        spaceAfter=4,
        leftIndent=10,
    )
    styles["toc_entry"] = ParagraphStyle(
        "toc_entry",
        fontName="Helvetica",
        fontSize=10,
        textColor=GRAY_LIGHT,
        leading=16,
    )
    styles["section_label"] = ParagraphStyle(
        "section_label",
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=PURPLE_BASE,
        spaceBefore=12,
        spaceAfter=2,
        letterSpacing=1.5,
    )
    return styles


# ── Helper builders ───────────────────────────────────────────────────────────

S = make_styles()


def P(text, style="body"):
    return Paragraph(text, S[style])


def HR(color=PURPLE_BASE, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=4, spaceBefore=4)


def SP(h=5):
    return Spacer(1, h*mm)


def section_break(title, chapter_num=None):
    """Full-width section divider with a dark background strip."""
    prefix = f"{chapter_num}.  " if chapter_num else ""
    return [
        SP(6),
        HR(PURPLE_NEON, 1.5),
        P(f"<font color='#{PURPLE_NEON.hexval()[2:]}'>{'─' * 3}</font>  "
          f"<b>{prefix}{title}</b>", "h1"),
        HR(PURPLE_BASE, 0.4),
    ]


def sub_section(title):
    return [SP(2), P(title, "h2"), HR(GRAY_DARK, 0.3)]


def code_block(lines, label=None):
    """A monospaced code block with optional label."""
    items = []
    if label:
        items.append(P(f"  {label}", "code_label"))
    for line in lines:
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        items.append(P(f"  {safe}", "code"))
    return items


def info_box(text, color=CYAN_GLOW):
    col_hex = color.hexval()[2:]
    return [
        SP(2),
        P(f'<font color="#{col_hex}">ℹ  </font><i>{text}</i>', "body_dim"),
        SP(1),
    ]


def bullet_list(items, symbol="◆"):
    return [P(f'<font color="#{PURPLE_NEON.hexval()[2:]}">{symbol}</font>  {item}', "bullet")
            for item in items]


def two_col_table(rows, col_widths=None, header=None):
    """Generic 2-column table with Operon dark theme."""
    if col_widths is None:
        col_widths = [70*mm, PAGE_W - 36*mm - 70*mm]
    data = []
    if header:
        data.append([P(header[0], "table_header"), P(header[1], "table_header")])
    for row in rows:
        data.append([
            P(str(row[0]), "table_cell"),
            P(str(row[1]), "table_desc"),
        ])
    style = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0 if header else -1), GRAY_DARK),
        ("BACKGROUND",  (0, 1 if header else 0), (-1, -1), GRAY_ROW),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_ROW, GRAY_DARK]),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE_BRIGHT),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("GRID",        (0, 0), (-1, -1), 0.3, GRAY_MID),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
    ])
    return Table(data, colWidths=col_widths, style=style, repeatRows=1 if header else 0)


def three_col_table(rows, col_widths=None, header=None):
    if col_widths is None:
        w = PAGE_W - 36*mm
        col_widths = [w*0.25, w*0.35, w*0.40]
    data = []
    if header:
        data.append([P(h, "table_header") for h in header])
    for row in rows:
        data.append([P(str(c), "table_cell") for c in row])
    style = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), GRAY_MID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_ROW, GRAY_DARK]),
        ("GRID",        (0, 0), (-1, -1), 0.3, GRAY_MID),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE_BRIGHT),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
    ])
    return Table(data, colWidths=col_widths, style=style, repeatRows=1 if header else 0)


# ── Cover page ────────────────────────────────────────────────────────────────

def build_cover():
    elems = []
    # Background + decorative canvas layer painted by handle_pageBegin.
    # This function only adds platypus flowable content.

    # ── Top spacer (canvas top-bar is 3 mm; give breathing room) ─────────────
    elems.append(SP(6))

    # ── Hero logo ─────────────────────────────────────────────────────────────
    logo_size = 88 * mm
    if LOGO_PATH.exists():
        try:
            img = Image(str(LOGO_PATH), width=logo_size, height=logo_size)
            img.hAlign = "CENTER"
            elems.append(img)
        except Exception:
            elems.append(P('<font color="#C084FC">▲</font>', "cover_title"))
    else:
        elems.append(P('<font color="#C084FC">▲</font>', "cover_title"))

    elems.append(SP(4))

    # ── Product name ─────────────────────────────────────────────────────────
    elems.append(P("OPERON", "cover_title"))
    elems.append(SP(1))
    elems.append(P("A I   T E R M I N A L   C O C K P I T", "cover_sub"))
    elems.append(SP(2))
    elems.append(P(
        "Hermes Planner  ×  Open-Claw Tools  ×  Zero Amnesia",
        "cover_tagline"))

    elems.append(SP(5))
    elems.append(HR(PURPLE_NEON, 1.5))
    elems.append(SP(4))

    # ── Stats strip (2-row table: big number + small label) ───────────────────
    w4 = (PAGE_W - 36 * mm) / 4
    stats_data = [
        ["185+",  "8+",            "60+",      "∞"],
        ["TOOLS", "AI PROVIDERS",  "COMMANDS", "MCP TOOLS"],
    ]
    stats_style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#0D0520")),
        ("BOX",           (0, 0), (-1, -1), 1,   PURPLE_BASE),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, PURPLE_BASE),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        # Numbers row
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 28),
        ("TEXTCOLOR",     (0, 0), (-1, 0), CYAN_GLOW),
        ("TOPPADDING",    (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        # Labels row
        ("FONTNAME",      (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 1), (-1, 1), 7.5),
        ("TEXTCOLOR",     (0, 1), (-1, 1), GRAY_TEXT),
        ("TOPPADDING",    (0, 1), (-1, 1), 2),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
    ])
    elems.append(Table(stats_data, colWidths=[w4] * 4, style=stats_style))

    elems.append(SP(4))
    elems.append(HR(PURPLE_BASE, 0.5))
    elems.append(SP(4))

    # ── Feature highlights (3×3 grid) ─────────────────────────────────────────
    _dot = f'<font color="#{PURPLE_NEON.hexval()[2:]}">◆</font>'

    def _feat(text):
        return P(f"{_dot}  {text}", "cover_feat")

    feat_data = [
        [_feat("Semantic Vector Memory"),   _feat("Smart Model Router"),        _feat("Live Web Dashboard")],
        [_feat("Autonomous Skill Curator"), _feat("MCP Server Integration"),    _feat("Desktop Automation")],
        [_feat("SWE Agent Loop"),           _feat("Voice + Multimodal"),        _feat("Claude Code-style TUI")],
        [_feat("Obsidian Vault Sync"),      _feat("Multi-Agent Mesh"),          _feat("macOS .app + EXE")],
        [_feat("Plugin Marketplace"),       _feat("Headless Browser (11 cmds)"), _feat("DALL-E 3 + TTS")],
    ]
    fw = (PAGE_W - 36 * mm) / 3
    feat_style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), GRAY_DARK),
        ("BOX",           (0, 0), (-1, -1), 0.5, GRAY_MID),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, GRAY_MID),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ])
    elems.append(Table(feat_data, colWidths=[fw] * 3, style=feat_style, rowHeights=[None]*5))

    # ── Bottom meta (sits in the dark panel drawn by handle_pageBegin) ────────
    elems.append(SP(5))
    elems.append(HR(HexColor("#2A1060"), 0.5))
    elems.append(SP(3))
    now = datetime.now().strftime("%B %Y")
    elems.append(P(
        f'Version 3.1.0  •  {now}  •  Full Technical Reference',
        "cover_version"))
    elems.append(SP(1))
    elems.append(P(
        'Phase 11 Build  •  185+ Tools  •  60+ Commands  •  1,896 Tests  •  © 2026 Kunwar Mann',
        "cover_version"))

    elems.append(PageBreak())
    return elems


# ── Table of Contents ─────────────────────────────────────────────────────────

def build_toc():
    elems = []
    elems += section_break("Table of Contents")
    elems.append(SP(3))

    toc_items = [
        ("1",  "Introduction &amp; Architecture Overview", "4"),
        ("2",  "Installation &amp; Quick Start", "6"),
        ("3",  "Configuration &amp; Providers", "9"),
        ("4",  "Core Architecture — 39+ Modules", "12"),
        ("5",  "Complete Tool Reference — 185+ Tools", "16"),
        ("6",  "All 60+ Slash Commands", "26"),
        ("7",  "Memory System — FTS5 + Vector + Obsidian", "32"),
        ("8",  "SKILL.md &amp; Curator / Skill Synthesizer", "34"),
        ("9",  "MCP Server Integration", "37"),
        ("10", "Web Dashboard", "40"),
        ("11", "SSH Remote Execution", "42"),
        ("12", "Telegram Gateway &amp; Messaging", "44"),
        ("13", "Browser Automation &amp; Computer Use", "46"),
        ("14", "SWE Agent — Automated Software Engineering", "49"),
        ("15", "Voice Pipeline &amp; Multimodal", "51"),
        ("16", "Multi-Agent Mesh &amp; Delegation", "53"),
        ("17", "Kanban Board &amp; Task Management", "55"),
        ("18", "Vision, Image Generation &amp; TTS", "57"),
        ("19", "Docker &amp; Packaging (.app/.exe)", "59"),
        ("20", "Security &amp; Best Practices", "62"),
        ("21", "Comparison: Operon vs Hermes vs OpenClaw vs OpenHuman", "64"),
        ("22", "Troubleshooting &amp; FAQ", "66"),
    ]

    for num, title, page in toc_items:
        dots = "." * max(5, 58 - len(title) - len(num) - len(page))
        elems.append(
            P(f'<font color="#{CYAN_GLOW.hexval()[2:]}">{num}.</font>  '
              f'{title}  '
              f'<font color="#{GRAY_TEXT.hexval()[2:]}">{dots} {page}</font>',
              "toc_entry")
        )

    elems.append(PageBreak())
    return elems


# ── Section 1: Introduction ───────────────────────────────────────────────────

def build_intro():
    elems = []
    elems += section_break("Introduction &amp; Architecture Overview", 1)

    elems.append(P(
        "Operon v3.1.0 is an advanced AI Terminal Cockpit — a fully agentic Python REPL "
        "supporting <b>185+ tools</b>, <b>8+ AI providers</b>, <b>60+ slash commands</b>, "
        "and a deeply integrated Phase 11 feature set spanning semantic vector memory "
        "(LanceDB), Obsidian vault sync, smart per-turn model routing, "
        "a self-improvement skill synthesizer, full desktop computer use automation, "
        "a Claude Code-style TUI, SWE agent loop, voice pipeline, multi-agent mesh, "
        "and a plugin marketplace. Built and verified with <b>1,896 passing tests</b>.", "body"))

    elems.append(P(
        "Every response follows a strict <b>JSON-first scratchpad pattern</b>: Operon "
        "always thinks before acting, records its objective and workspace variables, "
        "drafts code in-flight, then invokes tools or delivers a final answer — "
        "all in a single, machine-parsable JSON envelope. This makes Operon "
        "deterministic, debuggable, and far more reliable than chat-first agents.", "body"))

    elems += sub_section("High-Level Architecture")
    elems.append(SP(2))

    arch_rows = [
        ("main.py", "REPL entry point, slash command router, agent loop orchestrator"),
        ("core/config.py", "ConfigManager — ~/.operon/config.json, API keys, profiles"),
        ("core/router.py", "ModelRouter — 6 providers, retry, 7-pass JSON repair, context truncation"),
        ("core/memory.py", "MemoryPipeline — SQLite FTS5, auto-inject, /memory commands"),
        ("core/session.py", "SessionManager — snapshots, rollback, compress, search"),
        ("core/planner.py", "HermesPlannerRenderer — structured scratchpad format injection"),
        ("core/skills.py", "SkillLoader — SKILL.md packs from ~/.operon/skills/"),
        ("core/curator.py", "Curator — autonomous background skill generation"),
        ("core/mcp.py", "MCPManager — JSON-RPC 2.0 stdio + HTTP MCP client"),
        ("core/dashboard.py", "DashboardServer — live localhost:7270 web UI"),
        ("core/gateway.py", "TelegramGateway — bot polling → agent sub-sessions"),
        ("core/scheduler.py", "TaskScheduler — cron-like background task queue"),
        ("core/soul.py", "SoulSystem — personality / persona injection"),
        ("tools/registry.py", "ToolRegistry — 40 tool definitions + dynamic dispatch"),
        ("tools/file_ops.py", "File system tools (read/write/append/patch/delete/list)"),
        ("tools/shell_exec.py", "Shell command execution with timeout"),
        ("tools/web_search.py", "DuckDuckGo + web scrape + X/Twitter search"),
        ("tools/code_exec.py", "Python sandbox execution"),
        ("tools/http_client.py", "Full HTTP client (GET/POST/PUT/PATCH/DELETE)"),
        ("tools/browser.py", "Playwright headless browser (11 tools)"),
        ("tools/vision.py", "Vision analysis + DALL-E 3 image gen + TTS"),
        ("tools/ssh_exec.py", "SSH exec/upload/download with paramiko + subprocess fallback"),
        ("tools/messaging.py", "Telegram send, clarify, todo tools"),
        ("tools/email_send.py", "SMTP email with attachments (Gmail/Outlook/Yahoo/iCloud)"),
        ("tools/file_search.py", "Recursive regex file content search"),
        ("ui/banner.py", "ASCII banner with telemetry on startup"),
        ("ui/theme.py", "ANSI colour theme, spinner, box formatting"),
    ]

    elems.append(two_col_table(arch_rows,
        col_widths=[62*mm, PAGE_W - 36*mm - 62*mm],
        header=["Module / File", "Responsibility"]))

    elems += sub_section("Data Flow")
    elems.append(P(
        "User input → <b>Slash-command check</b> (returns immediately) → "
        "<b>build_system_prompt()</b> assembles tools + memory + soul + skills + context → "
        "<b>ModelRouter.chat()</b> (retry + JSON repair) → "
        "<b>parse_response()</b> → <b>tool dispatch</b> (ToolRegistry) → "
        "<b>log_tool_call()</b> (dashboard) → <b>Curator.maybe_curate()</b> → "
        "loop back or return final answer to user.", "body"))

    elems.append(PageBreak())
    return elems


# ── Section 2: Installation ───────────────────────────────────────────────────

def build_installation():
    elems = []
    elems += section_break("Installation &amp; Quick Start", 2)

    elems += sub_section("Prerequisites")
    elems += bullet_list([
        "Python 3.9 or later (3.11+ recommended)",
        "git (for cloning)",
        "macOS, Linux, or Windows",
        "<i>Optional:</i> Ollama for fully-offline local models (no API key needed)",
        "<i>Optional:</i> Docker + Docker Compose (for containerised deployment)",
    ])

    elems += sub_section("Step 1 — One-Command Install (recommended)")
    elems.append(P(
        "The installer handles <b>everything</b> — Python dependencies <b>and</b> the "
        "Playwright Chromium browser binary (the ~120 MB download that a plain "
        "<code>pip install</code> skips). It creates a <code>.venv</code>, installs "
        "Operon, and registers the <code>operon</code> command.", "body"))
    elems += code_block([
        "git clone https://github.com/OperonAgent/Operon.git",
        "cd operon",
        "",
        "# macOS / Linux:",
        "./install.sh",
        "",
        "# Windows:",
        "powershell -ExecutionPolicy Bypass -File install.ps1",
        "",
        "# Any platform (Python):",
        "python install.py            # core + recommended + browser binary",
        "python install.py --full     # also voice, databases, screen capture",
    ], label="Shell")
    elems.append(P(
        "<b>Why the browser is a separate step:</b> <code>pip install playwright</code> "
        "installs only the Python package. The actual Chromium browser binary needs a "
        "separate <code>playwright install chromium</code>. Operon's installer — and a "
        "runtime self-heal hook — do this for you. If you ever browse without the binary, "
        "Operon downloads it on demand.", "body"))

    elems += sub_section("Step 1b — Manual install (alternative)")
    elems += code_block([
        "pip install -r requirements.txt",
        "python -m core.bootstrap --browser   # downloads Chromium (~120 MB)",
        "",
        "# Provision helpers:",
        "python -m core.bootstrap            # core + recommended + browser",
        "python -m core.bootstrap --full     # every optional feature",
        "python -m core.bootstrap --check    # status only, install nothing",
    ], label="Shell")

    elems += sub_section("Step 2 — First Launch")
    elems += code_block(["python main.py"], label="Shell")
    elems.append(P(
        "On first launch the ASCII banner displays, then the <b>8-step setup wizard</b> "
        "runs automatically. You can skip fields with Enter and configure later via "
        "<code>/setup</code>.", "body"))

    elems += sub_section("Step 3 — Setup Wizard Fields")
    wizard_rows = [
        ("1", "Default AI provider", "openai / anthropic / openrouter / ollama / lmstudio / jan"),
        ("2", "Default model ID", "gpt-4o, claude-3-5-sonnet-20241022, openrouter/…"),
        ("3", "OpenAI API key", "sk-… (or press Enter to skip)"),
        ("4", "Anthropic API key", "sk-ant-… (or press Enter to skip)"),
        ("5", "OpenRouter key", "sk-or-… (or press Enter to skip)"),
        ("6", "Telegram bot token", "1234567890:ABC… (or skip — start gateway later)"),
        ("7", "Telegram chat ID", "Your chat_id for /status notifications"),
        ("8", "Memory enabled", "yes (recommended) — SQLite FTS5 persistent memory"),
    ]
    elems.append(three_col_table(wizard_rows, header=["#", "Field", "Value / Notes"]))

    elems += sub_section("Step 4 — Verify Installation")
    elems += code_block([
        "operon --check-deps     # dependency + browser-binary status",
        "operon                  # launch, then type:",
        "/doctor                 # full in-app health check",
    ], label="Shell / Operon REPL")
    elems.append(P(
        "<b>operon --check-deps</b> reports the status of every package plus the Chromium "
        "browser binary. The <b>/doctor</b> command (inside Operon) checks API keys, local "
        "model servers, tool count, memory status, and optional services "
        "(dashboard, MCP, Curator, gateway).", "body"))

    elems += sub_section("Dependencies Reference")
    dep_rows = [
        ("requests>=2.31.0", "Required", "All HTTP, provider API calls, web search"),
        ("beautifulsoup4>=4.12.0", "Required", "Web scraping, X/Twitter Nitter parsing"),
        ("psutil>=5.9.8", "Recommended", "CPU/RAM telemetry in banner and /status"),
        ("pypdf>=4.0.0", "Optional", "PDF reading tool"),
        ("reportlab>=4.0.0", "Optional", "PDF generation tool"),
        ("playwright>=1.40.0", "Optional", "Headless browser automation (11 tools)"),
        ("paramiko>=3.4.0", "Optional", "SSH exec/upload/download (falls back to system ssh)"),
        ("openai>=1.30.0", "Optional", "Direct OpenAI SDK (auto-fallback to requests)"),
        ("anthropic>=0.25.0", "Optional", "Direct Anthropic SDK (auto-fallback to requests)"),
    ]
    elems.append(three_col_table(dep_rows, header=["Package", "Status", "Used For"]))

    elems.append(PageBreak())
    return elems


# ── Section 3: Configuration ──────────────────────────────────────────────────

def build_config():
    elems = []
    elems += section_break("Configuration &amp; Providers", 3)

    elems.append(P(
        "All configuration is stored in <code>~/.operon/config.json</code> and managed "
        "through the <b>/setup</b> wizard or direct <b>/config</b> commands. "
        "API keys are stored there; for sensitive environments use environment variables "
        "instead — Operon reads them at startup.", "body"))

    elems += sub_section("Supported AI Providers")
    prov_rows = [
        ("openai", "OpenAI", "GPT-4o, GPT-4.1, o3, o4-mini", "OPENAI_API_KEY", "https://api.openai.com/v1"),
        ("anthropic", "Anthropic", "Claude 3.5 Sonnet, Haiku, Opus", "ANTHROPIC_API_KEY", "https://api.anthropic.com/v1"),
        ("openrouter", "OpenRouter", "100+ models via single API", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
        ("ollama", "Ollama (local)", "llama3, mistral, codestral, …", "—", "http://localhost:11434"),
        ("lmstudio", "LM Studio (local)", "Any GGUF model", "—", "http://localhost:1234"),
        ("jan", "Jan (local)", "Any GGUF model", "—", "http://localhost:1337"),
    ]
    w = PAGE_W - 36*mm
    provider_table = Table(
        [[P(h, "table_header") for h in ["Key", "Name", "Notable Models", "Env Var", "Base URL"]]] +
        [[P(str(c), "table_cell") for c in row] for row in prov_rows],
        colWidths=[w*0.11, w*0.14, w*0.25, w*0.20, w*0.30],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GRAY_MID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_ROW, GRAY_DARK]),
            ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE_BRIGHT),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ]),
        repeatRows=1
    )
    elems.append(provider_table)

    elems += sub_section("Model Profiles")
    elems.append(P(
        "Operon supports <b>named model profiles</b> — you can save multiple provider/model "
        "combinations and switch between them instantly. Profiles are stored under "
        "<code>model_profiles</code> in config.json.", "body"))
    elems += code_block([
        "/model gpt-4o                         # switch to GPT-4o (openai)",
        "/model claude-3-5-sonnet-20241022      # switch to Claude",
        "/model openrouter/meta-llama/llama-3   # OpenRouter model",
        "/local use llama3                      # local Ollama model",
        "/local use lmstudio:gemma-2-9b-it      # LM Studio model",
    ], label="Operon REPL")

    elems += sub_section("Environment Variables")
    elems.append(P(
        "For production or Docker environments, set credentials as environment variables "
        "rather than storing them in config.json:", "body"))
    env_rows = [
        ("OPENAI_API_KEY", "OpenAI API key"),
        ("ANTHROPIC_API_KEY", "Anthropic API key"),
        ("OPENROUTER_API_KEY", "OpenRouter key"),
        ("TELEGRAM_BOT_TOKEN", "Telegram bot token"),
        ("TELEGRAM_CHAT_ID", "Telegram chat ID for notifications"),
        ("OLLAMA_BASE_URL", "Custom Ollama server URL (default localhost:11434)"),
        ("OPERON_DATA_DIR", "Override ~/.operon data directory"),
    ]
    elems.append(two_col_table(env_rows, header=["Variable", "Purpose"]))

    elems += sub_section("Retry &amp; Resilience")
    elems += bullet_list([
        "<b>Exponential backoff:</b> up to 3 retries, base delay 1.5s (doubles each attempt)",
        "<b>Retry-After header:</b> honored for 429 rate-limit responses",
        "<b>7-pass JSON repair:</b> balanced-brace walk, fence strip, trailing comma fix, Python literal fix, single-quote fix, object-prefix strip, forced wrap",
        "<b>Context auto-truncation:</b> fires before each API call when history exceeds 120 messages",
        "<b>Provider fallback:</b> configure a secondary provider via /setup",
    ])

    elems.append(PageBreak())
    return elems


# ── Section 4: Core Architecture ─────────────────────────────────────────────

def build_architecture():
    elems = []
    elems += section_break("Core Architecture — 39+ Modules", 4)

    modules = [
        {
            "name": "core/config.py — ConfigManager",
            "desc": (
                "Manages all persistent settings in <code>~/.operon/config.json</code>. "
                "Provides typed getters for API keys, model profiles, feature flags, "
                "and provider URLs. Config is write-through: every <code>set()</code> call "
                "immediately flushes to disk. Supports default values and atomic writes."
            ),
            "features": [
                "get_api_key(provider) — returns key from config or env var",
                "is_configured() — True only when default_model is set",
                "get_model_profiles() — dict of named provider/model combos",
                "Atomic JSON write (write-to-temp + rename) prevents corruption",
            ]
        },
        {
            "name": "core/router.py — ModelRouter",
            "desc": (
                "The unified LLM interface. Translates a list of messages into a provider-"
                "specific API call using pure <code>requests</code> (no SDK required). "
                "Handles streaming, non-streaming, and vision-capable models. "
                "Returns the raw assistant message string."
            ),
            "features": [
                "6 providers: openai, anthropic, openrouter, ollama, lmstudio, jan",
                "Automatic retry with exponential backoff (3 attempts, 1.5s base)",
                "7-pass JSON repair for malformed LLM responses",
                "parse_response() — extracts action dict from any JSON envelope",
                "Context auto-truncation via maybe_truncate(hard_limit=120)",
                "Vision support: base64 image encoding for multimodal calls",
            ]
        },
        {
            "name": "core/memory.py — MemoryPipeline",
            "desc": (
                "Zero-amnesia persistent memory using SQLite FTS5. Facts, preferences, "
                "and important information are automatically extracted from conversation "
                "and stored with timestamps. Relevant memories are injected into the "
                "system prompt at the start of every turn."
            ),
            "features": [
                "SQLite FTS5 virtual table with content=memories and auto-sync triggers",
                "Ranked relevance search via rank column (BM25)",
                "Auto-extraction: agent calls memory_add during conversations",
                "get_context_string() — top-N memories injected as system block",
                "Backfill on DB upgrade (FTS5 table rebuilt from existing rows)",
                "/memory add|search|list|delete|clear commands",
            ]
        },
        {
            "name": "core/session.py — SessionManager",
            "desc": (
                "Full conversation history management with named snapshots. Sessions "
                "are stored as JSON in <code>~/.operon/sessions/</code>. Supports "
                "rollback to any saved checkpoint, compression of old turns, "
                "and full-text search within session history."
            ),
            "features": [
                "save(name) — snapshot current messages to a named file",
                "load(name) — restore a previous session",
                "list() — enumerate saved sessions with metadata",
                "compress(keep_last=N) — reduce context by summarising old turns",
                "search(query) — find turns matching a string in current session",
                "rollback() — restore the most recent auto-snapshot",
            ]
        },
        {
            "name": "core/planner.py — HermesPlannerRenderer",
            "desc": (
                "Injects the Hermes-style structured scratchpad format into the system "
                "prompt. This forces the model to always reason before acting: fill out "
                "<code>objective</code>, <code>workspace_vars</code>, <code>code_draft</code>, "
                "and <code>next_step</code> before issuing a tool call or response. "
                "Dramatically improves multi-step task reliability."
            ),
            "features": [
                "JSON schema enforcement — all responses must match the scratchpad format",
                "Two action types: 'tool' (call a tool) and 'response' (final answer)",
                "CRITICAL rules injected to prevent common failure modes",
                "Works with all 6 providers — pure prompt engineering, no SDK hooks",
            ]
        },
        {
            "name": "core/skills.py — SkillLoader",
            "desc": (
                "Loads SKILL.md instruction packs from <code>~/.operon/skills/</code>. "
                "Each <code>.md</code> file is a domain-specific system-prompt extension "
                "that gives Operon specialised knowledge (e.g. a company's internal APIs, "
                "a coding style guide, domain terminology). Skills are injected as an "
                "additional system block on every turn."
            ),
            "features": [
                "Hot-reload — new skills are picked up on the next turn after /skills reload",
                "auto__*.md prefix — files created by Curator for auto-generated skills",
                "as_system_block() — returns all skills merged into one prompt block",
                "len(skills) — number of loaded skills for /doctor display",
            ]
        },
        {
            "name": "core/gateway.py — TelegramGateway",
            "desc": (
                "Connects Operon to Telegram as a fully-functional bot. When started, "
                "it polls the Telegram API for new messages and routes each one through "
                "a fresh sub-session of the agent loop. Replies are sent back to the "
                "chat. Supports concurrent multi-user sessions."
            ),
            "features": [
                "Long-polling via Telegram Bot API (no webhook setup required)",
                "Per-message sub-sessions — each user message gets a full agent loop",
                "Graceful /gateway start|stop|status commands",
                "Supports all Operon tools from Telegram (file ops, web, code, etc.)",
                "Token configurable via /setup or TELEGRAM_BOT_TOKEN env var",
            ]
        },
        {
            "name": "core/scheduler.py — TaskScheduler",
            "desc": (
                "A cron-like background task queue. Register recurring or one-shot tasks "
                "with intervals in seconds. Used internally for memory cleanup, session "
                "auto-save, and any user-defined periodic automation."
            ),
            "features": [
                "Thread-safe task registry with named entries",
                "add_task(name, fn, interval_seconds) — schedule any callable",
                "cancel(name) — remove a scheduled task",
                "Non-blocking — all tasks run in daemon threads",
            ]
        },
        {
            "name": "core/soul.py — SoulSystem",
            "desc": (
                "Optional persona and personality injection. Load a 'soul file' "
                "(<code>.md</code>) to give Operon a custom persona, communication style, "
                "domain expertise framing, or compliance constraints. The soul block is "
                "prepended to the system prompt on every turn."
            ),
            "features": [
                "load_from_file(path) — read a soul definition from any .md file",
                "as_system_block() — formatted system prompt prefix",
                "/soul load|unload|status commands",
                "Stacks with skills — soul comes first in the system prompt",
            ]
        },
    ]

    for mod in modules:
        elems.append(SP(3))
        elems.append(P(mod["name"], "h3"))
        elems.append(P(mod["desc"], "body"))
        elems += bullet_list(mod["features"], "→")
        elems.append(HR(GRAY_DARK, 0.3))

    elems.append(PageBreak())
    return elems


# ── Section 5: Tool Reference ─────────────────────────────────────────────────

def build_tools():
    elems = []
    elems += section_break("Complete Tool Reference — 185+ Tools", 5)

    elems.append(P(
        "Operon exposes <b>40 built-in tools</b> to the agent, plus any tools dynamically "
        "registered via MCP servers. Every tool returns a normalised dict with at minimum "
        "<code>success</code>, <code>output</code>, and <code>error</code> keys.",
        "body"))

    # Category groups
    categories = [
        {
            "title": "File System Tools (8)",
            "tools": [
                ("file_read", "path, encoding?", "Read full file contents as string"),
                ("file_write", "path, content", "Create or overwrite a file; auto-creates parents"),
                ("file_append", "path, content", "Append text to a file (creates if needed)"),
                ("file_patch", "path, old_text, new_text", "Find-and-replace first occurrence in file"),
                ("file_delete", "path", "Delete file or directory (recursive)"),
                ("dir_list", "path?, max_depth?", "ASCII tree listing of directory"),
                ("file_exists", "path", "Check if file/dir exists → bool"),
                ("file_info", "path", "Metadata: size, mtime, permissions"),
            ]
        },
        {
            "title": "Shell &amp; Code Execution (2)",
            "tools": [
                ("shell_exec", "command, cwd?, timeout?", "Run bash command; returns stdout, stderr, exit code"),
                ("python_exec", "code, timeout?, cwd?", "Execute Python in subprocess sandbox"),
            ]
        },
        {
            "title": "Web &amp; Search Tools (3)",
            "tools": [
                ("duckduckgo_search", "query, max_results?", "Web search — titles, URLs, snippets; no API key"),
                ("web_scrape", "url, max_chars?", "Fetch URL → readable text via BeautifulSoup"),
                ("x_search", "query, max_results?", "X/Twitter search via Nitter → DuckDuckGo fallback"),
            ]
        },
        {
            "title": "HTTP Client (1)",
            "tools": [
                ("http_request", "url, method?, headers?, body?, params?, bearer_token?, timeout?",
                 "Full HTTP client: GET/POST/PUT/PATCH/DELETE with JSON body and Bearer auth"),
            ]
        },
        {
            "title": "File Search (1)",
            "tools": [
                ("file_search", "pattern, path?, recursive?, case_sensitive?, file_pattern?, max_results?, context_lines?",
                 "Recursive regex/plaintext content search; returns file:line:match"),
            ]
        },
        {
            "title": "Email (2)",
            "tools": [
                ("email_draft", "to, subject, body, sender_email?, app_password?",
                 "Compose a draft, show formatted preview, wait for user approval, then send. "
                 "Credentials auto-loaded from GMAIL_SENDER_EMAIL/GMAIL_APP_PASSWORD env vars "
                 "or knowledge base. Returns {approved, sent, feedback} — redraft if rejected."),
                ("email_send", "sender_email, app_password, to, subject, body, body_type?, cc?, bcc?, reply_to?, attachments?",
                 "Silent SMTP send (no preview). Use email_draft instead for user-requested emails."),
            ]
        },
        {
            "title": "Browser Automation — Playwright (11)",
            "tools": [
                ("browser_navigate", "url, wait_until?, timeout?", "Navigate headless browser to URL"),
                ("browser_snapshot", "max_chars?", "Get visible text + top links from current page"),
                ("browser_screenshot", "save_path?, full_page?", "Save PNG screenshot of current page"),
                ("browser_click", "selector, timeout?", "Click CSS selector or text='…' locator"),
                ("browser_type", "selector, text, clear_first?", "Type into input field"),
                ("browser_scroll", "direction?, amount?", "Scroll page up or down"),
                ("browser_eval", "javascript", "Execute JS on current page; returns result"),
                ("browser_back", "—", "Navigate back in browser history"),
                ("browser_get_url", "—", "Return current URL and page title"),
                ("browser_wait", "milliseconds?", "Wait N ms (useful after dynamic loads)"),
                ("browser_close", "—", "Close browser and release all resources"),
            ]
        },
        {
            "title": "Vision, Image &amp; TTS (3)",
            "tools": [
                ("vision_analyze", "image_path? or image_url?, prompt?, provider?",
                 "Analyze image with GPT-4o or Claude vision; supports local file or URL"),
                ("image_generate", "prompt, size?, quality?, save_path?",
                 "Generate image with DALL-E 3; saves to ~/Desktop/operon_img_<ts>.png"),
                ("tts_speak", "text, voice?, save_path?, play?",
                 "Text-to-speech via OpenAI TTS or macOS say fallback; 6 voices"),
            ]
        },
        {
            "title": "Messaging &amp; Interaction (3)",
            "tools": [
                ("telegram_send", "chat_id, text, parse_mode?", "Send Telegram message via configured bot"),
                ("clarify", "question", "Ask user a blocking question; returns typed answer"),
                ("todo", "action, item?, index?", "Session task list: add/list/complete/remove/clear"),
            ]
        },
        {
            "title": "SSH Remote Execution (3)",
            "tools": [
                ("ssh_exec", "host, command, port?, user?, password?, key_path?, timeout?, cwd?",
                 "Execute command on remote host; paramiko primary, system ssh fallback"),
                ("ssh_upload", "host, local_path, remote_path, port?, user?, password?, key_path?",
                 "Upload local file to remote via SFTP/SCP"),
                ("ssh_download", "host, remote_path, local_path, port?, user?, password?, key_path?",
                 "Download remote file to local via SFTP/SCP"),
            ]
        },
        {
            "title": "Agent Coordination (1)",
            "tools": [
                ("sub_agent", "prompt", "Spawn a focused sub-agent; returns last assistant message"),
            ]
        },
        {
            "title": "MCP Dynamic Tools (∞)",
            "tools": [
                ("mcp__<server>__<tool>", "per tool schema", "Dynamically registered MCP tools (see /mcp tools)"),
            ]
        },
    ]

    for cat in categories:
        elems.append(SP(2))
        elems.append(P(cat["title"], "h3"))
        rows = [(t[0], t[1], t[2]) for t in cat["tools"]]
        w = PAGE_W - 36*mm
        tbl = Table(
            [[P(h, "table_header") for h in ["Tool Name", "Parameters", "Description"]]] +
            [[P(str(c), "table_cell") if i < 2 else P(str(c), "table_desc")
              for i, c in enumerate(row)] for row in rows],
            colWidths=[w*0.22, w*0.30, w*0.48],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), GRAY_MID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_ROW, GRAY_DARK]),
                ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE_BRIGHT),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
            ]),
            repeatRows=1
        )
        elems.append(tbl)

    elems.append(PageBreak())
    return elems


# ── Section 6: Slash Commands ─────────────────────────────────────────────────

def build_commands():
    elems = []
    elems += section_break("All 60+ Slash Commands", 6)

    elems.append(P(
        "All slash commands are processed before the agent loop and return immediately "
        "without consuming any LLM tokens. Type <b>/help</b> to see the full list at any time.",
        "body"))

    command_groups = [
        {
            "title": "System &amp; Configuration",
            "commands": [
                ("/help", "Show all commands with short descriptions"),
                ("/setup", "Run the interactive 8-step configuration wizard"),
                ("/status", "Show current model, memory count, gateway, dashboard, MCP status"),
                ("/doctor", "Full system health check — keys, deps, local servers, all components"),
                ("/config get <key>", "Read a config value by key"),
                ("/config set <key> <value>", "Write a config value"),
                ("/config show", "Dump the full config.json"),
            ]
        },
        {
            "title": "Model &amp; Provider",
            "commands": [
                ("/model <name>", "Switch the active model/provider (takes effect immediately)"),
                ("/local", "List all running local model servers (Ollama, LM Studio, Jan)"),
                ("/local use <model>", "Switch to a local model"),
                ("/local url <provider> <url>", "Override a local server's base URL"),
                ("/approve on|off", "Enable/disable tool approval mode (ask before every tool call)"),
            ]
        },
        {
            "title": "Session Management",
            "commands": [
                ("/session save <name>", "Save current conversation to a named snapshot"),
                ("/session load <name>", "Restore a previously saved session"),
                ("/session list", "List all saved sessions with timestamps"),
                ("/session compress [N]", "Compress session — keep last N turns (default 10)"),
                ("/session rollback", "Restore the most recent auto-snapshot"),
                ("/session search <query>", "Find turns in current session matching query"),
                ("/clear", "Clear the current session (start fresh)"),
            ]
        },
        {
            "title": "Memory",
            "commands": [
                ("/memory list [N]", "Show the last N memory items (default 20)"),
                ("/memory search <query>", "FTS5 ranked search of memory"),
                ("/memory add <text>", "Manually add a memory item"),
                ("/memory delete <id>", "Delete a memory item by ID"),
                ("/memory clear", "Wipe all memories (irreversible)"),
                ("/forget", "Shortcut: clear all memories immediately"),
            ]
        },
        {
            "title": "Skills",
            "commands": [
                ("/skills list", "List all loaded SKILL.md files"),
                ("/skills reload", "Hot-reload all skills from ~/.operon/skills/"),
                ("/skills show <name>", "Display the content of a skill file"),
            ]
        },
        {
            "title": "Telegram Gateway",
            "commands": [
                ("/gateway start", "Start the Telegram polling bot"),
                ("/gateway stop", "Stop the Telegram bot"),
                ("/gateway status", "Show gateway running state and message count"),
            ]
        },
        {
            "title": "Web Dashboard",
            "commands": [
                ("/dashboard start", "Start the web dashboard on localhost:7270"),
                ("/dashboard stop", "Stop the dashboard server"),
                ("/dashboard open", "Open dashboard in the default browser"),
                ("/dashboard status", "Show whether dashboard is running and URL"),
            ]
        },
        {
            "title": "MCP Server Management",
            "commands": [
                ("/mcp list", "List all connected MCP servers and tool counts"),
                ("/mcp tools [server]", "Show all tools from all (or one) MCP server"),
                ("/mcp connect <name> <transport> [args]", "Connect a new MCP server (stdio or http)"),
                ("/mcp disconnect <name>", "Disconnect and remove an MCP server"),
            ]
        },
        {
            "title": "Curator &amp; Soul",
            "commands": [
                ("/curator on|off", "Enable or disable autonomous skill generation"),
                ("/curator status", "Show Curator state: enabled, cooldown, auto-skill count"),
                ("/curator clear", "Delete all auto-generated skill files"),
                ("/soul load <path>", "Load a soul/persona file"),
                ("/soul unload", "Remove the active soul"),
                ("/soul status", "Show current soul name and excerpt"),
            ]
        },
        {
            "title": "Utilities",
            "commands": [
                ("/exit / /quit", "Exit Operon cleanly (saves session first)"),
                ("/version", "Print Operon version and build info"),
            ]
        },
    ]

    for grp in command_groups:
        elems.append(SP(2))
        elems.append(P(grp["title"], "h3"))
        rows = [(cmd, desc) for cmd, desc in grp["commands"]]
        elems.append(two_col_table(rows,
            col_widths=[82*mm, PAGE_W - 36*mm - 82*mm],
            header=["Command", "Description"]))

    elems.append(PageBreak())
    return elems


# ── Section 7: Memory ─────────────────────────────────────────────────────────

def build_memory():
    elems = []
    elems += section_break("Memory System — FTS5 + Vector + Obsidian", 7)

    elems.append(P(
        "Operon's memory system provides <b>zero-amnesia persistence</b> across all sessions. "
        "Facts, preferences, and task context are stored in an SQLite database at "
        "<code>~/.operon/memory.db</code> using the FTS5 full-text search extension. "
        "Relevant memories are automatically retrieved and injected into the system prompt "
        "before every LLM call.", "body"))

    elems += sub_section("Database Schema")
    elems += code_block([
        "-- Main table",
        "CREATE TABLE IF NOT EXISTS memories (",
        "    id        INTEGER PRIMARY KEY AUTOINCREMENT,",
        "    content   TEXT NOT NULL,",
        "    timestamp TEXT DEFAULT (datetime('now')),",
        "    tags      TEXT DEFAULT '',",
        "    importance INTEGER DEFAULT 5",
        ");",
        "",
        "-- FTS5 virtual table (BM25 ranked search)",
        "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts",
        "    USING fts5(content, content=memories, content_rowid=id);",
        "",
        "-- Auto-sync triggers",
        "CREATE TRIGGER memories_ai AFTER INSERT ON memories ...",
        "CREATE TRIGGER memories_au AFTER UPDATE ON memories ...",
        "CREATE TRIGGER memories_bd BEFORE DELETE ON memories ...",
    ], label="SQL Schema")

    elems += sub_section("How Auto-Injection Works")
    elems += bullet_list([
        "After every assistant turn, Operon checks if the agent referenced <code>memory_add</code> in its scratchpad",
        "The <code>get_context_string()</code> method retrieves top-N memories by recency and importance",
        "The memory block is injected as a system section: <i>'Persistent memory from past sessions:'</i>",
        "On search queries, FTS5 returns results ranked by BM25 relevance score",
        "The database uses WAL mode for concurrent reads during agent loops",
    ])

    elems += sub_section("FTS5 Upgrade Path")
    elems.append(P(
        "If upgrading from an older Operon version without FTS5, the <code>MemoryPipeline</code> "
        "detects the missing virtual table on startup and automatically runs a backfill — "
        "all existing rows are inserted into the new FTS5 index without data loss.", "body"))

    elems += sub_section("Example Usage")
    elems += code_block([
        "# From the REPL",
        "/memory add My preferred Python style is Black-formatted with type hints",
        "/memory search python style",
        "/memory list 10",
        "/memory delete 5",
        "",
        "# The agent uses memory automatically:",
        "# In scratchpad: objective = 'User prefers Black-formatted Python — will apply'",
    ], label="REPL + agent behaviour")

    elems.append(PageBreak())
    return elems


# ── Section 8: Skills & Curator ───────────────────────────────────────────────

def build_skills():
    elems = []
    elems += section_break("SKILL.md, Curator &amp; Skill Synthesizer", 8)

    elems.append(P(
        "Skills are <b>domain-specific instruction packs</b> that extend Operon's behaviour "
        "for specialised use cases. Each skill is a Markdown file in <code>~/.operon/skills/</code> "
        "containing prompts, examples, API references, coding standards, or workflow descriptions. "
        "The <b>Curator</b> autonomously generates new skills by observing your usage patterns.",
        "body"))

    elems += sub_section("Skill File Format")
    elems += code_block([
        "# SKILL: Python Expert",
        "## Domain: Software Engineering",
        "",
        "## Instructions",
        "When writing Python code:",
        "- Always use Black formatting (88-char line length)",
        "- Add type hints to all function signatures",
        "- Write docstrings in Google style",
        "- Prefer pathlib over os.path",
        "",
        "## Examples",
        "Good: `def process(path: Path) -> list[str]:`",
        "Bad:  `def process(path):`",
    ], label="~/.operon/skills/python_expert.md")

    elems += sub_section("Manual Skill Management")
    elems += code_block([
        "/skills list               # show all loaded skills",
        "/skills reload             # hot-reload from disk",
        "/skills show python_expert # display skill content",
        "",
        "# Create a new skill manually:",
        "# Just drop a .md file in ~/.operon/skills/ and /skills reload",
    ], label="REPL")

    elems += sub_section("Curator — Autonomous Skill Generation")
    elems.append(P(
        "The <b>Curator</b> runs as a background daemon. After observing <b>4 or more tool calls</b> "
        "in a session, it extracts a usage pattern, calls the LLM with a skill-generation prompt, "
        "and saves the result as <code>auto__&lt;fingerprint&gt;.md</code> in the skills directory. "
        "This makes Operon progressively smarter the more you use it.", "body"))

    curator_rows = [
        ("Trigger", "≥ 4 tool calls in the current session"),
        ("Cooldown", "120 seconds between curator runs (prevents over-generation)"),
        ("Max auto-skills", "30 files (oldest removed when limit reached)"),
        ("Deduplication", "80-character content fingerprint — skips duplicate patterns"),
        ("File prefix", "auto__<hash8>.md in ~/.operon/skills/"),
        ("LLM call", "Separate call to configured model with structured SKILL.md prompt"),
        ("Enable/Disable", "/curator on or /curator off (persisted in config)"),
    ]
    elems.append(two_col_table(curator_rows, header=["Property", "Value"]))

    elems += sub_section("Curator Prompt Template")
    elems += code_block([
        "You are the Operon Curator. Analyse the following tool call sequence",
        "and extract a reusable SKILL.md that captures the expertise demonstrated.",
        "",
        "Tool calls observed:",
        "{tool_call_summary}",
        "",
        "Generate a SKILL.md with:",
        "# SKILL: <descriptive name>",
        "## Domain: <area>",
        "## Instructions",
        "<10-20 concrete actionable rules extracted from the pattern>",
        "## Examples",
        "<2-3 before/after examples>",
    ], label="Curator prompt")

    elems.append(PageBreak())
    return elems


# ── Section 9: MCP Integration ────────────────────────────────────────────────

def build_mcp():
    elems = []
    elems += section_break("MCP Server Integration", 9)

    elems.append(P(
        "Operon implements a full <b>Model Context Protocol (MCP)</b> client supporting "
        "the JSON-RPC 2.0 wire format. Connect any MCP-compatible server — filesystem, "
        "database, Git, Slack, Stripe, or custom tools — and Operon dynamically registers "
        "all discovered tools into its live registry.", "body"))

    elems += sub_section("Transport Types")
    transport_rows = [
        ("stdio", "Launches a local subprocess; communicates over stdin/stdout. Each message is a newline-terminated JSON-RPC 2.0 line."),
        ("http", "Connects to a running HTTP/SSE MCP server via POST. Tries /mcp endpoint first (Streamable HTTP), falls back to base URL."),
    ]
    elems.append(two_col_table(transport_rows, header=["Transport", "Description"]))

    elems += sub_section("Connection Lifecycle")
    elems += bullet_list([
        "<b>initialize</b> — send <code>protocolVersion: 2024-11-05</code>, capabilities, clientInfo",
        "<b>notifications/initialized</b> — send handshake completion notification",
        "<b>tools/list</b> — discover all tools from the server",
        "<b>tools/call</b> — invoke individual tools with arguments",
        "<b>Auto-reconnect</b> — saved servers in <code>~/.operon/mcp_servers.json</code> are reconnected on startup",
    ])

    elems += sub_section("Connecting MCP Servers")
    elems += code_block([
        "# Connect a filesystem MCP server (stdio)",
        "/mcp connect filesystem stdio npx @modelcontextprotocol/server-filesystem /tmp",
        "",
        "# Connect a Git MCP server",
        "/mcp connect git stdio npx @modelcontextprotocol/server-git /path/to/repo",
        "",
        "# Connect an HTTP MCP server",
        "/mcp connect my-api http http://localhost:3000",
        "",
        "# List connected servers and tool counts",
        "/mcp list",
        "",
        "# Show all available MCP tools",
        "/mcp tools",
        "",
        "# Show tools for one server only",
        "/mcp tools filesystem",
    ], label="REPL")

    elems += sub_section("Tool Naming Convention")
    elems.append(P(
        "MCP tools are registered with the prefix <code>mcp__&lt;server_name&gt;__&lt;tool_name&gt;</code>. "
        "For example, a tool named <code>read_file</code> on the <code>filesystem</code> server "
        "is accessible as <code>mcp__filesystem__read_file</code>. "
        "These appear in the agent's tool list automatically after connection.", "body"))

    elems += sub_section("Python API")
    elems += code_block([
        "from core.mcp import MCPManager",
        "",
        "mcp = MCPManager()",
        "",
        "# Connect servers",
        'mcp.connect("filesystem", "stdio",',
        '            command=["npx", "@modelcontextprotocol/server-filesystem", "/tmp"])',
        'mcp.connect("my-api", "http", url="http://localhost:3000")',
        "",
        "# List discovered tools",
        "tools = mcp.list_all_tools()  # [{name, description, server, input_schema}, ...]",
        "",
        "# Call a tool directly",
        'result = mcp.call_tool("filesystem", "read_file", {"path": "/tmp/test.txt"})',
        "",
        "# Inject into Operon registry (done automatically in main.py)",
        "added = mcp.inject_into_registry(tool_registry.tools, _TOOL_DEFINITIONS)",
    ], label="Python")

    elems += sub_section("Persistence")
    elems.append(P(
        "Every <code>/mcp connect</code> call persists the server config to "
        "<code>~/.operon/mcp_servers.json</code>. On the next Operon startup, "
        "<code>MCPManager._auto_load()</code> reconnects all saved servers automatically.", "body"))

    elems.append(PageBreak())
    return elems


# ── Section 10: Dashboard ─────────────────────────────────────────────────────

def build_dashboard():
    elems = []
    elems += section_break("Web Dashboard", 10)

    elems.append(P(
        "The Operon Web Dashboard is a <b>zero-dependency SPA</b> (pure stdlib HTTPServer) "
        "running on <code>http://localhost:7270</code>. It provides real-time visibility "
        "into the agent's state: current session, memory, tool call history, and system status. "
        "Auto-refreshes every 5 seconds.", "body"))

    elems += sub_section("Dashboard Tabs")
    tab_rows = [
        ("Status", "Model, memory count, tool count, gateway state, MCP servers, Curator state"),
        ("Session", "Full current conversation with role badges and timestamps"),
        ("Memory", "All stored memories with delete buttons and clear-all action"),
        ("Tool Log", "Chronological log of every tool call: tool name, params, result, duration"),
    ]
    elems.append(two_col_table(tab_rows, header=["Tab", "Content"]))

    elems += sub_section("REST API Endpoints")
    api_rows = [
        ("GET /api/status", "JSON object: model, memory_count, tool_count, gateway_running, mcp_servers, curator_enabled"),
        ("GET /api/session", "JSON array of current session messages [{role, content, timestamp}]"),
        ("GET /api/memory", "JSON array of all memory items [{id, content, timestamp, tags, importance}]"),
        ("GET /api/tools", "JSON array of last 200 tool calls with timing data"),
        ("POST /api/memory/delete?id=N", "Delete memory item by ID"),
        ("POST /api/memory/clear", "Wipe all memory items"),
    ]
    elems.append(two_col_table(api_rows, header=["Endpoint", "Response"]))

    elems += sub_section("Usage")
    elems += code_block([
        "/dashboard start     # Start on localhost:7270",
        "/dashboard open      # Open in browser (macOS: open, Linux: xdg-open)",
        "/dashboard status    # Check if running",
        "/dashboard stop      # Shut down the server",
        "",
        "# Or access the API directly:",
        "curl http://localhost:7270/api/status | python3 -m json.tool",
    ], label="REPL + Shell")

    elems += sub_section("Integration with Tool Calls")
    elems.append(P(
        "Every tool call in the agent loop automatically calls "
        "<code>log_tool_call(tool_name, params, result, duration_ms)</code> "
        "which appends to an in-memory ring buffer (last 200 entries). "
        "The Tool Log tab in the dashboard shows this in real time, making it "
        "easy to debug multi-step agent workflows.", "body"))

    elems.append(PageBreak())
    return elems


# ── Section 11: SSH ───────────────────────────────────────────────────────────

def build_ssh():
    elems = []
    elems += section_break("SSH Remote Execution", 11)

    elems.append(P(
        "The SSH module gives Operon full remote execution capabilities. "
        "Commands, file uploads, and downloads are supported with both "
        "<b>paramiko</b> (Python SSH library) and a <b>system ssh/scp fallback</b> "
        "for environments without paramiko. Connections are pooled within a session.", "body"))

    elems += sub_section("ssh_exec")
    elems += code_block([
        "# Basic remote command",
        'ssh_exec(host="my-server.com", command="df -h", user="ubuntu")',
        "",
        "# With SSH key",
        'ssh_exec(host="10.0.1.5", command="systemctl status nginx",',
        '         user="admin", key_path="~/.ssh/id_ed25519")',
        "",
        "# With custom working directory",
        'ssh_exec(host="server", command="python3 main.py",',
        '         cwd="/opt/myapp", timeout=60)',
    ], label="Python / Agent tool call")

    elems += sub_section("ssh_upload and ssh_download")
    elems += code_block([
        "# Upload a file",
        'ssh_upload(host="server", local_path="/tmp/deploy.tar.gz",',
        '           remote_path="/opt/app/deploy.tar.gz", user="deploy")',
        "",
        "# Download logs",
        'ssh_download(host="server", remote_path="/var/log/app.log",',
        '             local_path="/tmp/app.log")',
    ], label="Python / Agent tool call")

    elems += sub_section("Connection Pooling")
    elems.append(P(
        "Connections are keyed by <code>user@host:port</code> and kept alive for the session. "
        "On reuse, a lightweight keepalive ping verifies the connection before executing. "
        "Stale connections are automatically reconnected.", "body"))

    elems += sub_section("Return Values")
    ret_rows = [
        ("success", "bool — True if exit code was 0"),
        ("output", "string — stdout from the command"),
        ("stderr", "string — stderr from the command"),
        ("exit_code", "int — process exit code"),
        ("host", "string — host that was connected"),
        ("error", "string — error message if connection or execution failed"),
    ]
    elems.append(two_col_table(ret_rows, header=["Key", "Description"]))

    elems.append(PageBreak())
    return elems


# ── Section 12: Telegram Gateway ─────────────────────────────────────────────

def build_telegram():
    elems = []
    elems += section_break("Telegram Gateway", 12)

    elems.append(P(
        "The Telegram Gateway turns Operon into a <b>fully functional Telegram bot</b>. "
        "Send any message to your bot and Operon processes it through a complete agent loop — "
        "with access to all tools — then replies to your chat. Run Operon headlessly on a "
        "server and control it entirely from your phone.", "body"))

    elems += sub_section("Setup")
    elems += code_block([
        "# 1. Create a bot via @BotFather on Telegram → get token",
        "# 2. Configure Operon:",
        "/setup  # Enter token at step 6",
        "# or:",
        "export TELEGRAM_BOT_TOKEN=1234567890:ABC-xyz...",
        "",
        "# 3. Start the gateway",
        "/gateway start",
        "",
        "# 4. Send a message to your bot",
        "# → Operon processes it and replies",
    ], label="Setup + REPL")

    elems += sub_section("Architecture")
    elems += bullet_list([
        "Long-polling — no webhook or public URL required",
        "Each incoming message creates a fresh <b>sub-session</b> with full tool access",
        "Responses are sent back to the originating chat_id",
        "The gateway runs in a background daemon thread — Operon REPL stays interactive",
        "Use <code>/gateway status</code> to see message count and uptime",
    ])

    elems += sub_section("Use Cases")
    elems += bullet_list([
        "Control Operon from your phone while travelling",
        "Receive proactive notifications from scheduled tasks",
        "Allow team members to query Operon via a shared bot",
        "Automate workflow triggers (GitHub webhook → Telegram → Operon agent)",
        "Build a customer-facing assistant with custom soul/skills",
    ])

    elems.append(PageBreak())
    return elems


# ── Section 13: Browser Automation ───────────────────────────────────────────

def build_browser():
    elems = []
    elems += section_break("Browser Automation &amp; Computer Use", 13)

    elems.append(P(
        "Operon includes a full headless browser powered by <b>Playwright</b>. "
        "The browser runs as a singleton — one instance per Operon session — and "
        "supports navigation, interaction, JavaScript evaluation, and screenshots. "
        "Large media resources are automatically blocked for performance.", "body"))

    elems += sub_section("Installation")
    elems += code_block([
        "pip install playwright",
        "playwright install chromium",
        "",
        "# Verify",
        "/doctor  # should show 'playwright: ✓'",
    ], label="Shell")

    elems += sub_section("Common Workflow Pattern")
    elems += code_block([
        "# Typical browser automation sequence:",
        '1. browser_navigate(url="https://example.com")',
        '2. browser_snapshot()                    # see page content',
        '3. browser_click(selector="button#login")',
        '4. browser_type(selector="#email", text="user@example.com")',
        '5. browser_type(selector="#pass",  text="secret")',
        '6. browser_click(selector="button[type=submit]")',
        '7. browser_wait(milliseconds=2000)',
        '8. browser_screenshot(save_path="/tmp/result.png")',
        '9. browser_snapshot()                    # verify result',
    ], label="Agent tool call sequence")

    elems += sub_section("Resource Blocking")
    elems.append(P(
        "Operon blocks the following resource types in the browser to reduce bandwidth "
        "and speed up page loads: <code>image</code>, <code>media</code>, <code>font</code>, "
        "<code>stylesheet</code> (unless needed for interaction). "
        "Use <code>browser_eval</code> to re-enable if required.", "body"))

    elems += sub_section("JavaScript Evaluation")
    elems += code_block([
        "# Extract data from the page",
        'browser_eval(javascript="document.title")',
        '',
        "# Trigger a click on an element Playwright can't locate",
        'browser_eval(javascript="document.querySelector(\'#hidden-btn\').click()")',
        '',
        "# Get all links",
        'browser_eval(javascript="[...document.links].map(l=>l.href)")',
    ], label="Examples")

    elems.append(PageBreak())
    return elems


# ── Section 14: Vision / Image / TTS ─────────────────────────────────────────

def build_vision():
    elems = []
    elems += section_break("Vision, Image Generation &amp; TTS", 14)

    elems += sub_section("vision_analyze")
    elems.append(P(
        "Analyzes images using vision-capable models. Supports both local file paths and "
        "public URLs. Automatically selects GPT-4o (OpenAI) or Claude (Anthropic) based "
        "on your configured provider.", "body"))
    elems += code_block([
        'vision_analyze(image_path="/tmp/screenshot.png",',
        '               prompt="List all UI bugs visible in this screenshot")',
        '',
        'vision_analyze(image_url="https://example.com/chart.png",',
        '               prompt="Extract the data points from this bar chart")',
        '',
        '# Force a specific provider',
        'vision_analyze(image_path="/tmp/x.jpg", provider="anthropic")',
    ], label="Examples")

    elems += sub_section("image_generate")
    elems.append(P(
        "Generates images using <b>DALL-E 3</b>. Requires an OpenAI API key. "
        "Saves the result to disk and returns the file path.", "body"))
    elems += code_block([
        'image_generate(prompt="A futuristic terminal cockpit in space, purple neon glow",',
        '               size="1792x1024", quality="hd",',
        '               save_path="~/Desktop/operon_concept.png")',
    ], label="Examples")

    elems += sub_section("tts_speak")
    elems.append(P(
        "Converts text to speech using <b>OpenAI TTS</b> (6 voices) with automatic "
        "fallback to macOS <code>say</code> if no API key is configured.", "body"))
    voice_rows = [
        ("alloy", "Neutral, balanced — default"),
        ("echo", "Smooth, resonant"),
        ("fable", "Warm, narrative"),
        ("onyx", "Deep, authoritative"),
        ("nova", "Clear, energetic"),
        ("shimmer", "Soft, gentle"),
    ]
    elems.append(two_col_table(voice_rows, header=["Voice", "Character"]))
    elems += code_block([
        'tts_speak(text="Operon mission complete. All systems nominal.",',
        '          voice="onyx", play=True)',
    ], label="Example")

    elems.append(PageBreak())
    return elems


# ── Section 15: Docker ────────────────────────────────────────────────────────

def build_docker():
    elems = []
    elems += section_break("Docker Deployment &amp; Packaging (.app / .exe)", 19)

    elems.append(P(
        "Operon ships with a production-ready Dockerfile and docker-compose.yml for "
        "containerized deployment. The multi-stage build keeps the image lean. "
        "An optional Playwright layer can be included by setting "
        "<code>INSTALL_PLAYWRIGHT=1</code>.", "body"))

    elems += sub_section("Dockerfile (Multi-Stage)")
    elems += code_block([
        "FROM python:3.11-slim AS base",
        "WORKDIR /app",
        "COPY requirements.txt .",
        "RUN pip install --no-cache-dir -r requirements.txt",
        "",
        "# Optional: Playwright browser layer",
        "ARG INSTALL_PLAYWRIGHT=0",
        "RUN if [ \"$INSTALL_PLAYWRIGHT\" = \"1\" ]; then \\",
        "    pip install playwright && playwright install --with-deps chromium; fi",
        "",
        "COPY . .",
        "EXPOSE 7270",
        "CMD [\"python\", \"main.py\"]",
    ], label="Dockerfile")

    elems += sub_section("docker-compose.yml")
    elems += code_block([
        "version: '3.8'",
        "services:",
        "  operon:",
        "    build: .",
        "    ports:",
        "      - '7270:7270'",
        "    volumes:",
        "      - operon_data:/root/.operon",
        "    environment:",
        "      - OPENAI_API_KEY=${OPENAI_API_KEY}",
        "      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}",
        "      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}",
        "      - OLLAMA_BASE_URL=http://host.docker.internal:11434",
        "    extra_hosts:",
        "      - 'host.docker.internal:host-gateway'",
        "    restart: unless-stopped",
        "",
        "volumes:",
        "  operon_data:",
    ], label="docker-compose.yml")

    elems += sub_section("Build &amp; Run")
    elems += code_block([
        "# Copy and fill the env template",
        "cp .env.example .env",
        "# Edit .env with your API keys",
        "",
        "# Build standard image",
        "docker-compose build",
        "",
        "# Build with Playwright (adds ~800MB for Chromium)",
        "docker-compose build --build-arg INSTALL_PLAYWRIGHT=1",
        "",
        "# Start",
        "docker-compose up -d",
        "",
        "# View logs",
        "docker-compose logs -f operon",
        "",
        "# Dashboard is accessible at:",
        "# http://localhost:7270",
    ], label="Shell")

    elems += sub_section("Data Persistence")
    elems.append(P(
        "The <code>operon_data</code> Docker volume maps to <code>/root/.operon</code> "
        "inside the container — the same path used on bare metal. All memories, sessions, "
        "skills, MCP server configs, and settings are persisted across container restarts "
        "and image updates.", "body"))

    elems.append(PageBreak())
    return elems


# ── Section 16: Security ──────────────────────────────────────────────────────

def build_security():
    elems = []
    elems += section_break("Security &amp; Best Practices", 16)

    elems += sub_section("Credential Management")
    elems += bullet_list([
        "<b>Never type API keys directly into the chat.</b> Operon stores memory — keys typed into chat could be recalled later.",
        "Use <b>environment variables</b> for production: <code>export OPENAI_API_KEY=sk-...</code>",
        "For Gmail, create an <b>App Password</b> at myaccount.google.com/apppasswords — never use your main password.",
        "If a credential is accidentally typed into chat, run <b>/forget</b> immediately to wipe all memories.",
        "Config file at <code>~/.operon/config.json</code> is only readable by your user (chmod 600 recommended).",
    ])

    elems += sub_section("Tool Approval Mode")
    elems.append(P(
        "Enable <b>/approve on</b> to require explicit confirmation before any tool call. "
        "The following 'safe' tools never require approval: "
        "<code>file_read, file_exists, file_info, dir_list, duckduckgo_search, "
        "web_scrape, file_search, x_search, browser_get_url, browser_snapshot, "
        "clarify, todo</code>. "
        "All other tools (shell_exec, file_write, ssh_exec, etc.) will pause and ask.", "body"))

    elems += sub_section("Shell Execution Security")
    elems += bullet_list([
        "shell_exec runs commands as the current user — give Operon only the permissions it needs.",
        "For production Docker deployments, run as a non-root user (add <code>USER operon</code> to Dockerfile).",
        "python_exec runs in a subprocess (not isolated sandbox) — avoid executing untrusted user input.",
        "SSH key paths should use dedicated deploy keys with minimal permissions (read-only where possible).",
    ])

    elems += sub_section("Network Security")
    elems += bullet_list([
        "The web dashboard (port 7270) listens on localhost only — do not expose to the internet.",
        "If running on a server, use SSH port forwarding: <code>ssh -L 7270:localhost:7270 user@server</code>",
        "Telegram gateway uses HTTPS long-polling — no inbound ports required.",
        "HTTP MCP servers should only be trusted if running locally or on your internal network.",
    ])

    elems += sub_section("Memory Privacy")
    elems += bullet_list([
        "The memory DB at <code>~/.operon/memory.db</code> stores everything in plaintext.",
        "Encrypt the <code>~/.operon/</code> directory if working with sensitive data.",
        "Use <code>/memory delete &lt;id&gt;</code> to remove specific items or <code>/forget</code> to clear all.",
        "Memory is scoped to your local machine — it is never sent to any external service by default.",
    ])

    elems.append(PageBreak())
    return elems


# ── Section 17: Comparison ────────────────────────────────────────────────────

def build_comparison():
    elems = []
    elems += section_break("Comparison: Operon vs Hermes vs OpenClaw vs OpenHuman", 21)

    elems.append(P(
        "Operon v3.1.0 (Phase 11) is compared below against Hermes Agent v0.14, "
        "OpenClaw (TypeScript, 1.2M LOC), and OpenHuman (Tauri/React desktop app). "
        "Scores are based on actual source code inspection. All four are active open-source projects.", "body"))

    elems.append(SP(2))
    # ── Scores table ─────────────────────────────────────────────────────────
    score_rows = [
        ("Core Agent Loop",           "8.5", "9.5", "9.0", "5.0"),
        ("Context Compression",       "7.5", "9.5", "8.5", "4.0"),
        ("Tool Depth &amp; Breadth",  "8.5", "9.0", "8.0", "5.5"),
        ("Browser / Computer Use",    "8.0", "9.0", "9.5", "9.0"),
        ("Voice &amp; Multimodal",    "6.5", "8.5", "9.5", "9.5"),
        ("Multi-Agent &amp; Delegation","7.5","9.5","9.5","2.0"),
        ("Memory &amp; Persistence",  "8.5", "9.0", "9.0", "8.0"),
        ("Multi-Channel Messaging",   "6.5", "8.5","10.0", "7.0"),
        ("Security Hardening",        "8.0", "8.5", "8.0", "5.0"),
        ("Test Coverage",             "8.5", "9.5", "9.5", "4.0"),
        ("Plugin Ecosystem",          "7.5", "9.5","10.0", "6.0"),
        ("Production Readiness",      "8.0", "8.5", "9.5", "8.5"),
        ("Kanban &amp; Task Mgmt",    "7.5", "9.5", "8.0", "2.0"),
        ("SWE / Code Agent",          "8.0", "7.0", "9.5", "3.0"),
        ("<b>WEIGHTED AVERAGE</b>",   "<b>7.79</b>","<b>9.04</b>","<b>9.11</b>","<b>5.96</b>"),
    ]
    w = PAGE_W - 36*mm
    score_tbl = Table(
        [[P(h, "table_header") for h in ["Category", "Operon v3.1.0", "Hermes v0.14", "OpenClaw", "OpenHuman"]]] +
        [[P(str(c), "table_desc") if i == 0 else P(str(c), "table_cell")
          for i, c in enumerate(row)] for row in score_rows],
        colWidths=[w*0.36, w*0.16, w*0.16, w*0.16, w*0.16],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GRAY_MID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_ROW, GRAY_DARK]),
            ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE_BRIGHT),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BACKGROUND", (0, -1), (-1, -1), PURPLE_BASE),
        ]),
        repeatRows=1
    )
    elems.append(score_tbl)
    elems.append(SP(3))

    elems.append(P("<b>Feature-by-feature comparison</b>", "h2"))
    comp_rows = [
        ("Feature",                  "Operon v3.1.0",          "Hermes",          "OpenClaw",          "OpenHuman"),
        ("Tool count",               "185+ built-in",           "~200+",           "~150+ TS",          "~30 (GUI)"),
        ("AI providers",             "8+ (cloud + local)",      "6+",              "8+",                "5+"),
        ("Vector memory (LanceDB)",  "✓ SentenceTransformers",  "✓",               "✓",                 "✓"),
        ("Obsidian vault sync",      "✓ Full read/write",       "✗",               "✗",                 "✗"),
        ("Smart model routing",      "✓ hint:code/fast/reason", "✓",               "~",                 "~"),
        ("Skill synthesizer",        "✓ Trajectory→Python fn",  "~",               "~",                 "✗"),
        ("Desktop computer use",     "✓ pyautogui + mss",       "~",               "~",                 "✓"),
        ("SWE agent loop",           "✓ Open→fix→test→PR",      "~",               "✓",                 "✗"),
        ("Voice pipeline",           "✓ STT/TTS/VAD",           "✓",               "✓ streaming",       "✓ streaming"),
        ("Multi-agent mesh",         "✓ /mesh parallel/auto",   "✓ ACP",           "✓ ACP",             "✗"),
        ("Browser automation",       "✓ 11 cmds Playwright",    "✓ CDP+stealth",   "✓ CDP+stealth",     "✓ Tauri/CDP"),
        ("Kanban board (SQLite)",     "✓ Full CRUD",             "✓ 10k+ LOC",      "~",                 "✗"),
        ("MCP server",               "✓ stdio + HTTP",          "✓",               "✓",                 "✓"),
        ("Web dashboard",            "✓ localhost:7270",         "✓",               "✓",                 "✓ Tauri GUI"),
        ("SSH remote execution",     "✓ paramiko",              "✓",               "~",                 "✗"),
        ("Telegram gateway",         "✓ bot polling",           "✓",               "✓",                 "✗"),
        ("Plugin marketplace",       "✓ SDK + registry",        "✓ marketplace",   "✓ 186k TS LOC",     "~"),
        ("Claude Code-style TUI",    "✓ prompt_toolkit",        "✗",               "~",                 "✗"),
        ("macOS .app bundle",        "✓ PyInstaller",           "✗",               "✗",                 "✓ Tauri"),
        ("Windows .exe",             "✓ PyInstaller",           "✗",               "✗",                 "✓ Tauri"),
        ("Test count",               "1,896 passing",           "~100,000+",       "~500,000+",         "~600"),
        ("60+ slash commands",       "✓",                       "✓",               "✗ (extension UI)",  "✗"),
        ("Email draft (SMTP)",       "✓ email_draft (safe)",    "email_send ⚠",    "~",                 "✗"),
        ("Scratchpad reasoning",     "✓ JSON-first",            "✓ Hermes format", "~",                 "✗"),
        ("Soul / persona system",    "✓ ~/.operon/SOUL.md",     "✓",               "~",                 "✗"),
        ("Local model support",      "✓ Ollama/LMStudio/Jan",   "✓",               "✓",                 "✓"),
        ("Codebase size",            "~46,700 LOC Python",      "~873k LOC Python","~1.26M LOC TS",     "~130k TS + 35k Rust"),
    ]

    w = PAGE_W - 36*mm
    tbl = Table(
        [[P(str(c), "table_header") for c in comp_rows[0]]] +
        [[P(str(c), "table_desc") if i == 0 else
          (P(str(c), "table_cell") if "✓" in str(c) else P(str(c), "table_desc"))
          for i, c in enumerate(row)] for row in comp_rows[1:]],
        colWidths=[w*0.32, w*0.17, w*0.17, w*0.17, w*0.17],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GRAY_MID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_ROW, GRAY_DARK]),
            ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE_BRIGHT),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
        ]),
        repeatRows=1
    )
    elems.append(tbl)

    elems.append(SP(3))
    elems += info_box(
        "Gap vs Hermes/OpenClaw: Main gaps are async tool dispatch (synchronous Python loop), "
        "monolithic main.py (5,278 LOC), and messaging channel depth. "
        "Operon is MORE capable per-LOC than either competitor — the gap is architecture, not features.",
        CYAN_GLOW
    )

    elems.append(PageBreak())
    return elems


# ── Section 14: SWE Agent ─────────────────────────────────────────────────────

def build_swe_agent():
    elems = []
    elems += section_break("SWE Agent — Automated Software Engineering", 14)

    elems.append(P(
        "The <b>SWE Agent</b> (<code>core/swe_agent.py</code>, 1,161 LOC) provides a "
        "fully automated software engineering loop: given a GitHub issue or bug description, "
        "it clones the repo, reproduces the bug, writes a fix, runs tests, and optionally "
        "opens a pull request — all without human intervention.", "body"))

    elems.append(SP(2))
    elems.append(P("<b>Key capabilities</b>", "h2"))
    swe_rows = [
        ("Command",              "Description"),
        ("/swe fix <issue_url>", "Fetch issue, write fix, run tests, open PR"),
        ("/swe dry <issue_url>", "Show proposed fix without applying it"),
        ("/swe test <path>",     "Run test suite and report failures"),
        ("/swe status",          "Show current SWE loop state"),
    ]
    w = PAGE_W - 36*mm
    elems.append(Table(
        [[P(c, "table_header") for c in swe_rows[0]]] +
        [[P(c, "table_desc") if i == 0 else P(c, "table_cell") for i, c in enumerate(r)]
         for r in swe_rows[1:]],
        colWidths=[w*0.38, w*0.62],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GRAY_MID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_ROW, GRAY_DARK]),
            ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE_BRIGHT),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]),
    ))
    elems.append(SP(2))
    elems.append(P("<b>Skill Synthesizer</b> (<code>core/skill_synthesizer.py</code>): "
        "Analyses conversation trajectories and automatically synthesises new Python tool "
        "functions — effectively giving Operon the ability to write and install its own tools "
        "at runtime. Uses <code>TrajectoryAnalyser</code>, <code>SkillWriter</code>, and "
        "<code>SkillStore</code> components.", "body"))
    elems.append(PageBreak())
    return elems


# ── Section 15: Voice Pipeline ────────────────────────────────────────────────

def build_voice_pipeline():
    elems = []
    elems += section_break("Voice Pipeline &amp; Multimodal", 15)

    elems.append(P(
        "<code>core/voice_pipeline.py</code> (1,095 LOC) provides a complete "
        "speech-to-text → AI processing → text-to-speech pipeline with voice activity "
        "detection (VAD), whisper transcription, and multi-backend TTS.", "body"))

    elems.append(SP(2))
    voice_rows = [
        ("Command / Tool",         "Description"),
        ("/voice speak <text>",    "Convert text to speech (TTS output)"),
        ("/voice listen",          "Record microphone, transcribe with Whisper"),
        ("/voice transcribe <file>","Transcribe an audio file"),
        ("/voice status",          "Show active STT/TTS backends"),
        ("/voice backends",        "List all available STT/TTS engines"),
    ]
    w = PAGE_W - 36*mm
    elems.append(Table(
        [[P(c, "table_header") for c in voice_rows[0]]] +
        [[P(c, "table_desc") if i == 0 else P(c, "table_cell") for i, c in enumerate(r)]
         for r in voice_rows[1:]],
        colWidths=[w*0.38, w*0.62],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GRAY_MID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_ROW, GRAY_DARK]),
            ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE_BRIGHT),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]),
    ))
    elems.append(PageBreak())
    return elems


# ── Section 16: Multi-Agent Mesh ──────────────────────────────────────────────

def build_mesh():
    elems = []
    elems += section_break("Multi-Agent Mesh &amp; Delegation", 16)

    elems.append(P(
        "The <b>Multi-Agent Mesh</b> (<code>core/delegation_bus.py</code>) enables "
        "decomposed parallel and sequential agent execution. Named specialist roles — "
        "<b>RESEARCHER</b>, <b>CODER</b>, <b>ANALYST</b>, <b>WRITER</b>, <b>PLANNER</b> "
        "— each run with their own context window, tool set, and per-turn model routing.", "body"))

    elems.append(SP(2))
    mesh_rows = [
        ("Command",                  "Description"),
        ("/mesh parallel <task>",    "Run all specialist roles concurrently"),
        ("/mesh pipeline <task>",    "Run roles sequentially (chain output)"),
        ("/mesh auto <task>",        "PLANNER decomposes + auto-executes task"),
        ("/mesh roles",              "List available agent roles"),
        ("/mesh status",             "Show mesh bus health and dead-letter queue"),
    ]
    w = PAGE_W - 36*mm
    elems.append(Table(
        [[P(c, "table_header") for c in mesh_rows[0]]] +
        [[P(c, "table_desc") if i == 0 else P(c, "table_cell") for i, c in enumerate(r)]
         for r in mesh_rows[1:]],
        colWidths=[w*0.40, w*0.60],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GRAY_MID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_ROW, GRAY_DARK]),
            ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE_BRIGHT),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]),
    ))
    elems.append(PageBreak())
    return elems


# ── Section 17: Kanban ────────────────────────────────────────────────────────

def build_kanban():
    elems = []
    elems += section_break("Kanban Board &amp; Task Management", 17)

    elems.append(P(
        "Operon includes a full <b>SQLite-backed Kanban board</b> "
        "(<code>core/kanban.py</code>) that persists tasks across sessions. "
        "The board is accessible both from the terminal (/kanban commands) and "
        "programmatically as an agent tool, enabling Operon to track its own work.", "body"))

    elems.append(SP(2))
    kanban_rows = [
        ("Command",                              "Description"),
        ("/kanban board [sprint]",               "ASCII board view of all tasks"),
        ("/kanban add <title> [-- desc]",        "Create a new task"),
        ("/kanban list [status]",                "List tasks (filter by status)"),
        ("/kanban show <id>",                    "Show task details + subtasks"),
        ("/kanban start <id>",                   "Move task to in_progress"),
        ("/kanban done <id> [note]",             "Complete a task"),
        ("/kanban block <id> [reason]",          "Mark task as blocked"),
        ("/kanban export",                       "Export full board to JSON"),
        ("/checkpoint create [msg]",             "Git snapshot of current state"),
        ("/checkpoint restore [sha]",            "Roll back to a snapshot"),
        ("/checkpoint list",                     "List all operon checkpoints"),
    ]
    w = PAGE_W - 36*mm
    elems.append(Table(
        [[P(c, "table_header") for c in kanban_rows[0]]] +
        [[P(c, "table_desc") if i == 0 else P(c, "table_cell") for i, c in enumerate(r)]
         for r in kanban_rows[1:]],
        colWidths=[w*0.44, w*0.56],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GRAY_MID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_ROW, GRAY_DARK]),
            ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE_BRIGHT),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]),
    ))
    elems.append(PageBreak())
    return elems


# ── Section 18: Troubleshooting ───────────────────────────────────────────────

def build_troubleshooting():
    elems = []
    elems += section_break("Troubleshooting &amp; FAQ", 18)

    faqs = [
        (
            "Operon says 'LLM returned empty/non-JSON response'",
            "This usually means the API key is invalid, rate-limited, or the model name is wrong. "
            "Run /doctor to verify your API key is set. Check your provider's dashboard for quota. "
            "Try a different model with /model <name>."
        ),
        (
            "Browser tools fail with 'playwright not installed'",
            "Run: pip install playwright && playwright install chromium. "
            "On Linux you may need: playwright install-deps chromium. "
            "Verify with /doctor — it shows 'playwright: ✓' when ready."
        ),
        (
            "SSH tool fails with 'paramiko not installed'",
            "Run: pip install paramiko. The tool will then use paramiko. "
            "Without paramiko, Operon falls back to the system 'ssh' binary — "
            "ensure it's in your PATH. Key-based auth is recommended over passwords."
        ),
        (
            "Memory database is malformed / corrupted",
            "Delete the database files and restart — Operon recreates them fresh: "
            "rm -f ~/.operon/memory.db ~/.operon/memory.db-wal ~/.operon/memory.db-shm"
        ),
        (
            "MCP server connects but shows 0 tools",
            "The server's tools/list response may be empty or use a different format. "
            "Check the server logs. Some MCP servers require specific initialization parameters. "
            "Run /mcp tools <server> to inspect the raw tool list after connecting."
        ),
        (
            "The agent loop seems stuck / not responding",
            "The model may be on a slow provider. Check your internet connection. "
            "Use /status to see which model is active. Try switching to a local model "
            "with /local use llama3 if you have Ollama running. Press Ctrl+C to interrupt."
        ),
        (
            "Telegram gateway: bot doesn't respond",
            "Verify the token with /doctor. Make sure you started the bot with /gateway start. "
            "Check /gateway status — it shows message count. "
            "The bot token must come from @BotFather and the bot must not be stopped."
        ),
        (
            "Dashboard not accessible at localhost:7270",
            "Run /dashboard start — it's not started automatically. "
            "If port 7270 is in use: lsof -i :7270 to find the conflict. "
            "Verify with /dashboard status. On Docker, ensure port 7270 is mapped."
        ),
        (
            "How do I make Operon forget sensitive information typed in chat?",
            "Run /forget immediately — this wipes all memories from the SQLite database. "
            "For extra safety: rm -f ~/.operon/memory.db to delete the file entirely. "
            "Future sessions will start with a clean memory slate."
        ),
        (
            "Operon's tool calls are truncating large responses",
            "File reads are limited to prevent context overflow. Use file_read on specific "
            "sections or file_search to find relevant parts. Session context is auto-truncated "
            "at 120 messages — use /session compress to reduce it manually."
        ),
        (
            "How do I switch between multiple projects with different skills?",
            "Create separate skill files in ~/.operon/skills/ with descriptive names. "
            "Use /skills reload after adding files. You can also create project-specific "
            ".operon.md files in each project directory — Operon auto-loads them on startup."
        ),
        (
            "Can I use Operon with local models only (no API keys)?",
            "Yes. Install Ollama (ollama.com), pull a model (ollama pull llama3), "
            "then run: /local use llama3. Operon will route all calls to localhost:11434. "
            "Vision and image generation require OpenAI keys regardless."
        ),
    ]

    for question, answer in faqs:
        elems.append(SP(2))
        elems.append(P(f"<b>Q: {question}</b>", "h3"))
        elems.append(P(f"A: {answer}", "body_dim"))
        elems.append(HR(GRAY_DARK, 0.2))

    # Final note
    elems.append(SP(4))
    elems.append(HR(PURPLE_BASE, 1.0))
    elems.append(SP(2))
    elems.append(P("Built with the Hermes + Open-Claw hybrid architecture", "cover_tagline"))
    elems.append(P("Operon AI Terminal Cockpit  •  v3.1.0  •  © 2026", "cover_tagline"))
    elems.append(SP(2))
    elems.append(HR(PURPLE_BASE, 1.0))

    return elems


# ── Back cover splash ─────────────────────────────────────────────────────────

def build_back_cover():
    elems = [PageBreak()]
    elems.append(SP(18))

    if LOGO_PATH.exists():
        try:
            img = Image(str(LOGO_PATH), width=40*mm, height=40*mm)
            img.hAlign = "CENTER"
            elems.append(img)
        except Exception:
            pass

    elems.append(SP(6))
    elems.append(P("OPERON", "cover_title"))
    elems.append(SP(2))
    elems.append(P("AI Terminal Cockpit  •  v3.1.0  •  Phase 11 Build", "cover_sub"))
    elems.append(SP(4))
    elems.append(P(
        "185+ Tools  •  8+ AI Providers  •  60+ Commands  •  1,896 Tests<br/>"
        "MCP • SSH • Dashboard • Curator • Telegram • Docker",
        "cover_tagline"))
    elems.append(SP(4))
    elems.append(HR(PURPLE_BASE, 0.8))
    elems.append(SP(3))
    elems.append(P(
        'github.com/OperonAgent/Operon  •  '
        'Run with: <font color="#22D3EE">python main.py</font>',
        "cover_tagline"))
    return elems


# ── NEW: Feature Setup Guides ────────────────────────────────────────────────

def build_feature_setup():
    elems = []
    elems += section_break("Feature Setup Guides", 3)

    elems.append(P(
        "This chapter walks you through setting up every major Operon feature step by step. "
        "Each guide is self-contained — read only the sections you need. "
        "The <b>/doctor</b> command checks the status of every feature at any time.",
        "body"))

    # ── Email ──────────────────────────────────────────────────────────────────
    elems += sub_section("Email — Gmail Setup &amp; Email Draft Workflow")
    elems.append(P(
        "Operon can compose, preview, and send emails on your behalf. "
        "You describe what you want and Operon writes the full email. "
        "A formatted preview is shown in the terminal — you approve, request changes, "
        "or discard before anything is sent.", "body"))

    elems.append(P("<b>Step 1 — Create a Gmail App Password</b>", "h3"))
    elems += bullet_list([
        "Go to <b>myaccount.google.com/apppasswords</b> in your browser",
        "Sign in with the Gmail account you want Operon to send from",
        "Click <b>Select app</b> → choose <i>Mail</i>",
        "Click <b>Select device</b> → choose <i>Mac</i> (or Other)",
        "Click <b>Generate</b> — you will get a 16-character password like <code>xxxx xxxx xxxx xxxx</code>",
        "Copy it immediately — you will not see it again",
        "<b>Do NOT type this password into Operon chat</b> — use an environment variable instead",
    ])

    elems.append(P("<b>Step 2 — Store Credentials as Environment Variables</b>", "h3"))
    elems += code_block([
        "# Add these two lines to your ~/.zshrc (macOS) or ~/.bashrc (Linux)",
        "export GMAIL_SENDER_EMAIL=your.address@gmail.com",
        "export GMAIL_APP_PASSWORD=xxxxxxxxxxxxxxxxxxxx",
        "",
        "# Apply immediately (or restart your terminal)",
        "source ~/.zshrc",
        "",
        "# Verify Operon can read them — start Operon, then:",
        "# Just ask: 'send email to someone@example.com about anything'",
        "# Operon picks up credentials automatically from the environment",
    ], label="Shell — ~/.zshrc")

    elems.append(P("<b>Step 3 — How the Email Draft Workflow Works</b>", "h3"))
    elems.append(P(
        "Once credentials are set, you never have to specify them again. "
        "Just describe the email you want and Operon composes it:", "body"))
    elems += code_block([
        "👤 YOU  ❯  send an email to example@mail.com asking if Operon is great",
        "",
        "  ⚙  email_draft(to='example@mail.com',",
        "                  subject='Is Operon Great?',",
        "                  body='Hi,\\n\\nI wanted to reach out and ask...')",
        "",
        " ╭──────────────────────────────────────────────────────╮",
        " │  ✉  EMAIL DRAFT — PREVIEW                           │",
        " ├──────────────────────────────────────────────────────┤",
        " │  From   : your.address@gmail.com                    │",
        " │  To     : example@mail.com                          │",
        " │  Subject: Is Operon Great?                          │",
        " ├──────────────────────────────────────────────────────┤",
        " │  Hi,                                                 │",
        " │  I wanted to reach out and ask — do you think       │",
        " │  Operon is great?  I have been using it as an AI    │",
        " │  terminal assistant and would love your thoughts.   │",
        " │  Best regards                                        │",
        " ├──────────────────────────────────────────────────────┤",
        " │  [y] Send  [n] Discard  or type feedback for redraft│",
        " ╰──────────────────────────────────────────────────────╯",
        "",
        "  Your decision ❯ make it more formal",
        "",
        "  [Operon composes a new draft and shows preview again]",
        "",
        "  Your decision ❯ y",
        "  ✓ Email sent to example@mail.com",
    ], label="Terminal — Email Draft Workflow")

    decision_rows = [
        ("y / yes / send", "Sends the email immediately"),
        ("n / no", "Discards the draft — no email sent"),
        ("make it shorter", "Operon writes a shorter version and shows preview again"),
        ("more professional", "Operon rewrites in a formal tone"),
        ("add my name at the end", "Operon updates the draft with your requested change"),
        ("Any feedback text", "Operon incorporates your feedback and redrafts"),
    ]
    elems.append(two_col_table(decision_rows, header=["Your Response", "What Happens"]))

    elems += info_box(
        "Credentials priority: explicit params > GMAIL_SENDER_EMAIL/GMAIL_APP_PASSWORD "
        "env vars > ~/.operon/knowledge.json. Set env vars once and never think about "
        "credentials again.", CYAN_GLOW)

    # ── Knowledge Base ─────────────────────────────────────────────────────────
    elems += sub_section("Knowledge Base — Permanent Memory")
    elems.append(P(
        "The Knowledge Base is Operon's <b>permanent facts store</b>. Unlike conversation memory "
        "(which is session-based), facts in the knowledge base persist forever across "
        "<i>all</i> sessions until you explicitly delete them. "
        "They are stored at <code>~/.operon/knowledge.json</code> and automatically injected "
        "into every session's system prompt.", "body"))

    elems.append(P("<b>What to store there</b>", "h3"))
    elems += bullet_list([
        "Your name, email address, preferred name",
        "Your sender email and credentials for email (saves asking every time)",
        "Project paths, work directories, repository URLs",
        "Preferred coding style, language preferences",
        "API base URLs for your services",
        "Timezone, working hours, any long-lived preference",
    ])

    elems += code_block([
        "# Operon saves facts automatically when you tell it things:",
        "👤 YOU  ❯  my name is Alex",
        "🤖 OPERON  ❯  Got it! I've saved your name permanently.",
        "",
        "👤 YOU  ❯  my sender email is alex@gmail.com",
        "🤖 OPERON  ❯  Saved! I'll use that for all future emails.",
        "",
        "# Manage facts with /knowledge:",
        "/knowledge list              # show all stored facts",
        "/knowledge set email alex@gmail.com  # set a fact directly",
        "/knowledge delete email      # remove a fact",
        "/knowledge clear             # wipe everything",
        "/knowledge path              # show the file location",
    ], label="REPL — Knowledge Base")

    elems.append(two_col_table([
        ("/knowledge list",          "Show all stored facts with their values"),
        ("/knowledge set key value", "Set a fact (spaces in value are fine)"),
        ("/knowledge delete key",    "Remove a specific fact"),
        ("/knowledge clear",         "Wipe all facts (confirmation required)"),
        ("/knowledge path",          "Show path to knowledge.json file"),
    ], header=["Command", "Description"]))

    # ── Ollama / Local Models ──────────────────────────────────────────────────
    elems += sub_section("Local AI Models — Ollama Setup")
    elems.append(P(
        "Operon works with local models via <b>Ollama</b>, <b>LM Studio</b>, and <b>Jan</b>. "
        "No API key required. Runs fully offline.", "body"))

    elems += code_block([
        "# 1. Install Ollama",
        "curl -fsSL https://ollama.com/install.sh | sh    # Linux",
        "# macOS: Download from https://ollama.com",
        "",
        "# 2. Pull a model",
        "ollama pull llama3.2          # 3B — fast, good for basic tasks",
        "ollama pull mistral           # 7B — better reasoning",
        "ollama pull codestral         # 7B — specialized for code",
        "ollama pull llama3.1:70b      # 70B — GPT-4 quality (needs 40GB+ RAM)",
        "",
        "# 3. Switch Operon to use it",
        "/local use llama3.2           # within Operon REPL",
        "# or at setup wizard step 1/2: provider=ollama, model=llama3.2",
        "",
        "# 4. Verify",
        "/doctor   # LOCAL SERVERS section should show: ● ollama  RUNNING",
        "/local    # shows available models",
    ], label="Shell + REPL")

    elems += info_box(
        "Tip: For local models, Operon automatically switches to a simplified system prompt "
        "that works better with smaller models. No configuration needed.", PURPLE_NEON)

    # ── Cloud AI Providers ─────────────────────────────────────────────────────
    elems += sub_section("Cloud AI Providers — API Key Setup")

    cloud_rows = [
        ("OpenAI", "platform.openai.com/api-keys",
         "export OPENAI_API_KEY=sk-...", "/model gpt-4o"),
        ("Anthropic", "console.anthropic.com/settings/keys",
         "export ANTHROPIC_API_KEY=sk-ant-...", "/model claude-3-5-sonnet-20241022"),
        ("OpenRouter", "openrouter.ai/keys",
         "export OPENROUTER_API_KEY=sk-or-...", "/model openrouter/meta-llama/llama-3-70b"),
    ]
    w = PAGE_W - 36*mm
    cloud_table = Table(
        [[P(h, "table_header") for h in ["Provider", "Get API Key", "Set Env Var", "Switch Command"]]] +
        [[P(str(c), "table_cell") for c in row] for row in cloud_rows],
        colWidths=[w*0.13, w*0.28, w*0.30, w*0.29],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GRAY_MID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_ROW, GRAY_DARK]),
            ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE_BRIGHT),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ]),
        repeatRows=1
    )
    elems.append(cloud_table)

    elems += code_block([
        "# Add to ~/.zshrc and run: source ~/.zshrc",
        "export OPENAI_API_KEY=sk-proj-...",
        "export ANTHROPIC_API_KEY=sk-ant-...",
        "export OPENROUTER_API_KEY=sk-or-...",
        "",
        "# Or use the setup wizard inside Operon:",
        "/setup",
        "",
        "# Test — switch model and ask something:",
        "/model gpt-4o",
        "👤 YOU ❯ what model are you?",
    ], label="Shell + REPL")

    # ── Web Search ─────────────────────────────────────────────────────────────
    elems += sub_section("Web Search Setup")
    elems.append(P(
        "Web search requires the <code>duckduckgo_search</code> Python package. "
        "No API key is needed — it's completely free.", "body"))
    elems += code_block([
        "pip install duckduckgo_search",
        "",
        "# Verify in Operon:",
        "/doctor   # should show: ✓ duckduckgo_search",
        "",
        "# Try it:",
        "👤 YOU ❯ search for the latest Python 3.13 release notes",
    ], label="Shell + REPL")

    # ── Browser Automation ─────────────────────────────────────────────────────
    elems += sub_section("Browser Automation Setup — Playwright")
    elems += code_block([
        "# 1. Install Playwright",
        "pip install playwright",
        "",
        "# 2. Install Chromium browser",
        "playwright install chromium",
        "",
        "# On Linux you may also need system dependencies:",
        "playwright install-deps chromium",
        "",
        "# 3. Verify",
        "/doctor   # should show: ✓ playwright chromium  (browser automation ready)",
        "",
        "# 4. Try it:",
        "👤 YOU ❯ go to bbc.com and tell me the top headline",
    ], label="Shell + REPL")

    # ── SSH ────────────────────────────────────────────────────────────────────
    elems += sub_section("SSH Setup")
    elems.append(P(
        "SSH works with or without <b>paramiko</b> — if paramiko is not installed Operon "
        "falls back to the system <code>ssh</code> binary automatically.", "body"))
    elems += code_block([
        "# Optional but recommended: install paramiko",
        "pip install paramiko",
        "",
        "# Test SSH in Operon:",
        "👤 YOU ❯ ssh into my-server.com as ubuntu and show me disk usage",
        "",
        "# Or directly:",
        "ssh_exec(host='my-server.com', user='ubuntu', command='df -h')",
    ], label="Shell + REPL")

    # ── Telegram ───────────────────────────────────────────────────────────────
    elems += sub_section("Telegram Gateway Setup")
    elems += code_block([
        "# 1. Create a bot via @BotFather on Telegram",
        "#    → /newbot → give it a name → copy the token",
        "",
        "# 2. Set the token",
        "export TELEGRAM_BOT_TOKEN=1234567890:ABC-xyz...",
        "# or use /setup step 6 in Operon",
        "",
        "# 3. Get your chat ID (message your bot, then visit:)",
        "# https://api.telegram.org/bot<TOKEN>/getUpdates",
        "# → look for 'chat': {'id': <your_id>}",
        "",
        "# 4. Start the gateway from inside Operon",
        "/gateway start",
        "",
        "# 5. Send a Telegram message to your bot — Operon will reply",
        "# 6. Stop at any time",
        "/gateway stop",
    ], label="Shell + REPL")

    elems.append(PageBreak())
    return elems


# ── NEW: Knowledge Base Reference ────────────────────────────────────────────

def build_knowledge_base():
    elems = []
    elems += section_break("Knowledge Base — Permanent Facts Store", 8)

    elems.append(P(
        "The Knowledge Base is a structured, key-value store of permanent facts. "
        "It is stored at <code>~/.operon/knowledge.json</code> and injected into "
        "<b>every</b> session's system prompt — so Operon always knows the facts "
        "you've told it, regardless of when you told it. "
        "This is separate from session memory (SQLite FTS5), which only covers "
        "recent conversation snippets.", "body"))

    elems += sub_section("How Facts Are Stored")
    elems += code_block([
        '# ~/.operon/knowledge.json',
        '{',
        '  "user_name": {',
        '    "value": "Alex",',
        '    "updated": "2026-05-21 14:30"',
        '  },',
        '  "sender_email": {',
        '    "value": "alex@gmail.com",',
        '    "updated": "2026-05-21 14:31"',
        '  },',
        '  "preferred_language": {',
        '    "value": "Python",',
        '    "updated": "2026-05-21 15:00"',
        '  }',
        '}',
    ], label="~/.operon/knowledge.json")

    elems += sub_section("Automatic Saving")
    elems.append(P(
        "Operon proactively saves facts when you share them in conversation. "
        "You do not need to use a slash command:", "body"))
    elems += code_block([
        "👤 YOU ❯  my name is Alex",
        "  ⚙  knowledge_set(key='user_name', value='Alex')",
        "🤖 OPERON ❯  Got it, Alex! I've saved your name permanently.",
        "",
        "👤 YOU ❯  my timezone is EST",
        "  ⚙  knowledge_set(key='timezone', value='EST')",
        "🤖 OPERON ❯  Saved your timezone as EST.",
        "",
        "# Next session — Operon already knows:",
        "👤 YOU ❯  hi",
        "🤖 OPERON ❯  Hello, Alex! How can I help you today?",
    ], label="REPL — Automatic saving")

    elems += sub_section("System Prompt Injection")
    elems.append(P(
        "Stored facts appear in every session's system prompt as:", "body"))
    elems += code_block([
        "════════════════════════════════════════",
        "PERMANENT KNOWLEDGE  (persists across all sessions)",
        "════════════════════════════════════════",
        "  user_name: Alex",
        "  timezone: EST",
        "  sender_email: alex@gmail.com",
        "  preferred_language: Python",
        "Use knowledge_set to update these facts when you learn new information.",
    ], label="System prompt injection block")

    elems += sub_section("Slash Command Reference")
    kb_rows = [
        ("/knowledge list",            "List all stored key-value facts"),
        ("/knowledge set key value",   "Set a fact (overwrites if exists)"),
        ("/knowledge delete key",      "Remove a specific fact"),
        ("/knowledge clear",           "Wipe all knowledge (requires confirmation)"),
        ("/knowledge path",            "Show the full path to knowledge.json"),
    ]
    elems.append(two_col_table(kb_rows, header=["Command", "Effect"]))

    elems += sub_section("Tool Reference")
    tool_rows = [
        ("knowledge_set",    "key, value",  "Save or overwrite a permanent fact"),
        ("knowledge_get",    "key",         "Retrieve a specific fact by key"),
        ("knowledge_delete", "key",         "Delete a specific fact"),
        ("knowledge_list",   "—",           "Return all stored facts as a formatted string"),
    ]
    w = PAGE_W - 36*mm
    elems.append(Table(
        [[P(h, "table_header") for h in ["Tool", "Params", "Description"]]] +
        [[P(str(c), "table_cell") if i < 2 else P(str(c), "table_desc")
          for i, c in enumerate(row)] for row in tool_rows],
        colWidths=[w*0.22, w*0.20, w*0.58],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GRAY_MID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_ROW, GRAY_DARK]),
            ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE_BRIGHT),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ]),
        repeatRows=1
    ))

    elems += sub_section("Recommended Facts to Store")
    elems += bullet_list([
        "<b>user_name</b> — your first name so Operon can address you",
        "<b>sender_email</b> — your Gmail address for sending emails",
        "<b>app_password</b> — your Gmail App Password (never type in chat)",
        "<b>preferred_language</b> — coding language preference (Python, JavaScript…)",
        "<b>timezone</b> — your timezone for time-based tasks",
        "<b>project_path</b> — default project directory",
        "<b>github_username</b> — for repository operations",
    ])

    elems += info_box(
        "Keys are auto-normalised: spaces become underscores, all lowercase. "
        "So 'My Email' and 'my_email' are the same key.", CYAN_GLOW)

    elems.append(PageBreak())
    return elems


# ── NEW: Practical Examples ───────────────────────────────────────────────────

def build_examples():
    elems = []
    elems += section_break("Practical Examples &amp; Common Workflows", 9)

    elems.append(P(
        "These are real prompts you can paste into Operon right now. "
        "Each example shows exactly what Operon will do step by step.", "body"))

    # ── File & Code ────────────────────────────────────────────────────────────
    elems += sub_section("File &amp; Code Tasks")
    elems += code_block([
        "# List all Python files and count their lines",
        "list all .py files in the current directory and count the lines in each",
        "",
        "# Find a function definition across the project",
        "find where 'def process_payment' is defined in the codebase",
        "",
        "# Create a new Python file",
        "create a file called utils.py with a function that validates email addresses",
        "",
        "# Run a script and show output",
        "run main.py and show me the output",
        "",
        "# Fix a bug",
        "read router.py and fix the parse_response function",
    ], label="File + Code prompts")

    # ── Web & Research ─────────────────────────────────────────────────────────
    elems += sub_section("Web &amp; Research")
    elems += code_block([
        "# Search the web",
        "search for the latest news about Python 3.13",
        "",
        "# Scrape a page",
        "go to news.ycombinator.com and summarise the top 5 stories",
        "",
        "# Research and summarise",
        "search for 'best practices for REST API design 2025' and give me a summary",
        "",
        "# Get current info",
        "what is the current price of Bitcoin? search the web",
    ], label="Web + Research prompts")

    # ── Email ──────────────────────────────────────────────────────────────────
    elems += sub_section("Email")
    elems += code_block([
        "# Compose and send with approval",
        "send an email to boss@company.com asking for Friday off",
        "",
        "# Follow-up email",
        "send a follow-up email to client@example.com about the project status",
        "",
        "# Professional introduction",
        "send an introduction email to partner@startup.com",
        "saying I'm from Operon and would like to explore collaboration",
        "",
        "# Quick note",
        "email dad@gmail.com saying I'll call him tonight",
    ], label="Email prompts")

    # ── System & Shell ─────────────────────────────────────────────────────────
    elems += sub_section("System &amp; Shell")
    elems += code_block([
        "# Check system info",
        "show me CPU and memory usage",
        "",
        "# Find large files",
        "find the 10 largest files in my Downloads folder",
        "",
        "# Manage processes",
        "show me all Python processes currently running",
        "",
        "# Git workflow",
        "show me the git log for this repository (last 10 commits)",
        "",
        "# Install and verify a package",
        "install the requests library and verify it works",
    ], label="System prompts")

    # ── Knowledge ─────────────────────────────────────────────────────────────
    elems += sub_section("Knowledge Base")
    elems += code_block([
        "# Tell Operon facts about you — saved permanently",
        "my name is Alex and I prefer Python over JavaScript",
        "",
        "my sender email is alex@gmail.com",
        "",
        "I work in the directory /Users/alex/projects/myapp",
        "",
        "# These are saved instantly and remembered forever:",
        "/knowledge list",
        "  user_name: Alex",
        "  preferred_language: Python",
        "  sender_email: alex@gmail.com",
        "  project_path: /Users/alex/projects/myapp",
    ], label="Knowledge prompts")

    # ── Browser Automation ─────────────────────────────────────────────────────
    elems += sub_section("Browser Automation")
    elems += code_block([
        "# Read a web page",
        "go to github.com/trending and tell me the top 5 trending repos",
        "",
        "# Take a screenshot",
        "take a screenshot of apple.com and save it to my Desktop",
        "",
        "# Fill a form",
        "go to https://example-form.com, fill in the contact form",
        "with name 'Alex', email 'alex@gmail.com', message 'Hello'",
        "",
        "# Research",
        "open python.org and find the download link for the latest version",
    ], label="Browser prompts")

    # ── Quick reference table ─────────────────────────────────────────────────
    elems += sub_section("Quick Prompt Patterns")
    quick_rows = [
        ("'search for X'",          "duckduckgo_search → summary"),
        ("'find X in the code'",    "file_search → results"),
        ("'run X'",                 "shell_exec or python_exec → output"),
        ("'create file X with Y'",  "file_write → confirmation"),
        ("'send email to X about Y'","email_draft → preview → send on y"),
        ("'go to URL and do X'",    "browser_navigate → interact → result"),
        ("'my name is X'",          "knowledge_set(user_name=X) → saved"),
        ("'what can you do'",       "text reply — no tool call"),
        ("'hi / hello'",            "friendly reply — no tool call"),
    ]
    elems.append(two_col_table(quick_rows, header=["Prompt Pattern", "What Operon Does"]))

    elems.append(PageBreak())
    return elems


# ── Main assembler ────────────────────────────────────────────────────────────

def build_pdf():
    print(f"Building Operon documentation PDF…")
    print(f"  Logo: {LOGO_PATH} ({'found' if LOGO_PATH.exists() else 'NOT FOUND'})")
    print(f"  Output: {OUTPUT_PATH}")

    doc = OperonDocTemplate(str(OUTPUT_PATH))

    story = []
    story += build_cover()
    story += build_toc()
    story += build_intro()
    story += build_installation()
    story += build_feature_setup()      # NEW: step-by-step feature setup guides
    story += build_config()
    story += build_architecture()
    story += build_tools()
    story += build_knowledge_base()     # NEW: knowledge base reference
    story += build_examples()           # NEW: practical examples & workflows
    story += build_commands()
    story += build_memory()
    story += build_skills()
    story += build_mcp()
    story += build_dashboard()
    story += build_ssh()
    story += build_telegram()
    story += build_browser()
    story += build_swe_agent()
    story += build_voice_pipeline()
    story += build_mesh()
    story += build_kanban()
    story += build_vision()
    story += build_docker()
    story += build_security()
    story += build_comparison()
    story += build_troubleshooting()
    story += build_back_cover()

    doc.build(story, canvasmaker=OperonCanvas)
    size_kb = OUTPUT_PATH.stat().st_size // 1024
    print(f"\n  PDF generated successfully!")
    print(f"  Size: {size_kb} KB")
    print(f"  Path: {OUTPUT_PATH}")
    return str(OUTPUT_PATH)


if __name__ == "__main__":
    build_pdf()
