#!/usr/bin/env python3
"""
Generate Operon — Competitive Comparison PDF.
Output: ./Operon_Comparison.pdf

Brutally-honest comparison of Operon vs Hermes Agent vs OpenClaw vs OpenHuman,
based on source-code inspection. Same brand palette + logo as the other docs.
Run:  python generate_comparison.py
"""

from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image,
)
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.colors import HexColor

# ── Brand palette ──────────────────────────────────────────────────────────────
BLACK        = HexColor("#0A0A0F")
PURPLE_BASE  = HexColor("#7B2FBE")
PURPLE_NEON  = HexColor("#C084FC")
CYAN_GLOW    = HexColor("#22D3EE")
WHITE_BRIGHT = HexColor("#F8FAFC")
GRAY_TEXT    = HexColor("#94A3B8")
GRAY_LIGHT   = HexColor("#CBD5E1")
GRAY_DARK    = HexColor("#141B2D")
GRAY_MID     = HexColor("#1E2D45")
GRAY_ROW     = HexColor("#0F1623")
GREEN_OK     = HexColor("#22C55E")
ORANGE_WARN  = HexColor("#F97316")
RED_WARN     = HexColor("#EF4444")
OPERON_C     = HexColor("#9D5CE5")

PAGE_W, PAGE_H = A4
OUTPUT_PATH = Path(__file__).resolve().parent / "Operon_Comparison.pdf"

_LOGO = next((p for p in [
    Path.home() / "Downloads" / "Operon Logo.png",
    Path.home() / "Downloads" / "Operon Logo 1.png",
    Path.home() / "Desktop"   / "Operon Logo.png",
] if p.exists()), None)


class CmpCanvas(pdf_canvas.Canvas):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self._states = []
    def showPage(self):
        self._states.append(dict(self.__dict__)); self._startPage()
    def save(self):
        for st in self._states:
            self.__dict__.update(st); self._footer(); super().showPage()
        super().save()
    def _footer(self):
        self.setStrokeColor(HexColor("#2A1060")); self.setLineWidth(0.5)
        self.line(18*mm, 14*mm, PAGE_W-18*mm, 14*mm)
        self.setFont("Helvetica", 8); self.setFillColor(GRAY_TEXT)
        self.drawString(18*mm, 9*mm, "OPERON  •  Competitive Comparison")
        self.drawRightString(PAGE_W-18*mm, 9*mm, f"v3.1.0  •  Page {self._pageNumber}")


class CmpDoc(SimpleDocTemplate):
    def __init__(self, path):
        super().__init__(path, pagesize=A4, leftMargin=16*mm, rightMargin=16*mm,
                         topMargin=16*mm, bottomMargin=20*mm,
                         title="Operon — Competitive Comparison",
                         author="Operon", subject="Operon vs Hermes vs OpenClaw vs OpenHuman")
    def handle_pageBegin(self):
        super().handle_pageBegin()
        c = self.canv; c.saveState()
        c.setFillColor(BLACK); c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        c.restoreState()


def _styles():
    S = {}
    S["title"]    = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=26,
                                   textColor=WHITE_BRIGHT, alignment=TA_CENTER, leading=30)
    S["subtitle"] = ParagraphStyle("subtitle", fontName="Helvetica", fontSize=11,
                                   textColor=CYAN_GLOW, alignment=TA_CENTER, leading=15, spaceBefore=4)
    S["h1"]   = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=16,
                               textColor=PURPLE_NEON, leading=20, spaceBefore=12, spaceAfter=6)
    S["body"] = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5,
                               textColor=GRAY_LIGHT, leading=14, spaceAfter=5)
    S["cell"] = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5,
                               textColor=GRAY_LIGHT, leading=11)
    S["cellb"]= ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=8.5,
                               textColor=WHITE_BRIGHT, leading=11)
    S["th"]   = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=9,
                               textColor=WHITE_BRIGHT, leading=12, alignment=TA_CENTER)
    S["note"] = ParagraphStyle("note", fontName="Helvetica", fontSize=9.5,
                               textColor=WHITE_BRIGHT, leading=14)
    S["bullet"]= ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.5,
                                textColor=GRAY_LIGHT, leading=14, leftIndent=12, spaceAfter=2)
    return S

S = _styles()
def P(t, s="body"): return Paragraph(t, S[s])
def SP(h): return Spacer(1, h)
def HR(c=PURPLE_BASE, w=0.6): return HRFlowable(width="100%", thickness=w, color=c, spaceBefore=4, spaceAfter=6)

def bullets(items):
    return [Paragraph(f'<font color="#C084FC">◆</font>&nbsp;&nbsp;{i}', S["bullet"]) for i in items]

def callout(text, color=CYAN_GLOW, tag="TIP"):
    inner = Paragraph(f'<font color="#{color.hexval()[2:]}"><b>{tag}</b></font>&nbsp;&nbsp;{text}', S["note"])
    t = Table([[inner]], colWidths=[PAGE_W - 32*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),GRAY_DARK), ("BOX",(0,0),(-1,-1),0.6,color),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    return [t, SP(8)]


def matrix_table(header, rows, highlight_last=False, score_cols=False):
    w = PAGE_W - 32*mm
    ncol = len(header)
    first_w = w * 0.34
    rest = (w - first_w) / (ncol - 1)
    widths = [first_w] + [rest] * (ncol - 1)
    data = [[Paragraph(h, S["th"]) for h in header]]
    for r in rows:
        row = [Paragraph(str(r[0]), S["cell"])]
        for c in r[1:]:
            row.append(Paragraph(str(c), S["th"] if score_cols else S["cell"]))
        data.append(row)
    style = [
        ("BACKGROUND",(0,0),(-1,0),GRAY_MID),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[GRAY_ROW, GRAY_DARK]),
        ("GRID",(0,0),(-1,-1),0.3,GRAY_MID),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),
    ]
    # Highlight the Operon column (col 1)
    style.append(("TEXTCOLOR",(1,1),(1,-1),OPERON_C))
    if highlight_last:
        style.append(("BACKGROUND",(0,-1),(-1,-1),PURPLE_BASE))
        style.append(("TEXTCOLOR",(0,-1),(-1,-1),WHITE_BRIGHT))
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle(style))
    return [t, SP(8)]


def build():
    e = []
    # Cover
    e.append(SP(30))
    if _LOGO:
        try:
            img = Image(str(_LOGO), width=60*mm, height=60*mm); img.hAlign="CENTER"; e.append(img)
        except Exception: pass
    e.append(SP(6))
    e.append(P("OPERON", "title"))
    e.append(P("Competitive Comparison — Brutally Honest", "subtitle"))
    e.append(P(f'<font color="#94A3B8">vs Hermes Agent · OpenClaw · OpenHuman  •  {datetime.now().strftime("%B %Y")}</font>', "subtitle"))
    e.append(SP(12))
    e += callout("Scores are based on direct source-code inspection of all four "
                 "codebases, not marketing claims. Operon is measured at v3.1.x "
                 "(186 tools, 2,434 tests).", CYAN_GLOW, "METHODOLOGY")
    e.append(PageBreak())

    # 1. Scale
    e.append(P("1. Codebase Scale", "h1")); e.append(HR())
    e.append(P("Raw size context — Operon is by far the smallest codebase, yet "
               "competitive per-feature.", "body"))
    e += matrix_table(
        ["Metric", "Operon", "Hermes", "OpenClaw", "OpenHuman"],
        [
            ["Language",      "Python",   "Python",   "TypeScript", "TS + Rust"],
            ["Form factor",   "Terminal", "Terminal", "Extension",  "Desktop GUI"],
            ["Total LOC",     "~81,300",  "~872,900", "~1,251,700", "~808,000"],
            ["LOC multiple",  "1×",       "10.7×",    "15.4×",      "9.9×"],
            ["Tool defs",     "186",      "~200+",    "~150+",      "~30"],
            ["Tests",         "2,434",    "~100k+",   "~500k+",     "~600"],
            ["Packaged app",  "app/exe",  "no",       "no",         "Tauri"],
        ])

    # 2. Score matrix
    e.append(P("2. Capability Matrix (0–10)", "h1")); e.append(HR())
    e += matrix_table(
        ["Category", "Operon", "Hermes", "OpenClaw", "OpenHuman"],
        [
            ["Core Agent Loop",        "8.8", "9.5", "9.0", "5.0"],
            ["Context Compression",    "9.0", "9.5", "8.5", "4.0"],
            ["Tool Depth & Breadth",   "8.7", "9.0", "8.0", "5.5"],
            ["Browser / Computer Use", "8.5", "9.0", "9.5", "9.0"],
            ["Voice & Multimodal",     "8.0", "8.5", "9.5", "9.5"],
            ["Multi-Agent & Delegation","8.5","9.5", "9.5", "2.0"],
            ["Memory & Persistence",   "8.7", "9.0", "9.0", "8.0"],
            ["Multi-Channel Messaging","8.0", "8.5","10.0", "7.0"],
            ["Security Hardening",     "8.5", "8.5", "8.0", "5.0"],
            ["Test Coverage",          "8.7", "9.5", "9.5", "4.0"],
            ["Plugin Ecosystem",       "7.5", "9.5","10.0", "6.0"],
            ["Production Readiness",    "9.0", "8.5", "9.5", "8.5"],
            ["Kanban & Task Mgmt",     "7.5", "9.5", "8.0", "2.0"],
            ["SWE / Code Agent",       "8.0", "7.0", "9.5", "3.0"],
            ["WEIGHTED AVERAGE",       "8.4", "9.0", "9.1", "6.0"],
        ], highlight_last=True, score_cols=True)
    e.append(PageBreak())

    # 3. Where Operon wins
    e.append(P("3. Where Operon Already Wins", "h1")); e.append(HR())
    e += bullets([
        "<b>Onboarding</b> — one command installs everything <i>including</i> the "
        "Chromium browser binary, with runtime self-heal. No competitor auto-fixes "
        "a missing browser at runtime.",
        "<b>Terminal-native UX</b> — prompt_toolkit TUI + per-turn model routing "
        "beats OpenClaw's extension model and OpenHuman's heavy GUI for keyboard users.",
        "<b>Startup speed</b> — ~0.4s via lazy import probes vs multi-second GUI boot.",
        "<b>Self-improvement</b> — skill synthesizer writes new Python tools from "
        "conversation trajectories. Unique among the four.",
        "<b>Security posture</b> — <font face='Courier'>email_send</font> is "
        "structurally un-callable by the model; CVE dependency scanning in /doctor.",
        "<b>Python reach</b> — <font face='Courier'>import pandas</font> inside a "
        "tool. OpenHuman (Rust/React) structurally cannot.",
    ])

    # 4. What's left
    e.append(P("4. What's Left to Close the Gap", "h1")); e.append(HR())
    e += matrix_table(
        ["Gap", "Priority", "Effort"],
        [
            ["Finish main.py modularization", "High", "~2-3 days"],
            ["Discord depth to match Slack/Telegram", "Medium", "1 wk"],
            ["Plugin community + marketplace seeding", "Medium", "ongoing"],
            ["Local realtime voice (offline streaming STT)", "Low", "2-3 days"],
            ["Deeper SWE multi-file refactors", "Low", "ongoing"],
        ])
    e.append(SP(8))
    e.append(P("5. Closed Since Last Revision (v3.1.x)", "h1")); e.append(HR())
    e += bullets([
        "<b>Streaming cloud voice</b> — real-time Deepgram WebSocket STT "
        "(Voice 6.5 → 8.0).",
        "<b>Messaging depth</b> — Slack threads/edit/schedule/pin/Block Kit and "
        "Telegram edit/delete/pin/photo/document, all wired to the agent "
        "(Messaging 6.5 → 8.0).",
        "<b>Hierarchical multi-agent engine</b> — Engineer/Auditor personas, a "
        "<font face='Courier'>spawn_agent</font> factory with explicit tool "
        "sandboxing, and an autonomous Engineer↔Auditor self-correction loop "
        "(Multi-Agent 7.5 → 8.5).",
        "<b>Non-blocking context compaction</b> wired into the loop "
        "(Context Compression 7.5 → 9.0).",
        "<b>Proactive rate-limit tracking</b> — reads x-ratelimit-* headers to "
        "pause before a 429, complementing the reactive Retry-After path.",
        "<b>Turn-completion notifications</b> — terminal bell (SSH-aware) + "
        "native desktop alerts.",
    ])
    e += callout("This revision lifts Operon's weighted average from 7.9 to 8.4. "
                 "The remaining gap to the Python leader (Hermes) is ~0.6 points and "
                 "is almost entirely architecture, not features. Operon remains the "
                 "most capable framework per line of code of the four.", PURPLE_NEON, "VERDICT")

    return e


def main():
    CmpDoc(str(OUTPUT_PATH)).build(build(), canvasmaker=CmpCanvas)
    kb = OUTPUT_PATH.stat().st_size // 1024
    print(f"\n  Comparison PDF generated!\n  Size: {kb} KB\n  Path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
