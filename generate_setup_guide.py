#!/usr/bin/env python3
"""
Generate Operon — Setup Guide PDF (beginner-friendly install walkthrough).
Output: ./Operon_Setup_Guide.pdf  (operon folder)

Self-contained: reuses the Operon brand palette and the same logo as the
full documentation. Run:  python generate_setup_guide.py
"""

from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image,
)
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.colors import HexColor

# ── Brand palette (matches generate_docs.py) ──────────────────────────────────
BLACK        = HexColor("#0A0A0F")
PURPLE_BASE  = HexColor("#7B2FBE")
PURPLE_LIGHT = HexColor("#9D5CE5")
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
CODE_BG      = HexColor("#0D0520")

PAGE_W, PAGE_H = A4
OUTPUT_PATH = Path(__file__).resolve().parent / "Operon_Setup_Guide.pdf"

_LOGO_CANDIDATES = [
    Path.home() / "Downloads" / "Operon Logo.png",
    Path.home() / "Downloads" / "Operon Logo 1.png",
    Path.home() / "Downloads" / "operon_logo.png",
    Path.home() / "Desktop"   / "Operon Logo.png",
]
LOGO_PATH = next((p for p in _LOGO_CANDIDATES if p.exists()), None)


# ── Canvas with footer ────────────────────────────────────────────────────────
class SetupCanvas(pdf_canvas.Canvas):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._states = []

    def showPage(self):
        self._states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._states)
        for st in self._states:
            self.__dict__.update(st)
            self._frame(n)
            super().showPage()
        super().save()

    def _frame(self, n):
        # full-page dark background
        self.setFillColor(BLACK)
        self.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        # footer
        self.setStrokeColor(HexColor("#2A1060"))
        self.setLineWidth(0.5)
        self.line(18 * mm, 14 * mm, PAGE_W - 18 * mm, 14 * mm)
        self.setFont("Helvetica", 8)
        self.setFillColor(GRAY_TEXT)
        self.drawString(18 * mm, 9 * mm, "OPERON  •  Setup Guide")
        self.drawRightString(PAGE_W - 18 * mm, 9 * mm,
                             f"v3.1.0  •  Page {self._pageNumber}")


class SetupDoc(SimpleDocTemplate):
    def __init__(self, path):
        super().__init__(
            path, pagesize=A4,
            leftMargin=18 * mm, rightMargin=18 * mm,
            topMargin=18 * mm, bottomMargin=20 * mm,
            title="Operon — Setup Guide",
            author="Operon", subject="Installation & Setup Guide v3.1.0",
        )


# ── Styles ────────────────────────────────────────────────────────────────────
def _styles():
    S = {}
    S["title"] = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=30,
                                 textColor=WHITE_BRIGHT, alignment=TA_CENTER, leading=34)
    S["subtitle"] = ParagraphStyle("subtitle", fontName="Helvetica", fontSize=12,
                                    textColor=CYAN_GLOW, alignment=TA_CENTER, leading=16,
                                    spaceBefore=4)
    S["h1"] = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17,
                             textColor=PURPLE_NEON, leading=21, spaceBefore=14, spaceAfter=6)
    S["h2"] = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12,
                             textColor=CYAN_GLOW, leading=16, spaceBefore=9, spaceAfter=3)
    S["body"] = ParagraphStyle("body", fontName="Helvetica", fontSize=10,
                               textColor=GRAY_LIGHT, leading=15, spaceAfter=5)
    S["bullet"] = ParagraphStyle("bullet", fontName="Helvetica", fontSize=10,
                                 textColor=GRAY_LIGHT, leading=15, leftIndent=12,
                                 bulletIndent=2, spaceAfter=2)
    S["code"] = ParagraphStyle("code", fontName="Courier", fontSize=9,
                               textColor=HexColor("#A5F3FC"), leading=13)
    S["note"] = ParagraphStyle("note", fontName="Helvetica", fontSize=9.5,
                               textColor=WHITE_BRIGHT, leading=14)
    return S


S = _styles()


def P(t, s="body"):
    return Paragraph(t, S[s])

def SP(h):
    return Spacer(1, h)

def HR(color=PURPLE_BASE, w=0.6):
    return HRFlowable(width="100%", thickness=w, color=color,
                      spaceBefore=4, spaceAfter=6)

def code_block(lines, label="Shell"):
    body = "<br/>".join(
        l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
        or "&nbsp;" for l in lines
    )
    inner = Paragraph(body, S["code"])
    lab   = Paragraph(f'<font color="#7B2FBE"><b>{label}</b></font>', S["code"])
    t = Table([[lab], [inner]], colWidths=[PAGE_W - 36 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.6, PURPLE_BASE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (0, 0), 5),
        ("BOTTOMPADDING", (0, 0), (0, 0), 1),
        ("TOPPADDING", (0, 1), (-1, 1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
    ]))
    return [t, SP(8)]

def bullets(items):
    out = []
    for it in items:
        out.append(Paragraph(
            f'<font color="#C084FC">◆</font>&nbsp;&nbsp;{it}', S["bullet"]))
    return out

def callout(text, color=CYAN_GLOW, tag="TIP"):
    inner = Paragraph(f'<font color="#{color.hexval()[2:]}"><b>{tag}</b></font>&nbsp;&nbsp;{text}', S["note"])
    t = Table([[inner]], colWidths=[PAGE_W - 36 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GRAY_DARK),
        ("BOX", (0, 0), (-1, -1), 0.6, color),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return [t, SP(8)]

def table(rows, header, widths):
    data = [[Paragraph(h, ParagraphStyle("th", fontName="Helvetica-Bold",
             fontSize=9.5, textColor=WHITE_BRIGHT, leading=13)) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), ParagraphStyle("td", fontName="Helvetica",
                     fontSize=9, textColor=GRAY_LIGHT, leading=12)) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GRAY_MID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_ROW, GRAY_DARK]),
        ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return [t, SP(8)]


# ── Build ─────────────────────────────────────────────────────────────────────
def build():
    e = []

    # Cover
    e.append(SP(40))
    if LOGO_PATH and LOGO_PATH.exists():
        try:
            img = Image(str(LOGO_PATH), width=70 * mm, height=70 * mm)
            img.hAlign = "CENTER"
            e.append(img)
        except Exception:
            e.append(P('<font color="#C084FC">▲</font>', "title"))
    e.append(SP(6))
    e.append(P("OPERON", "title"))
    e.append(P("S E T U P   G U I D E", "subtitle"))
    e.append(SP(4))
    e.append(P(f'<font color="#94A3B8">Version 3.1.0  •  {datetime.now().strftime("%B %Y")}  •  '
               f'Get running in under 5 minutes</font>', "subtitle"))
    e.append(SP(14))
    e += callout("This guide takes you from a fresh clone to a working Operon install. "
                 "If you only read one thing: run <font face='Courier'>./install.sh</font> "
                 "(or <font face='Courier'>python install.py</font>) and you're done.",
                 CYAN_GLOW, "START HERE")
    e.append(PageBreak())

    # 1. What you need
    e.append(P("1. What you need", "h1"))
    e.append(HR())
    e += bullets([
        "<b>Python 3.9 or later</b> (3.11+ recommended) — check with <font face='Courier'>python3 --version</font>",
        "<b>git</b> — to clone the repository",
        "<b>macOS, Linux, or Windows</b> — all supported",
        "<b>An AI provider</b> — either an API key (OpenAI / Anthropic / OpenRouter) "
        "OR a local model via Ollama (no key, fully offline)",
    ])
    e += callout("No API key? Install <b>Ollama</b> (ollama.com) and Operon runs "
                 "100% offline with local models. The setup wizard auto-detects it.",
                 PURPLE_NEON, "NOTE")

    # 2. Install
    e.append(P("2. Install (one command)", "h1"))
    e.append(HR())
    e.append(P("The installer creates a virtual environment, installs Operon and its "
               "dependencies, <b>and downloads the Chromium browser binary</b> — the "
               "step a plain pip install always misses.", "body"))

    e.append(P("macOS / Linux", "h2"))
    e += code_block([
        "git clone https://github.com/OperonAgent/Operon.git",
        "cd operon",
        "./install.sh",
    ])
    e.append(P("Windows (PowerShell)", "h2"))
    e += code_block([
        "git clone https://github.com/OperonAgent/Operon.git",
        "cd operon",
        "powershell -ExecutionPolicy Bypass -File install.ps1",
    ])
    e.append(P("Any platform (Python directly)", "h2"))
    e += code_block([
        "python install.py            # core + recommended + browser",
        "python install.py --full     # also voice, databases, screen capture",
        "python install.py --no-venv  # install into current environment",
    ])
    e += callout("<b>Why a separate browser download?</b> "
                 "<font face='Courier'>pip install playwright</font> installs only the Python "
                 "package — the actual Chromium browser (~120 MB) is a separate download. "
                 "Operon handles it automatically, and will even self-install it the first "
                 "time you ask it to browse.", ORANGE_WARN, "IMPORTANT")
    e.append(PageBreak())

    # 3. What gets installed
    e.append(P("3. What gets installed", "h1"))
    e.append(HR())
    e += table(
        [
            ["Core REPL (web search, HTTP, files)", "✓", "✓"],
            ["Telemetry, syntax highlighting, TUI", "✓", "✓"],
            ["PDF reading + generation", "✓", "✓"],
            ["Playwright + Chromium browser binary", "✓", "✓"],
            ["SSH remote execution", "✓", "✓"],
            ["Desktop computer use (mouse/keyboard/screen)", "—", "✓"],
            ["Voice (Whisper STT, TTS)", "—", "✓"],
            ["Databases (Postgres, MongoDB)", "—", "✓"],
            ["Secrets keychain, MCP server", "—", "✓"],
        ],
        header=["Component", "default", "--full"],
        widths=[(PAGE_W - 36 * mm) * 0.62, (PAGE_W - 36 * mm) * 0.19, (PAGE_W - 36 * mm) * 0.19],
    )

    # 4. First launch
    e.append(P("4. First launch &amp; setup wizard", "h1"))
    e.append(HR())
    e += code_block(["operon            # or: python main.py"])
    e.append(P("On first run, a setup wizard walks you through configuration. "
               "Press <b>Enter</b> to skip any field and set it later with "
               "<font face='Courier'>/setup</font>.", "body"))
    e += table(
        [
            ["AI provider", "openai / anthropic / openrouter / ollama / lmstudio / jan"],
            ["Default model", "e.g. gpt-4o, claude-3-5-sonnet, ollama:llama3.2"],
            ["API key(s)", "Paste your key, or Enter to skip (local models need none)"],
            ["Memory", "Yes — enables persistent cross-session memory (recommended)"],
            ["Messaging / gateway", "Optional — Telegram, Discord, Slack, etc."],
        ],
        header=["Wizard asks", "What to enter"],
        widths=[(PAGE_W - 36 * mm) * 0.28, (PAGE_W - 36 * mm) * 0.72],
    )
    e += callout("Operon will offer to download the browser binary on first run if it "
                 "wasn't installed yet. Say yes once and it's cached forever.", CYAN_GLOW, "TIP")
    e.append(PageBreak())

    # 5. Verify
    e.append(P("5. Verify your install", "h1"))
    e.append(HR())
    e += code_block([
        "operon --check-deps     # package + browser-binary status report",
        "",
        "# inside Operon:",
        "/doctor                 # full health check",
    ])
    e.append(P("A healthy install shows green checks for the core packages and "
               "<b>“Chromium browser binary installed — browsing ready”</b>. "
               "<font face='Courier'>/doctor</font> additionally verifies API keys, local "
               "model servers, tool count, and optional services.", "body"))

    # 6. Re-provision
    e.append(P("6. Re-running / repairing deps", "h1"))
    e.append(HR())
    e += code_block([
        "python -m core.bootstrap            # core + recommended + browser",
        "python -m core.bootstrap --full     # every optional feature",
        "python -m core.bootstrap --browser  # just the Chromium binary",
        "python -m core.bootstrap --check    # status only, installs nothing",
        "",
        "# Makefile shortcuts:",
        "make install   |   make browser   |   make check   |   make run",
    ])

    # 7. Troubleshooting
    e.append(P("7. Common issues", "h1"))
    e.append(HR())
    e += table(
        [
            ["“playwright … executable doesn't exist”",
             "Run: python -m core.bootstrap --browser  (downloads Chromium)"],
            ["Browser fails on Linux",
             "sudo python -m playwright install-deps chromium"],
            ["Python too old",
             "Install 3.11+ from python.org; re-run install.py"],
            ["Command 'operon' not found",
             "Activate the venv: source .venv/bin/activate  (Win: .venv\\Scripts\\activate)"],
            ["Want to skip the browser",
             "python install.py --no-browser"],
            ["Voice / whisper too heavy",
             "Use default install (no --full); voice is optional"],
        ],
        header=["Symptom", "Fix"],
        widths=[(PAGE_W - 36 * mm) * 0.42, (PAGE_W - 36 * mm) * 0.58],
    )

    # 8. Next steps
    e.append(P("8. Next steps", "h1"))
    e.append(HR())
    e += bullets([
        "Type a request in plain English — Operon plans, calls tools, and answers.",
        "Try <font face='Courier'>/help</font> to see all 60+ slash commands.",
        "Read <b>Operon_Documentation.pdf</b> for the full technical reference.",
        "Use <font face='Courier'>/doctor</font> anytime to check system health.",
        "Configure messaging, dashboard, or MCP servers via <font face='Courier'>/setup</font>.",
    ])
    e += callout("Pre-built apps are also available on the GitHub Releases page "
                 "(macOS .app, Windows .exe, Linux binary) — no Python required.",
                 PURPLE_NEON, "ALSO")

    return e


def main():
    doc = SetupDoc(str(OUTPUT_PATH))
    doc.build(build(), canvasmaker=SetupCanvas)
    kb = OUTPUT_PATH.stat().st_size // 1024
    print(f"\n  Setup guide generated!")
    print(f"  Size: {kb} KB")
    print(f"  Path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
