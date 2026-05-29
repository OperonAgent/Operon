# -*- mode: python ; coding: utf-8 -*-
"""
operon.spec — PyInstaller build specification for Operon v3.0

Produces:
  macOS:   dist/Operon.app  (drag-to-Applications DMG)
  Windows: dist/operon.exe  (single-file portable executable)
  Linux:   dist/operon      (ELF binary)

Build:
  macOS/Linux:  pyinstaller operon.spec
  Windows:      pyinstaller operon.spec
  Cross-compile: not supported — build on each target OS

macOS extras (create DMG):
  pip install dmgbuild
  dmgbuild -s build_dmg.py "Operon" dist/Operon.dmg
"""

import os
import sys
from pathlib import Path

HERE = Path(SPECPATH)   # noqa: F821 — PyInstaller injects SPECPATH

# ── Collect all data files ────────────────────────────────────────────────────
datas = [
    # UI and theme files
    (str(HERE / "ui"),    "ui"),
    # Core modules (already added as source, but include any .json defaults)
    (str(HERE / "core"),  "core"),
    # Tool definitions
    (str(HERE / "tools"), "tools"),
]

# Include default skill packs if they exist
if (HERE / "skills").exists():
    datas.append((str(HERE / "skills"), "skills"))

# Include context files
for ctx_file in [".operon.md", "AGENTS.md", "CLAUDE.md"]:
    if (HERE / ctx_file).exists():
        datas.append((str(HERE / ctx_file), "."))

# ── Hidden imports (lazy-loaded modules) ─────────────────────────────────────
hiddenimports = [
    # Core Python
    "json", "re", "os", "sys", "pathlib", "threading", "subprocess",
    "datetime", "hashlib", "base64", "struct", "zlib", "io",
    "dataclasses", "typing", "enum", "logging",

    # Operon core
    "core.config", "core.session", "core.memory", "core.router",
    "core.planner", "core.soul", "core.scheduler", "core.skills",
    "core.gateway", "core.mcp", "core.dashboard", "core.curator",
    "core.knowledge", "core.cost_tracker", "core.semantic_memory",
    "core.rag", "core.webhook_server", "core.orchestrator",
    "core.secrets", "core.heartbeat", "core.goal_tracker",
    "core.macros", "core.retry_policy", "core.context_compressor",
    "core.plugin_sdk", "core.tool_executor", "core.tokenjuice",
    "core.conversation_compression", "core.plugin_registry",
    "core.vector_memory", "core.obsidian_memory", "core.model_router",
    "core.skill_synthesizer", "core.computer_use",
    "core.swe_agent", "core.voice_pipeline", "core.memory_store",
    "core.delegation_bus", "core.browser_stealth",
    "core.browser_supervisor", "core.credential_pool",
    "core.tool_result_storage", "core.checkpoint_manager",
    "core.reflection", "core.parallel_executor",

    # UI
    "ui.theme", "ui.banner", "ui.tui",

    # Tools
    "tools.registry", "tools.knowledge_ops",
    "tools.slack_ops", "tools.telegram_ops",

    # Third-party (may be lazy-imported)
    "requests", "requests.adapters", "requests.auth",
    "urllib.request", "urllib.parse", "urllib.error",
    "prompt_toolkit", "prompt_toolkit.shortcuts",
    "prompt_toolkit.history", "prompt_toolkit.completion",
    "prompt_toolkit.styles", "prompt_toolkit.key_binding",
    "prompt_toolkit.formatted_text", "prompt_toolkit.lexers",
]

# Optional heavy deps — include if installed
for _dep in ["lancedb", "chromadb", "sentence_transformers",
             "slack_sdk", "pyautogui", "mss", "PIL", "psutil",
             "playwright", "paramiko", "pypdf", "reportlab",
             "sounddevice", "whisper", "pyttsx3"]:
    try:
        __import__(_dep)
        hiddenimports.append(_dep)
    except ImportError:
        pass

# ── Exclude unnecessary bloat ─────────────────────────────────────────────────
excludes = [
    "tkinter", "matplotlib", "scipy", "numpy",
    "IPython", "jupyter", "notebook",
    "test", "tests", "unittest",
    "_pytest", "pytest",
]

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(    # noqa: F821
    [str(HERE / "main.py")],
    pathex      = [str(HERE)],
    binaries    = [],
    datas       = datas,
    hiddenimports = hiddenimports,
    hookspath   = [],
    hooksconfig = {},
    excludes    = excludes,
    win_no_prefer_redirects = False,
    win_private_assemblies  = False,
    cipher      = None,
    noarchive   = False,
)

# ── PYZ archive ───────────────────────────────────────────────────────────────
pyz = PYZ(a.pure, a.zipped_data, cipher=None)   # noqa: F821

# ── macOS .app bundle (onedir mode — required for proper .app bundles) ────────
# On macOS we build a proper onedir .app bundle — NOT onefile.
# onefile + BUNDLE is deprecated and will error in PyInstaller v7.
if sys.platform == "darwin":
    exe = EXE(    # noqa: F821
        pyz,
        a.scripts,
        [],
        exclude_binaries = True,   # onedir: binaries go in _MEIPASS dir
        name            = "operon",
        debug           = False,
        bootloader_ignore_signals = False,
        strip           = False,
        upx             = True,
        console         = True,
        argv_emulation  = False,
        target_arch     = None,
        codesign_identity = None,
        entitlements_file = None,
        icon            = None,
    )
    coll = COLLECT(    # noqa: F821
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip   = False,
        upx     = True,
        upx_exclude = [],
        name    = "operon",
    )
    app = BUNDLE(    # noqa: F821
        coll,
        name            = "Operon.app",
        icon            = None,   # add .icns path here
        bundle_identifier = "ai.operon.terminal",
        info_plist = {
            "NSPrincipalClass":        "NSApplication",
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": "3.1.0",
            "CFBundleVersion":            "3.1.0",
            "CFBundleDisplayName":        "Operon",
            "NSHumanReadableCopyright":   "© 2026 Operon. MIT License.",
            "LSMinimumSystemVersion":     "12.0",
            # Required for pyautogui / computer use accessibility
            "NSAppleEventsUsageDescription": "Operon uses Apple Events for window management.",
            "NSAccessibilityUsageDescription": "Operon uses Accessibility for computer use features.",
            "NSScreenCaptureUsageDescription": "Operon captures the screen for computer use and screenshots.",
        },
    )

else:
    # ── Windows / Linux: single-file portable executable ─────────────────────
    exe = EXE(    # noqa: F821
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name            = "operon",
        debug           = False,
        bootloader_ignore_signals = False,
        strip           = False,
        upx             = True,
        upx_exclude     = [],
        runtime_tmpdir  = None,
        console         = True,
        disable_windowed_traceback = False,
        argv_emulation  = False,
        target_arch     = None,
        codesign_identity = None,
        entitlements_file = None,
        icon            = None,
    )
