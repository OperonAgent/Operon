# ── Operon AI Terminal Cockpit — Production Docker Image ───────────────────────
# Multi-stage build: slim runtime image with optional browser + computer-use support.
#
# ─── BUILD VARIANTS ────────────────────────────────────────────────────────────
#
# Standard (API-only, smallest image, ~500 MB):
#   docker build -t operon .
#
# With browser automation (Playwright Chromium, ~1.5 GB):
#   docker build --build-arg INSTALL_PLAYWRIGHT=1 -t operon-browser .
#
# MCP server mode (expose Operon to Claude Code / Cursor):
#   docker build -t operon-mcp .
#   docker run -i operon-mcp python -m core.mcp_server
#
# ─── RUN EXAMPLES ──────────────────────────────────────────────────────────────
#
# Interactive terminal:
#   docker run -it --rm \
#     -e OPENAI_API_KEY=$OPENAI_API_KEY \
#     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#     -v operon_data:/root/.operon \
#     operon
#
# With dashboard + webhook + MCP server exposed:
#   docker run -it --rm \
#     -p 7270:7270 -p 7271:7271 -p 3456:3456 \
#     -e OPENAI_API_KEY=$OPENAI_API_KEY \
#     -v operon_data:/root/.operon \
#     operon
#
# Non-interactive single prompt:
#   docker run --rm \
#     -e OPENAI_API_KEY=$OPENAI_API_KEY \
#     operon python main.py --prompt "list files in /tmp"
#
# MCP server (stdio mode for Claude Code):
#   docker run -i --rm \
#     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#     -v operon_data:/root/.operon \
#     operon python -m core.mcp_server
# ───────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="Operon"
LABEL org.opencontainers.image.description="Advanced AI Terminal Cockpit"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.source="https://github.com/your-org/operon"

# System deps: git for git_ops, openssh for ssh_exec, build-essential for C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git openssh-client build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker cache doesn't invalidate on source changes
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Optional: Playwright browser automation (large — chromium + deps) ─────────
ARG INSTALL_PLAYWRIGHT=0
RUN if [ "$INSTALL_PLAYWRIGHT" = "1" ]; then \
      pip install --no-cache-dir playwright \
      && playwright install --with-deps chromium \
      && echo "Playwright Chromium installed"; \
    fi

# ── Optional: MCP server package ─────────────────────────────────────────────
ARG INSTALL_MCP=0
RUN if [ "$INSTALL_MCP" = "1" ]; then \
      pip install --no-cache-dir "mcp>=1.0.0" \
      && echo "MCP package installed"; \
    fi

# ── App source ────────────────────────────────────────────────────────────────
COPY . .

# Pre-create all data directories so volume mounts work correctly
RUN mkdir -p \
    /root/.operon/skills \
    /root/.operon/sessions \
    /root/.operon/screenshots \
    /root/.operon/plugins \
    /root/.operon/memories \
    /root/.operon/knowledge

# ── Ports ─────────────────────────────────────────────────────────────────────
# 7270 — Web dashboard (DashboardServer)
# 7271 — REST webhook server (WebhookServer)
# 3456 — MCP HTTP server (core.mcp_server --http)
EXPOSE 7270 7271 3456

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPERON_DATA_DIR=/root/.operon \
    # Disable interactive setup wizard when running non-interactively
    OPERON_SKIP_WIZARD=0

# ── Health check ──────────────────────────────────────────────────────────────
# Quick import check — verifies the core package loads without errors.
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "from tools.registry import ToolRegistry; print('ok')" || exit 1

# ── Default: interactive REPL ─────────────────────────────────────────────────
CMD ["python", "main.py"]
