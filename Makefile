# Operon — Makefile convenience targets
#
#   make install        Install Operon + recommended deps + browser binary
#   make install-full   Install every optional feature (voice, db, …)
#   make browser        Download just the Playwright Chromium binary
#   make check          Report dependency status
#   make run            Launch Operon
#   make test           Run the test suite
#   make clean          Remove build artifacts and caches

PYTHON ?= python3

.PHONY: install install-full browser check run test clean doctor docs

install:
	$(PYTHON) install.py

install-full:
	$(PYTHON) install.py --full

browser:
	$(PYTHON) -m core.bootstrap --browser

check:
	$(PYTHON) -m core.bootstrap --check

run:
	$(PYTHON) main.py

doctor:
	$(PYTHON) main.py --check-deps

docs:
	$(PYTHON) generate_docs.py
	$(PYTHON) generate_setup_guide.py
	$(PYTHON) generate_comparison.py

test:
	$(PYTHON) -m pytest tests/ -q

clean:
	rm -rf build dist *.egg-info __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
