# Contributing to Operon

Thanks for your interest in improving Operon! This is an early public beta —
contributions, bug reports, and feedback are very welcome.

## Getting set up

```bash
git clone https://github.com/OWNER/operon.git
cd operon
python install.py            # installs deps + browser binary into .venv
source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
make test                    # run the test suite
```

## Development workflow

1. **Fork** the repo and create a feature branch:
   `git checkout -b fix/short-description`
2. **Make your change.** Keep it focused — one logical change per PR.
3. **Add or update tests.** New behaviour needs coverage; bug fixes should add
   a regression test.
4. **Run the suite:** `make test` (or `python -m pytest tests/ -q`).
   All tests must pass.
5. **Open a pull request** against `main` using the PR template.

## Code style

- Python 3.9+ compatible.
- Keep functions small and single-purpose.
- Prefer standard library; gate optional dependencies behind lazy imports so
  the core REPL always runs.
- Match the existing module structure (`core/`, `tools/`, `ui/`).

## Running specific tests

```bash
python -m pytest tests/test_bootstrap.py -q
python -m pytest tests/test_slash_commands.py -q
python -m pytest tests/ -q -k "kanban"
```

## Reporting bugs

Open an issue using the **Bug report** template. Include:
- What you ran and what happened
- Expected vs actual behaviour
- Output of `operon --check-deps` and your OS / Python version

## Security issues

**Do not** open public issues for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for private disclosure.

## License

By contributing, you agree your contributions are licensed under the project's
[MIT License](LICENSE).
