# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 3.1.x   | ✓         |
| < 3.1   | ✗         |

## Reporting a Vulnerability

If you discover a security vulnerability in Operon, please report it
**privately** — do not open a public issue.

- Use GitHub's **"Report a vulnerability"** button under the repository's
  **Security** tab (Private Vulnerability Reporting), or
- Email the maintainers at the address listed on the project's GitHub profile.

Please include:
- A description of the vulnerability and its impact
- Steps to reproduce
- Affected version(s)

We aim to acknowledge reports within 72 hours and to ship a fix or mitigation
as quickly as the severity warrants.

## Security Model & Hardening

Operon runs AI-driven tools on your machine. Key built-in protections:

- **`email_send` is never model-callable.** Only `email_draft` is exposed to
  the LLM; sending always requires explicit human confirmation. `email_send`
  appears in neither the tool definitions nor the dispatcher.
- **Credentials are intercepted** by regex before the model ever sees plaintext
  secrets, and the model is instructed never to request passwords in chat.
- **Sub-agents run with a blocked-tool list** (`email_draft`, `email_send`, and
  other sensitive tools are denied in delegated agents).
- **Command risk classifier** flags destructive shell commands
  (e.g. `rm -rf /`) as CRITICAL before execution.
- **Prompt-injection detection** scores incoming content for manipulation
  attempts.
- **Secrets at rest** can be stored via the OS keychain (keyring) or Fernet
  (cryptography) — never in plaintext config when the secrets manager is used.

## User Responsibilities

- **Never commit your `~/.operon/config.json`** or any file containing API keys.
  The bundled `.gitignore` excludes these by default.
- **Rotate any API key or app password** that may have been exposed.
- Review tool-approval prompts before granting access to shell, filesystem,
  or network tools in untrusted contexts.
- Run untrusted workloads inside the Docker sandbox where possible.
