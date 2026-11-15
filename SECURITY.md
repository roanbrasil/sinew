# Security policy

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability reporting
("Report a vulnerability" in the Security tab of this repository). You will get a
reply within 7 days. Once a fix is released, the advisory is published with credit
unless you prefer otherwise.

## Scope

Of particular interest:

- path escapes or command execution beyond the configured policy in agent tools;
- authentication bypass in `sinew-scope`, `sinew-serve` or A2A servers;
- secrets (API keys) leaking into traces, logs or error messages;
- prompt-injection paths that bypass documented guardrails.

Only the latest released version receives security fixes.
