# Qwen C2b Execution Protocol

Role: execute only. Do not edit source or reinterpret failures.

Use the exact commit and command from `skill/CODEX-C2B-IMPLEMENTATION-NOTE.md` on the same M4 Max used for C2a.

Before running, record:

```bash
git rev-parse HEAD
git status --short
```

Working tree must be clean.

Run the Codex-provided build and C2b diagnostic command exactly as written. Preserve full stdout/stderr and exit codes.

Return:

```text
HEAD SHA:
WORKTREE STATUS:
BUILD COMMAND + EXIT CODE:
RUN COMMAND + EXIT CODE:
ENVIRONMENT VARIABLES:
FULL LOG:
```

The log must contain:

```text
C2B_CONTROL A0_vs_A1 PASS|FAIL
C2B_PROBE   A0_vs_A2 PASS|FAIL
C2B_RESULT  PASS|FAIL
```

If build or runtime fails, return the raw failure without changing code or rerunning with altered settings.
