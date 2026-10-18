# Klaude

A Claude Code–style terminal coding agent, built from scratch in Python — one harness mechanism at a time.

Klaude grows across 20 cumulative stages. Each stage is a complete, runnable agent that adds exactly one mechanism on top of the previous one, so the history of the harness can be read stage by stage: from a 100-line loop with a single `bash` tool to a full harness with permissions, hooks, context compaction, cross-session memory, background jobs, cron, multi-agent teams, git-worktree isolation and MCP tools (28 tools in the final stage).

The core idea stays fixed throughout: **the loop belongs to the model, the mechanisms belong to the harness.** The agent loop never changes shape. Every capability plugs in around it.

## Stages

| # | Package | Mechanism |
|---|---------|-----------|
| 01 | `s01_agent_loop` | Minimal agent loop: call the model, run tools, feed results back until it stops |
| 02 | `s02_tool_use` | Tool registry with a dispatch map; workspace-jailed `read` / `write` / `edit` / `glob` |
| 03 | `s03_permissions` | Permission pipeline before execution: hard-deny rules, then ask a human for risky tools |
| 04 | `s04_hooks` | Lifecycle hooks (`user_prompt`, `pre_tool`, `post_tool`, `stop`) as the loop's extension points |
| 05 | `s05_todo` | Structured todo list with validation and stale-plan reminders |
| 06 | `s06_subagents` | Delegation to an isolated worker agent with a clean message history |
| 07 | `s07_skills` | Two-level skill loading: manifests in the prompt, bodies loaded on demand |
| 08 | `s08_compaction` | Escalating context compaction: persist, snip, point to archives, then summarize |
| 09 | `s09_memory` | Cross-session memory: recall at prompt time, extract at stop, consolidate when crowded |
| 10 | `s10_system_prompt` | System prompt assembled at runtime from cached sections and live state |
| 11 | `s11_error_recovery` | Error recovery: exponential backoff, token-limit escalation, model fallback |
| 12 | `s12_tasks` | Persistent, dependency-tracked tasks (one JSON file per task) |
| 13 | `s13_background` | Background jobs for slow commands, with completion notifications injected into the loop |
| 14 | `s14_scheduler` | Cron scheduler with durable jobs that fire while the agent is alive |
| 15 | `s15_teams` | Persistent teammates running as independent agent threads with file mailboxes |
| 16 | `s16_protocols` | Typed request/response protocols over the mailbox bus (shutdown, plan review) |
| 17 | `s17_autonomous` | Autonomous teammates that claim work from the shared task board |
| 18 | `s18_worktrees` | One git worktree per task so parallel teammates never touch the same files |
| 19 | `s19_mcp` | MCP-style external tool servers: discovery, namespacing and host-side gating |
| 20 | `s20_full_harness` | Comprehensive harness: every mechanism wired into one turn |

A single turn in the final stage flows through:

```
Intake (user_prompt hooks) → Guardrails (pre_tool hooks + permissions)
→ Route (tool registry + MCP pool) → Execute (tools, slow work to background)
→ External (teammates, cron, MCP servers) → Recover (backoff, escalation, reactive compaction)
→ Append (post_tool hooks, notifications, memory extraction at stop)
```

## Quick start

Requires Python 3.11+ and any OpenAI-compatible chat-completions endpoint that supports tool calling.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then fill in KLAUDE_API_KEY and KLAUDE_BASE_URL

.venv/bin/python -m s20_full_harness   # the full harness
.venv/bin/python -m s01_agent_loop     # or any earlier stage
```

Type a task at the prompt; `q` quits.

### Configuration

| Variable | Purpose |
|----------|---------|
| `KLAUDE_API_KEY` | API key for the endpoint (required) |
| `KLAUDE_BASE_URL` | Base URL of the OpenAI-compatible endpoint (required) |
| `KLAUDE_MODEL` / `KLAUDE_FALLBACK_MODEL` | Primary model and the fallback used during error recovery |
| `KLAUDE_WORKDIR` | Workspace the file tools are jailed to |
| `KLAUDE_MEMORY_DIR`, `KLAUDE_TASKS_DIR`, `KLAUDE_MAILBOX_DIR`, `KLAUDE_WORKTREES_DIR`, `KLAUDE_ARCHIVE_DIR`, `KLAUDE_SKILLS_DIR`, `KLAUDE_SCHEDULE_FILE` | Override where each mechanism keeps its on-disk state |

## Tests

```bash
.venv/bin/python -m pytest -q
```

155 tests across all 20 stages. The model client is stubbed, so the suite runs offline, and every stage's file tools are jailed to a temporary sandbox. During development, a stage was committed only after its tests were green and an end-to-end run against a live model exited cleanly.

## Repository layout

```
s01_agent_loop/ … s20_full_harness/   one self-contained package per stage
skills/                               example skills loaded by s07+
tests/                                one test module per stage
```

Stages are deliberately self-contained rather than importing from each other, so each directory can be read (and diffed against its predecessor) on its own.

## Acknowledgements

The stage outline follows the [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) course ([learn.shareai.run](https://learn.shareai.run/)). Klaude is an independent implementation.

## License

[MIT](LICENSE)
