---
name: release-notes
description: Summarize recent git history into user-facing release notes.
when_to_use: the user asks for release notes, a changelog, or a version summary.
---

# Release Notes Skill

1. Run `git log --oneline -20` to see the recent work.
2. Group the commits into Added / Changed / Fixed.
3. Write one bullet per commit in plain language — no raw commit hashes.
4. If the directory is not a git repository, say so and stop.
