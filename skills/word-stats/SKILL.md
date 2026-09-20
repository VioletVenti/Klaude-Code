---
name: word-stats
description: Count lines, words, and bytes of text files in the workspace.
when_to_use: the user asks for file statistics, sizes, or word counts.
---

# Word Stats Skill

1. Use glob to list the files that match.
2. Run `wc -lw` over the list.
3. Present a small table: file, lines, words.
4. Finish with a totals row.
