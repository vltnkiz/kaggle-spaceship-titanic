#!/bin/sh
# Point git at the tracked hooks/ directory. Run once per clone; worktrees share the setting.
set -e
git config core.hooksPath hooks
echo "core.hooksPath = hooks (pre-commit now refuses commits touching harness/)"
