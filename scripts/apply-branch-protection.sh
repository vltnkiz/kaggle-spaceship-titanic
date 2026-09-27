#!/bin/sh
# Reviewable source of truth for main's branch protection. Run by hand (or from a workflow)
# after a change here is merged to main — never from a feature branch.
#
# Ordering rule (see docs/agents/issue-tracker.md#branch-protection): a required status check
# can only ever report if its workflow already exists on the target branch. Add a check to
# required_status_checks.contexts only after the workflow that produces it has landed on
# main; adding it first makes every future pull request permanently BLOCKED, and
# enforce_admins leaves no override once that happens.
set -e

REPO="${REPO:-vltnkiz/kaggle-spaceship-titanic}"

gh api --method PUT "repos/$REPO/branches/main/protection" --input - <<'JSON'
{
  "required_status_checks": {
    "strict": false,
    "contexts": ["harness/ is frozen", "scorer runs (smoke)"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 0
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON

echo "Applied branch protection to $REPO#main from scripts/apply-branch-protection.sh"
