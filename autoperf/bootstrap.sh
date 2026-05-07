#!/usr/bin/env bash
# bootstrap.sh — one-time worktree setup for autoperf 1-agent design.
#
# Creates dedicated worktrees under ~/autoperf/repos/<repo>/ for each
# sibling tool repo, on a shared `autoperf-loop` branch. The autoperf
# agent fixes scoped bugs in these worktrees and opens PRs against each
# repo's main without disturbing the user's primary checkout.
#
# Idempotent — safe to re-run. Existing worktrees are left alone.
#
# Usage:
#   ~/jax-gpt/autoperf/bootstrap.sh
#
# Pre-requisites (verified by the script):
#   - ~/perfsim, ~/cde, ~/xla-shell exist and are git repos
#   - The user has run `pip install -e ~/perfsim` etc. for normal use
#     (autoperf scripts use PYTHONPATH override so they don't conflict)

set -euo pipefail

WORKTREE_ROOT="${HOME}/autoperf/repos"
LOOP_BRANCH="autoperf-loop"

# repo-name : primary-checkout-path
declare -A REPOS=(
    [perfsim]="${HOME}/perfsim"
    [cde]="${HOME}/cde"
    [xla-shell]="${HOME}/xla-shell"
)

mkdir -p "${WORKTREE_ROOT}"

for repo_name in "${!REPOS[@]}"; do
    primary="${REPOS[$repo_name]}"
    worktree="${WORKTREE_ROOT}/${repo_name}"

    echo "=== ${repo_name} ==="

    if [[ ! -d "${primary}/.git" && ! -L "${primary}" ]]; then
        # primary may be a symlink (e.g., ~/perfsim → ~/ml-experiments-perfsim)
        if [[ -L "${primary}" ]]; then
            resolved=$(readlink -f "${primary}")
            if [[ ! -d "${resolved}/.git" ]]; then
                echo "  ERROR: ${primary} is a symlink to ${resolved}, but ${resolved} is not a git repo."
                echo "  Skipping. Set up the repo first."
                continue
            fi
            primary="${resolved}"
        else
            echo "  ERROR: ${primary} is not a git repo. Skipping."
            echo "  (Clone it first: git clone <url> ${primary})"
            continue
        fi
    fi

    if [[ -d "${worktree}" ]]; then
        echo "  Worktree already exists at ${worktree}. Skipping creation."
        # Verify it's actually a worktree of the primary
        if ! git -C "${worktree}" rev-parse --git-dir > /dev/null 2>&1; then
            echo "  WARNING: ${worktree} exists but isn't a git worktree. Investigate manually."
        fi
        continue
    fi

    # Check if autoperf-loop branch exists in primary; create or use existing
    if git -C "${primary}" show-ref --verify --quiet "refs/heads/${LOOP_BRANCH}"; then
        echo "  Branch ${LOOP_BRANCH} already exists in ${primary}. Using it."
        # If branch is checked out elsewhere (e.g., in primary), worktree add will fail.
        # Detect and warn.
        if [[ "$(git -C "${primary}" branch --show-current)" == "${LOOP_BRANCH}" ]]; then
            echo "  ERROR: ${LOOP_BRANCH} is currently checked out at ${primary}."
            echo "  Switch primary to a different branch before bootstrapping the worktree."
            continue
        fi
        git -C "${primary}" worktree add "${worktree}" "${LOOP_BRANCH}"
    else
        echo "  Creating new branch ${LOOP_BRANCH} in worktree."
        git -C "${primary}" worktree add "${worktree}" -b "${LOOP_BRANCH}"
    fi

    echo "  Worktree ready: ${worktree} (branch: ${LOOP_BRANCH})"
done

echo
echo "=== Done. ==="
echo
echo "Use these worktrees for autoperf sibling-repo fixes:"
for repo_name in "${!REPOS[@]}"; do
    echo "  cd ${WORKTREE_ROOT}/${repo_name}  # branch: ${LOOP_BRANCH}"
done
echo
echo "Run perfsim/xla-shell scripts with PYTHONPATH override:"
echo "  PYTHONPATH=${WORKTREE_ROOT}/perfsim python -m perfsim.inference.scripts.headroom_report ..."
echo
echo "Open PRs from autoperf-loop branches against each repo's main:"
echo "  gh pr create --repo ultrons/<repo> --base main --head autoperf-loop ..."
echo
echo "Never merge autoperf-loop PRs. Reviewer agents (or humans) gate merges."
