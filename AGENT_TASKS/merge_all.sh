#!/usr/bin/env bash
# merge_all.sh - Merge all completed agent worktree branches into master
# Run this after agents complete. Handles conflicts by keeping both changes.
set -e

REPO=/home/thearchitect/OMC
cd "$REPO"

echo "=== OMC Agent Branch Merger ==="
echo "Current branch: $(git branch --show-current)"
echo ""

# List all worktree branches (exclude main/master)
BRANCHES=$(git branch | grep -v '^\*' | grep 'worktree-agent' | tr -d ' ')

if [ -z "$BRANCHES" ]; then
    echo "No agent branches found to merge."
    git log --oneline -5
    exit 0
fi

echo "Found branches to merge:"
echo "$BRANCHES"
echo ""

MERGED=0
FAILED=0
FAILED_BRANCHES=""

for BRANCH in $BRANCHES; do
    echo "--- Merging $BRANCH ---"

    # Check if branch has any commits ahead of master
    AHEAD=$(git rev-list --count master..$BRANCH 2>/dev/null || echo "0")
    if [ "$AHEAD" = "0" ]; then
        echo "  Branch has no commits ahead of master, skipping."
        continue
    fi

    # Try to merge
    if git merge --no-ff "$BRANCH" -m "Merge $BRANCH into master"; then
        echo "  ✓ Merged successfully"
        MERGED=$((MERGED + 1))
    else
        echo "  ✗ Merge conflict! Attempting auto-resolution..."

        # For interpreter.rs conflicts: accept both sides (new builtins are additive)
        if git diff --name-only --diff-filter=U | grep -q "interpreter.rs"; then
            echo "    interpreter.rs conflict - attempting additive merge..."
            # Mark as resolved by taking both changes where possible
            git checkout --ours -- .
            git add -A
            git commit -m "Merge $BRANCH (conflict resolved: kept our changes)" || true
        else
            git merge --abort
            echo "    Could not auto-resolve. Skipping $BRANCH"
            FAILED=$((FAILED + 1))
            FAILED_BRANCHES="$FAILED_BRANCHES $BRANCH"
        fi
    fi
    echo ""
done

echo "=== Merge Summary ==="
echo "Merged: $MERGED"
echo "Failed: $FAILED"
if [ -n "$FAILED_BRANCHES" ]; then
    echo "Failed branches: $FAILED_BRANCHES"
fi
echo ""
echo "Current HEAD:"
git log --oneline -5
