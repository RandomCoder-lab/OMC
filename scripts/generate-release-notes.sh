#!/bin/bash
# scripts/generate-release-notes.sh
# Extract changelog section and generate GitHub release notes

set -e

# Parse arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.1.0"
    exit 1
fi

VERSION=$1

# Check if CHANGELOG.md exists
if [ ! -f "CHANGELOG.md" ]; then
    echo "Error: CHANGELOG.md not found"
    exit 1
fi

# Extract section for this version
echo "Extracting changelog for version $VERSION..."

# Find the section for this version
PATTERN="## \[$VERSION\]"
START_LINE=$(grep -n "^$PATTERN" CHANGELOG.md | cut -d: -f1)

if [ -z "$START_LINE" ]; then
    echo "Error: No changelog entry found for version $VERSION"
    echo "Expected pattern: ## [$VERSION]"
    exit 1
fi

# Find the next version section (or end of file)
END_LINE=$(tail -n +$((START_LINE + 1)) CHANGELOG.md | grep -n "^## \[" | head -1 | cut -d: -f1)

if [ -z "$END_LINE" ]; then
    # No next section, get all lines from here
    SECTION=$(tail -n +$START_LINE CHANGELOG.md)
else
    # Get lines between START_LINE and END_LINE
    END_LINE=$((START_LINE + END_LINE - 1))
    SECTION=$(sed -n "${START_LINE},$((END_LINE - 1))p" CHANGELOG.md)
fi

# Clean up the section (remove the header line)
RELEASE_NOTES=$(echo "$SECTION" | tail -n +2)

# Create release notes file
OUTPUT_FILE="release-notes-$VERSION.md"
echo "$RELEASE_NOTES" > "$OUTPUT_FILE"

echo ""
echo "✅ Release notes generated: $OUTPUT_FILE"
echo ""
echo "Preview:"
echo "-----"
cat "$OUTPUT_FILE"
echo "-----"
