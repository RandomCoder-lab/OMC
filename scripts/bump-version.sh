#!/bin/bash
# Bump version across all crates and config files
# Usage: ./bump-version.sh <major|minor|patch>

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <major|minor|patch>"
  exit 1
fi

# Read current version from Cargo.toml
CURRENT_VERSION=$(grep -m1 'version = "' /home/thearchitect/OMC/Cargo.toml | sed 's/.*"$.*$".*/\1/')
echo "Current version: $CURRENT_VERSION"

# Split version into major, minor, patch
MAJOR=$(echo $CURRENT_VERSION | cut -d. -f1)
MINOR=$(echo $CURRENT_VERSION | cut -d. -f2)
PATCH=$(echo $CURRENT_VERSION | cut -d. -f3)

# Bump version
case $1 in
  major)
    MAJOR=$((MAJOR + 1))
    MINOR=0
    PATCH=0
    ;;
  minor)
    MINOR=$((MINOR + 1))
    PATCH=0
    ;;
  patch)
    PATCH=$((PATCH + 1))
    ;;
  *)
    echo "Invalid bump type: $1. Use major, minor, or patch."
    exit 1
    ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo "New version: $NEW_VERSION"

# Update all Cargo.toml files
find /home/thearchitect/OMC -name "Cargo.toml" -exec sed -i "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/g" {} \;

# Update VERSION.txt
echo $NEW_VERSION > /home/thearchitect/OMC/VERSION.txt

# Update package.json (Unity package)
sed -i "s/\"version\": \".*\"/\"version\": \"$NEW_VERSION\"/g" /home/thearchitect/OMC/packages/OMNIcode-Unity/package.json

# Create git tag
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"

echo "Version bumped to $NEW_VERSION. Remember to git push --tags"
