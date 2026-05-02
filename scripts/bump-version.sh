#!/bin/bash
# scripts/bump-version.sh
# Automate version bumping across all crates and configuration files

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <old-version> <new-version>"
    echo "Example: $0 1.0.0 1.1.0"
    exit 1
fi

OLD_VERSION=$1
NEW_VERSION=$2

echo -e "${YELLOW}Bumping version from ${OLD_VERSION} to ${NEW_VERSION}${NC}"

# Verify version format
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "${RED}Error: Invalid version format. Expected X.Y.Z${NC}"
    exit 1
fi

# 1. Update Cargo.toml workspace version
echo -e "${YELLOW}Updating Cargo.toml...${NC}"
sed -i.bak "s/version = \"$OLD_VERSION\"/version = \"$NEW_VERSION\"/g" Cargo.toml
rm -f Cargo.toml.bak

# 2. Update VERSION.txt
echo -e "${YELLOW}Updating VERSION.txt...${NC}"
echo "$NEW_VERSION" > VERSION.txt

# 3. Update package.json for Unity
if [ -f "packages/OMNIcode-Unity/package.json" ]; then
    echo -e "${YELLOW}Updating Unity package.json...${NC}"
    sed -i.bak "s/\"version\": \"$OLD_VERSION\"/\"version\": \"$NEW_VERSION\"/g" packages/OMNIcode-Unity/package.json
    rm -f packages/OMNIcode-Unity/package.json.bak
fi

# 4. Update OMNIcode.uplugin for Unreal
if [ -f "packages/OMNIcode-Unreal/OMNIcode.uplugin" ]; then
    echo -e "${YELLOW}Updating Unreal OMNIcode.uplugin...${NC}"
    sed -i.bak "s/\"VersionName\": \"$OLD_VERSION\"/\"VersionName\": \"$NEW_VERSION\"/g" packages/OMNIcode-Unreal/OMNIcode.uplugin
    rm -f packages/OMNIcode-Unreal/OMNIcode.uplugin.bak
fi

# 5. Create git tag
echo -e "${YELLOW}Creating git tag v${NEW_VERSION}...${NC}"
git add -A
git commit -m "Bump version to $NEW_VERSION"
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"

echo -e "${GREEN}✅ Version bumped successfully!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review changes: git show v$NEW_VERSION"
echo "2. Push tag: git push origin v$NEW_VERSION"
echo "3. GitHub Actions will automatically build and publish"
echo ""
echo -e "${GREEN}Files updated:${NC}"
echo "  - Cargo.toml (workspace version)"
echo "  - VERSION.txt"
echo "  - package.json (Unity)"
echo "  - OMNIcode.uplugin (Unreal)"
echo ""
