#!/bin/bash
# Generate release notes from CHANGELOG.md for GitHub releases
# Usage: ./generate-release-notes.sh <version>

set -e

VERSION="$1"

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version> (e.g., 1.0.0)"
    exit 1
fi

# Ensure CHANGELOG.md exists
if [ ! -f "/home/thearchitect/OMC/CHANGELOG.md" ]; then
    echo "CHANGELOG.md not found. Creating template..."
    cat > /home/thearchitect/OMC/CHANGELOG.md << 'EOF'
# Changelog

All notable changes to OMNIcode will be documented in this file.

## [Unreleased]

## [1.0.0] - 2026-05-02
### Added
- Initial release
- C FFI layer (omnimcode-ffi)
- Python bindings (omnimcode-python with PyO3)
- Unity package with C# wrappers
- Unreal Engine plugin
- Circuit evolution core (omnimcode-core)

### Performance
- 509 KB binary size (zero external dependencies)
- 215-693 ns per circuit evaluation
- 4.64M-1.44M evals/sec
EOF
fi

# Extract section for this version
CHANGELOG="/home/thearchitect/OMC/CHANGELOG.md"
TEMP_FILE=$(mktemp)

# Extract the version section (handles "## [1.0.0] - date" format)
awk -v ver="$VERSION" '
  /^## \[/ { 
    if (found) exit
    if ($0 ~ "\\[" ver "\\]") found=1
    next
  }
  found { print }
' "$CHANGELOG" > "$TEMP_FILE"

# Check if we found the version
if [ ! -s "$TEMP_FILE" ]; then
    echo "Version $VERSION not found in CHANGELOG.md"
    rm "$TEMP_FILE"
    exit 1
fi

# Generate GitHub release body
RELEASE_BODY=$(cat << EOF
## OMNIcode v${VERSION}

$(cat "$TEMP_FILE")

## Installation

### Cargo
\`\`\`bash
cargo install omnimcode-core
\`\`\`

### Unity
Import \`OMNIcode-Unity.unitypackage\` into your Unity project.

### Unreal
Copy the \`OMNIcode-Unreal\` plugin to your project's \`Plugins/\` directory.

### Python
\`\`\`bash
pip install omnimcode
\`\`\`

## Performance
- Binary size: 509 KB (zero dependencies)
- Circuit evaluation: 215-693 ns
- Throughput: 4.64M-1.44M evals/sec

## Links
- [Documentation](https://github.com/RandomCoder-lab/OMC/wiki)
- [Crate](https://crates.io/crates/omnimcode-core)
- [Unity Asset Store](https://assetstore.unity.com/)
EOF
)

echo "$RELEASE_BODY"

# Save to file
echo "$RELEASE_BODY" > "/home/thearchitect/OMC/RELEASE_BODY_v${VERSION}.md"

echo "Release notes generated: /home/thearchitect/OMC/RELEASE_BODY_v${VERSION}.md"

# Cleanup
rm "$TEMP_FILE"
