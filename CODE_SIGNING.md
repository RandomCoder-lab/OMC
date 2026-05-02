# Binary Code Signing Guide

## Overview
Code signing is **optional** but recommended for distribution. It prevents "unidentified developer" warnings on macOS and "Windows protected your PC" on Windows.

## macOS Code Signing

### Requirements
- **Apple Developer Account**: $99/year (required for distribution outside Mac App Store)
- **Developer ID Certificate**: Request from Apple Developer portal
- **Xcode Command Line Tools**: `xcode-select --install`

### Signing Process
```bash
# Sign the binary
codesign --force --deep --sign "Developer ID Application: Your Name (TEAMID)" \
    target/release/libomnimcode_ffi.dylib

# Verify signature
codesign --verify --verbose libomnimcode_ffi.dylib

# Check signature details
codesign --display --verbose libomnimcode_ffi.dylib
```

### Notarization (Required for Distribution)
```bash
# Create ZIP for notarization
ditto -c -k --sequesterRsrc --keepParent libomnimcode_ffi.dylib libomnimcode_ffi.zip

# Submit to Apple for notarization
xcrun altool --notarize-app \
    --primary-bundle-id "com.omnicode.lib" \
    --username "your@appleid.com" \
    --password "app-specific-password" \
    --file libomnimcode_ffi.zip

# Staple the notarization ticket
xcrun stapler staple libomnimcode_ffi.dylib
```

### Cost
- **Free**: Self-signed certificates (triggers warnings)
- **$99/year**: Apple Developer Program (required for distribution)

## Windows Code Signing

### Requirements
- **Code Signing Certificate**: 
  - Cheap: Sectigo/Comodo (~$80/year)
  - Free: Self-signed (triggers SmartScreen warnings)
- **signtool.exe**: Part of Windows SDK

### Signing Process (on Windows)
```cmd
REM Sign the DLL
signtool sign /tr http://timestamp.digicert.com /td sha256 \
    /fd sha256 /a target\release\omnimcode_ffi.dll

REM Verify signature
signtool verify /pa /v omnimcode_ffi.dll
```

### Cross-Signing from Linux (Advanced)
```bash
# Use osslsigncode
sudo apt-get install osslsigncode

osslsigncode sign \
    -certs your-certificate.pem \
    -key your-private-key.pem \
    -in omnimcode_ffi.dll \
    -out omnimcode_ffi-signed.dll \
    -t http://timestamp.digicert.com
```

### Cost
- **Free**: Self-signed (triggers SmartScreen "Windows protected your PC")
- **~$80/year**: Standard OV certificate (reduced warnings)
- **~$300/year**: EV certificate (immediate SmartScreen reputation)

## Linux (No Signing Required)
Linux doesn't require code signing for shared libraries. However:
- **GPG signatures**: For package repositories (optional)
- **Checksums**: Provide SHA256 checksums (recommended)

```bash
# Generate checksum
sha256sum libomnimcode_ffi.so > libomnimcode_ffi.so.sha256

# Verify
sha256sum -c libomnimcode_ffi.so.sha256
```

## Trade-offs Summary

| Option | Cost | User Experience | Recommended For |
|--------|------|-----------------|------------------|
| Unsigned | Free | ⚠️ Warnings on macOS/Windows | Internal/testing only |
| Self-signed | Free | ⚠️ Warnings (user must trust) | Development |
| Apple Developer ($99) | $99/year | ✅ No warnings on macOS | Distribution to macOS users |
| Standard OV Cert ($80) | $80/year | ⚠️ Some SmartScreen warnings | Small distribution |
| EV Cert ($300) | $300/year | ✅ No SmartScreen warnings | Commercial distribution |

## Recommendations for OMNIcode
Given OMNIcode's current stage:
1. **Phase 1-3**: Skip code signing (use unsigned binaries)
2. **Phase 4+**: If distributing via Unity Asset Store/Unreal Marketplace, follow their signing requirements
3. **Commercial launch**: Purchase EV certificate for Windows, Apple Developer for macOS

## Adding to CI
Once you have certificates:
```yaml
# In .github/workflows/build-binaries.yml
- name: Sign macOS binary
  if: runner.os == 'macOS'
  run: |
    codesign --force --sign "$MACOS_CERTIFICATE" libomnimcode_ffi.dylib
  env:
    MACOS_CERTIFICATE: ${{ secrets.MACOS_CERTIFICATE }}

- name: Sign Windows DLL
  if: runner.os == 'Windows'
  run: |
    signtool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 /a omnimcode_ffi.dll
  env:
    WINDOWS_CERTIFICATE: ${{ secrets.WINDOWS_CERTIFICATE }}
```

## Current Status for OMNIcode
- **Task 3.4**: Documented (this file)
- **Decision**: **Skip code signing for now** (Phase 3)
- **Budget**: $0 (use unsigned binaries)
- **Next step**: Revisit when distributing via asset stores (Phase 7+)
