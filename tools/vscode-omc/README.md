# OMNIcode for VS Code

OMNIcode language support for VS Code: syntax highlighting, parse-error
diagnostics, heal-pass suggestions, hover documentation, and basic
completion. Powered by `omnimcode-lsp`.

## Installation (developer mode)

1. **Build the language server:**
   ```bash
   cd /path/to/OMC
   cargo build --release -p omnimcode-lsp
   # → target/release/omnimcode-lsp
   ```

2. **Install the extension dependencies:**
   ```bash
   cd tools/vscode-omc
   npm install
   npm run compile
   ```

3. **Launch in dev mode:**
   - Open `tools/vscode-omc` in VS Code
   - Press `F5` to launch a new VS Code window with the extension active
   - Open any `.omc` file — diagnostics + hover + completion should work

4. **Configure the server path** (if `omnimcode-lsp` isn't on PATH):
   - VS Code → Settings → search "omc.serverPath"
   - Set to e.g. `/home/you/OMC/target/release/omnimcode-lsp`

## What it provides

- **Diagnostics**: parse errors appear inline with line/col info
- **Heal-pass hints**: typo corrections, off-attractor literal warnings
  (Information severity, not errors)
- **Hover**: signatures + one-line summaries for `fold`, `harmonic_*`,
  `arr_*`, `dict_*`, `py_*`, etc.
- **Completion**: trigger on `.` or any identifier prefix
- **Syntax highlighting**: TextMate grammar covers keywords, comments,
  strings, numbers, harmonic builtins, type tags

## Packaging for distribution

```bash
npm install -g @vscode/vsce
vsce package
# → vscode-omc-0.1.0.vsix — install via "Extensions: Install from VSIX..."
```

## Editor support (other editors)

The same `omnimcode-lsp` binary works with any LSP client. For Neovim:

```lua
-- In init.lua via nvim-lspconfig
require'lspconfig'.configs.omc = {
    default_config = {
        cmd = { 'omnimcode-lsp' },
        filetypes = { 'omc' },
        root_dir = require'lspconfig'.util.root_pattern('omc.toml', '.git'),
        settings = {},
    },
}
require'lspconfig'.omc.setup{}
```

For Helix, add to `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "omc"
scope = "source.omc"
file-types = ["omc"]
language-servers = ["omnimcode-lsp"]

[language-server.omnimcode-lsp]
command = "omnimcode-lsp"
```
