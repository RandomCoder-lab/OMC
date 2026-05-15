# OMNIcode Package Registry

Canonical index for `omc --install <name>`. Maps short package names
to canonical URLs + sha256 hashes for reproducible installs.

## How resolution works

1. `omc --install np` looks up `np` in this registry's `index.json`.
2. Fetches `packages.np.url`, verifies the sha256 matches.
3. Writes to `omc_modules/np.omc` in the project's working directory.
4. `import "np";` then resolves from `omc_modules/`.

## Submitting a package

PR a new entry to `registry/index.json`:

```json
"yourlib": {
    "url": "https://raw.githubusercontent.com/you/yourlib/main/yourlib.omc",
    "sha256": "<run `sha256sum yourlib.omc`>",
    "version": "0.1.0",
    "description": "one-line summary"
}
```

Hosting the actual `.omc` files is YOUR responsibility (any HTTPS
URL works — GitHub raw, your own server, a CDN). The registry
just maps names to URLs.

## Default registry URL

`omc --install` defaults to looking up names against:

    https://raw.githubusercontent.com/sovereignlattice/omnimcode/main/registry/index.json

Override with `OMC_REGISTRY=<url>` if you're running a private fork.
