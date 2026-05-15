// VS Code activation entry point. Spawns the omnimcode-lsp binary
// and connects via stdio per the LSP protocol. The actual language
// intelligence lives in the LSP server (Rust); this file is the
// minimum glue VS Code needs to wire everything up.

import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('omc');
    const serverPath = config.get<string>('serverPath') || 'omnimcode-lsp';

    const serverOptions: ServerOptions = {
        run: { command: serverPath, transport: TransportKind.stdio },
        debug: { command: serverPath, transport: TransportKind.stdio },
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'omc' }],
        synchronize: {
            // Re-trigger diagnostics when omc.toml changes (e.g. new
            // dependency added that the editor should now know about).
            fileEvents: vscode.workspace.createFileSystemWatcher('**/omc.toml'),
        },
    };

    client = new LanguageClient(
        'omnimcode',
        'OMNIcode Language Server',
        serverOptions,
        clientOptions,
    );

    client.start();
    context.subscriptions.push(client);
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
