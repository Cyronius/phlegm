# phlegm-tray

Windows system tray app for phlegm (Tauri v2 + React). Inference will run in a `phlegm serve` child process, so an NPU fault restarts the server, not the tray.

```sh
pnpm install
pnpm tauri dev      # run with hot reload
pnpm tauri build    # NSIS installer under src-tauri/target/release/bundle/
```
