---
name: fastkernel
description: Install, start, connect and stop fastkernel, the local Qwen3.8-27B and Qwen3.6-35B-A3B server for Apple Silicon. Use when the user wants to run fastkernel or point a coding agent (OpenCode, Claude Code, Codex, Hermes) at it.
---

# fastkernel

fastkernel serves Qwen3.8-27B or Qwen3.6-35B-A3B on one Apple silicon Mac, with the OpenAI and Anthropic APIs on
`http://127.0.0.1:8000`. Commands below run from the fastkernel folder (the clone of
https://github.com/loopai-hq/fastkernel). If you don't know where it is, ask the user.

## 1. Check the Mac

- Apple silicon (M3 or newer), macOS 26.4 or later, Python 3.12 to 3.14. Xcode with its Metal tools only to build
  from source.
- 24 GB Mac: before starting, ask the user to run `sudo sysctl iogpu.wired_limit_mb=20480` themselves (it resets on
  reboot). Never run sudo for them. Then add `SPLASH_TEXT_ONLY=1` in front of the serve command in step 3.

## 2. Install (once per model)

```bash
make install MODEL=incoai/Qwen3.8-27B-Splash
```

The first run downloads the model (17.4 GB). For Qwen3.6-35B-A3B, use `MODEL=incoai/Qwen3.6-35B-A3B-Splash`.

No Xcode? Use the prebuilt package instead: download `fastkernel-1.1.3-macos-arm64.tar.gz` from
https://github.com/loopai-hq/fastkernel/releases, run `tar -xzf` on it, and work in the `fastkernel` folder it
makes. If `./splash` says macOS quarantined the download, run the `xattr` command it prints. The model downloads on
the first `serve`.

## 3. Start the server

Run it in its own terminal or as a background process, and leave it running:

```bash
SPLASH_DRAFT_HEAD_IDS=$PWD/data/head-ranked.u32 ./splash serve --model incoai/Qwen3.8-27B-Splash
```

- It's ready when it prints `Ready` and `curl -s http://127.0.0.1:8000/status` shows `"ready": true`.
- The defaults are the tuned settings. Don't add other `SPLASH_` variables.
- `--port 8080` picks another port. `--max-context 128K` sets the context limit (the default is automatic).
- If startup prints a memory breakdown and stops, the model doesn't fit: close apps, or on a 24 GB Mac do the steps in
  section 1.

## 4. Connect a coding agent

With the server running, from the fastkernel folder:

```bash
./splash opencode    # or: ./splash claude / ./splash codex / ./splash hermes
```

- The agent starts pointed at fastkernel, with the model and context limit set.
- Arguments after `--` go to the agent, e.g. `./splash claude -- -p "hello"`.
- Set `SPLASH_PORT` if the server isn't on port 8000.

Any other app: OpenAI API at `http://127.0.0.1:8000/v1`, Anthropic API at `http://127.0.0.1:8000`, model
`incoai/Qwen3.8-27B-Splash`. Any API key works unless the server was started with `SPLASH_API_KEY`.

## 5. Stop

Press Ctrl+C in the server terminal, or stop the background process you started.
