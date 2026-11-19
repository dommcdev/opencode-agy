# opencode-agy

Use Google's [Antigravity CLI](https://antigravity.google/docs/cli) (`agy`) as an OpenCode provider, with your own
Antigravity sign-in and plan.

OpenCode runs your installed `agy` through its official [Agent Client Protocol](https://antigravity.google/docs/ide/extensions/acp)
server, `agy acp`. That is the same integration Google documents for Zed, JetBrains, Android Studio and Xcode, and the
one t3code ships. The plugin never reads agy's credentials, never talks to Google itself, and never puts anything
between agy and Google.

## Install

1. Install Antigravity CLI and sign in once by running `agy` (or `agy acp --login`).
2. Add the plugin to your OpenCode config (`~/.config/opencode/opencode.json` or a project's `opencode.json`):

   ```jsonc
   {
     "$schema": "https://opencode.ai/config.json",
     "plugins": ["/path/to/opencode-agy"]
   }
   ```

3. Run `/connect` and choose **Antigravity CLI** (or `opencode auth login agy --method agy-login`). Like OpenCode's
   Claude Agent, this only records the choice; the login belongs to `agy`.
4. Pick an `agy/…` model. Variants are Antigravity's thinking levels (`agy/gemini-4-neon#high`, …).

The model list comes from your account, including Claude and GPT models when your plan has them.

## How it works

Each OpenCode session maps to one ACP session in one long-lived `agy acp` process.

- **agy is the agent.** It keeps its own system prompt, its built-in tools, its slash commands and its model
  behaviour, exactly as in the terminal. Its tool calls (`read_file`, `run_command`, `edit_file`, …) show up in
  OpenCode as tool parts that agy ran itself.
- **OpenCode is the client.** It shows replies and thinking, keeps the history, and answers agy's permission prompts.
- **One permission system.** When agy asks before running a command or editing a file, the plugin evaluates
  OpenCode's own permission rules (actions `shell`, `edit`, `read`, `webfetch` and `websearch`). Those rules prompt
  you when they say `ask`. OpenCode's Plan agent also switches agy to its plan mode.
- **Isolated config.** agy runs with a private home (`~/.local/share/opencode-agy/gemini`), so your `~/.gemini`
  settings, global rules, skills, hooks and MCP servers don't apply. This is the same idea as Claude Agent ignoring
  `~/.claude`. Your OpenCode instruction files (AGENTS.md and `instructions`) are appended to agy's prompt instead.
- **History stays in OpenCode.** agy only receives input it hasn't seen. A normal turn sends just the new message. If
  the history changes under it (edit, undo, or a switch from another provider), agy starts a new session that gets the
  kept conversation once, inside a `<conversation-checkpoint>`, because agy can't rewind or fork. A plugin or server
  restart resumes the same agy session with `session/resume`.
- **Titles come from agy.** agy already titles every conversation, so OpenCode reuses that title instead of spending
  a model request on one.
- **No double retries.** agy retries failed requests itself, so OpenCode's retry is off for this provider.

## Why not the Claude Agent design?

OpenCode's Claude Agent turns off Claude Code's tools and gives Claude OpenCode's tools and prompt instead. The first
version of this plugin did the same for agy: a custom agent with `tools: []`, OpenCode's tools over MCP, and
OpenCode's system prompt. It worked in isolation. In real OpenCode, though, every request failed with
`RESOURCE_EXHAUSTED (429)`.

The cause turned out to be one string from OpenCode v2's system prompt, `tools.opencode.session_move` (from the Code
Mode catalog). When it appears in a system prompt, Google's gateway rejects the request:

- It fails on every model, including Claude through agy.
- It fails in a plain `agy acp` with your normal `~/.gemini` home and the default agent.
- `hello`, `tools.opencode`, other `tools.opencode.*` names, and even the same string inside a user message all pass.

That looks like a deliberate fingerprint of OpenCode's harness, which fits the
[reported bans](https://discuss.ai.google.dev/t/antigravity-account-disabled-violation-of-terms-of-service-requesting-support/123014)
of accounts that used OpenCode with Antigravity quota. Working around it would mean disguising OpenCode's prompt,
which is exactly the kind of thing that gets accounts banned. So this plugin never sends OpenCode's harness prompt or
tools; agy runs as agy.

## Ban safety

- It only uses the documented `agy acp` server, with the user's own `agy` and sign-in.
- It identifies honestly as `opencode` (`clientInfo`). agy 1.5.1 and later offer every model to every ACP client, so
  nothing needs to pretend to be Zed.
- It never reads agy's tokens, calls Google endpoints, or spoofs user agents.
- It doesn't send OpenCode's system prompt, Code Mode catalog or tool definitions.
- It doesn't disable telemetry or change anything else about how agy talks to Google.

## Usage parity

These checks were run on Gemini 3.8 Flash (Low), using agy's own per-turn token reports:

| Test | Plain `agy acp` | Through OpenCode |
| --- | --- | --- |
| One-call prompt ("Reply with just: hi"), input tokens | 9,018 / 9,020 | 9,030 / 9,034 |
| Follow-up turn in the same session | new message only | new message only (56–78 chars sent) |
| Session title | free (agy) | free (agy's title is reused; no title request) |

The +14 tokens is the appended AGENTS.md (916 characters). Plain agy reads the same text from `~/.gemini/GEMINI.md`.
Multi-turn sessions keep one agy session alive, so agy's prompt caching works as in the terminal. `agy -p /usage`
stayed at 99% of the five-hour Gemini window through all of development and testing.

## Limitations

- OpenCode's own tools (subagents, todo, skills, MCP servers, Code Mode) aren't available to agy models. agy uses its
  own equivalents.
- agy's multiple-choice questions arrive as permission prompts that OpenCode can't display. They are cancelled, and
  the model asks in its reply instead.
- Editing or undoing earlier messages costs one resend of the kept conversation, since agy can't rewind.
- Compaction and other side requests run in a throwaway agy session. OpenCode's system prompt is not sent there
  either.
- After changing the plugin's code, run `opencode service restart`. The background server caches loaded modules.

## Configuration

| Variable | Purpose |
| --- | --- |
| `OPENCODE_AGY_PATH` | Path to `agy` (defaults to `agy` on `PATH`) |
| `OPENCODE_AGY_HOME` | Private Antigravity home (defaults to `$XDG_DATA_HOME/opencode-agy/gemini`) |
| `OPENCODE_AGY_DEBUG` | Log what each turn sends and agy's token usage |
| `OPENCODE_AGY_DUMP` | Directory to dump each request OpenCode makes, for debugging |

## Tests

These run against your real `agy`, on Gemini 3.8 Flash (Low):

```sh
bun test/drive.ts   # tool calls, permissions, continue path
bun test/sync.ts    # interrupt, edit/undo, switching from another model
```

## License

MIT
