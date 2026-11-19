// A private Antigravity home (`agy acp --gemini-dir`). agy reads its settings, global rules (GEMINI.md), skills,
// hooks, MCP servers and plugins from here, so the user's ~/.gemini setup doesn't leak into opencode sessions (the
// same idea as opencode's Claude Agent passing `settingSources: []`). opencode supplies the user's instructions
// instead. The Google sign-in is not stored here, so the user's existing `agy` login keeps working.

import { mkdirSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"

export function directory(env: NodeJS.ProcessEnv = process.env) {
  if (env.OPENCODE_AGY_HOME) return env.OPENCODE_AGY_HOME
  const data = env.XDG_DATA_HOME || join(homedir(), ".local", "share")
  return join(data, "opencode-agy", "gemini")
}

export function prepare(env: NodeJS.ProcessEnv = process.env) {
  const root = directory(env)
  mkdirSync(root, { recursive: true })
  return root
}
