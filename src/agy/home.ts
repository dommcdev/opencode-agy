// A private Antigravity home (`agy acp --gemini-dir`). agy reads its settings, rules (GEMINI.md), skills, hooks, MCP
// servers, plugins and custom agents from here, so the user's ~/.gemini setup never leaks into opencode sessions, the
// same way opencode's Claude Agent passes `settingSources: []`. The Google sign-in is not stored here, so the user's
// existing `agy` login keeps working.

import { mkdirSync, writeFileSync, readFileSync, existsSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"

export const AGENT = "opencode"

// `tools: []` removes every built-in Antigravity tool, so the model only sees opencode's tools (served over MCP).
// The body becomes the base of agy's system prompt; opencode's own prompt is appended per session through
// `_meta.antigravity.systemPrompt.append`.
const AGENT_FILE = `---
name: ${AGENT}
description: Main agent for OpenCode. OpenCode supplies the instructions and every tool.
tools: []
mainAgent: true
subagent: false
---
You are running inside OpenCode, an open source AI coding agent, through the Antigravity CLI harness. OpenCode provides your instructions and all of your tools; use only the tools you are given.
`

export function directory(env: NodeJS.ProcessEnv = process.env) {
  if (env.OPENCODE_AGY_HOME) return env.OPENCODE_AGY_HOME
  const data = env.XDG_DATA_HOME || join(homedir(), ".local", "share")
  return join(data, "opencode-agy", "gemini")
}

export function prepare(env: NodeJS.ProcessEnv = process.env) {
  const root = directory(env)
  const agents = join(root, "config", "agents")
  mkdirSync(agents, { recursive: true })
  const file = join(agents, `${AGENT}.md`)
  if (!existsSync(file) || readFileSync(file, "utf8") !== AGENT_FILE) writeFileSync(file, AGENT_FILE)
  return root
}
