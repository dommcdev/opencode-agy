import { createHash } from "node:crypto"
import type { LanguageModelV3FunctionTool } from "@ai-sdk/provider"

export const SERVER = "opencode"
// agy names MCP tools `mcp_<server>_<tool>`
export const PREFIX = `mcp_${SERVER}_`
// Gemini function names are limited to 64 characters
const LIMIT = 64 - PREFIX.length

export function short(name: string) {
  if (name.length <= LIMIT && /^[a-zA-Z_][a-zA-Z0-9_-]*$/.test(name)) return name
  const hash = createHash("sha256").update(name).digest("hex").slice(0, 8)
  return name.replace(/[^a-zA-Z0-9_-]/g, "_").replace(/^[^a-zA-Z_]/, "_").slice(0, LIMIT - 9) + "_" + hash
}

/** The opencode tool for an MCP tool name, or undefined for a tool we did not offer. */
export function original(name: string, tools: readonly LanguageModelV3FunctionTool[]) {
  const bare = name.startsWith(PREFIX) ? name.slice(PREFIX.length) : name
  return tools.find((tool) => short(tool.name) === bare)?.name
}
