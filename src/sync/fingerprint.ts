// Stable hashes of opencode's history, so a session only sends agy what it has not seen.
// Adapted from opencode's Claude Agent provider (packages/core/src/plugin/provider/claude-agent/sync, MIT).

import { createHash } from "node:crypto"
import type { LanguageModelV3Message, LanguageModelV3Prompt } from "@ai-sdk/provider"

export interface Group {
  readonly hash: string
  readonly role: "user" | "assistant" | "tool"
  readonly start: number
  readonly end: number
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`
  if (value instanceof Uint8Array) return JSON.stringify(Buffer.from(value).toString("base64"))
  if (value instanceof URL) return JSON.stringify(value.href)
  if (value !== null && typeof value === "object") {
    const object = value as Record<string, unknown>
    const fields: string[] = []
    for (const key of Object.keys(object).sort()) {
      if (object[key] !== undefined) fields.push(`${JSON.stringify(key)}:${canonical(object[key])}`)
    }
    return `{${fields.join(",")}}`
  }
  return JSON.stringify(value) ?? "null"
}

export function hash(value: unknown) {
  return createHash("sha256").update(canonical(value)).digest("hex")
}

function inputValue(input: unknown) {
  if (typeof input !== "string") return input
  try {
    return JSON.parse(input)
  } catch {
    return input
  }
}

// opencode injects reminders (such as Plan mode's) for one request and moves them on the next, so they are not part
// of the conversation's identity.
const REMINDER = /^\s*<system-reminder>[\s\S]*<\/system-reminder>\s*$/

// Thinking and provider options are replayed differently depending on the target model, so they don't count.
function visible(message: Exclude<LanguageModelV3Message, { role: "system" }>) {
  const parts: unknown[] = []
  let text = ""
  for (const part of message.content) {
    if (part.type === "reasoning") continue
    if (part.type === "text" && REMINDER.test(part.text)) continue
    if (part.type === "text") {
      text += part.text
      continue
    }
    if (text) parts.push({ type: "text", text })
    text = ""
    if (part.type === "file") {
      parts.push({ type: part.type, mediaType: part.mediaType, data: part.data })
      continue
    }
    if (part.type === "tool-call") {
      parts.push({ type: part.type, id: part.toolCallId, name: part.toolName, input: inputValue(part.input) })
      continue
    }
    if (part.type === "tool-result") {
      const output: Record<string, unknown> = { ...part.output }
      delete output.providerOptions
      if (part.output.type === "content") {
        output.value = part.output.value.map((item) => {
          const copy: Record<string, unknown> = { ...item }
          delete copy.providerOptions
          return copy
        })
      }
      parts.push({ type: part.type, id: part.toolCallId, name: part.toolName, output })
    }
  }
  if (text) parts.push({ type: "text", text })
  return parts
}

export const TOOL_MEDIA = "Attached media from tool result:"

// opencode lowers tool attachments into this separate user message.
export function isToolMedia(message: LanguageModelV3Message) {
  return (
    message.role === "user" &&
    message.content.length > 1 &&
    message.content[0]?.type === "text" &&
    message.content[0].text === TOOL_MEDIA &&
    message.content.slice(1).every((part) => part.type === "file")
  )
}

export function groups(prompt: LanguageModelV3Prompt): Group[] {
  const result: Group[] = []
  let i = 0
  while (i < prompt.length) {
    const message = prompt[i]!
    const start = i
    i++
    if (message.role === "system") continue
    const parts = visible(message)
    if (parts.length === 0) continue
    const members = [hash(parts)]
    if (message.role === "user") {
      while (i < prompt.length && prompt[i]!.role === "user") {
        members.push(hash(visible(prompt[i] as Exclude<LanguageModelV3Message, { role: "system" }>)))
        i++
      }
    }
    if (message.role === "tool") {
      const next = prompt[i]
      if (next && isToolMedia(next)) {
        members.push(hash(visible(next as Exclude<LanguageModelV3Message, { role: "system" }>)))
        i++
      }
    }
    result.push({ hash: hash({ role: message.role, members }), role: message.role, start, end: i })
  }
  return result
}
