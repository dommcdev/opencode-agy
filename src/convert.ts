// opencode prompt messages → ACP content blocks.

import type { LanguageModelV3FilePart, LanguageModelV3Prompt } from "@ai-sdk/provider"
import * as Fingerprint from "./sync/fingerprint.ts"

export type ContentBlock =
  | { type: "text"; text: string }
  | { type: "image"; mimeType: string; data: string; uri?: string }
  | { type: "resource"; resource: { uri: string; mimeType?: string; text?: string; blob?: string } }
  | { type: "resource_link"; uri: string; name: string; mimeType?: string }

const IMAGE = /^image\/(png|jpeg|jpg|gif|webp)$/

function base64(data: string | Uint8Array) {
  return typeof data === "string" ? data : Buffer.from(data).toString("base64")
}

export function file(part: LanguageModelV3FilePart): ContentBlock {
  const name = part.filename ?? "attachment"
  if (part.data instanceof URL) {
    if (part.data.protocol === "data:") {
      const match = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(part.data.href)
      if (match) {
        const data = match[2] ? match[3]! : Buffer.from(decodeURIComponent(match[3]!)).toString("base64")
        return file({ ...part, mediaType: match[1] ?? part.mediaType, data })
      }
    }
    return { type: "resource_link", uri: part.data.href, name, mimeType: part.mediaType }
  }
  if (IMAGE.test(part.mediaType)) return { type: "image", mimeType: part.mediaType, data: base64(part.data) }
  const uri = `file:///${encodeURIComponent(name)}`
  if (part.mediaType.startsWith("text/") || part.mediaType === "application/json") {
    const text =
      typeof part.data === "string" ? Buffer.from(part.data, "base64").toString("utf8") : Buffer.from(part.data).toString("utf8")
    return { type: "resource", resource: { uri, mimeType: part.mediaType, text } }
  }
  return { type: "resource", resource: { uri, mimeType: part.mediaType, blob: base64(part.data) } }
}

function outputText(output: any): string {
  if (!output) return ""
  if (output.type === "text" || output.type === "error-text") return String(output.value)
  if (output.type === "json" || output.type === "error-json") return JSON.stringify(output.value)
  if (output.type === "execution-denied") return output.reason ?? "Tool execution was denied by the user."
  if (output.type === "content")
    return (output.value as any[]).map((part) => (part.type === "text" ? part.text : `[${part.type}]`)).join("\n")
  return JSON.stringify(output)
}

/** Messages as content blocks, with role tags when the slice holds earlier turns. Attachments stay real blocks. */
export function messages(prompt: LanguageModelV3Prompt, tags: boolean): ContentBlock[] {
  const blocks: ContentBlock[] = []
  for (const message of prompt) {
    // opencode's harness system prompt is never sent; agy uses its own.
    if (message.role === "system") continue
    if (tags) blocks.push({ type: "text", text: `<${message.role}>` })
    for (const part of message.content as any[]) {
      if (part.type === "text") blocks.push({ type: "text", text: part.text })
      else if (part.type === "file") blocks.push(file(part))
      else if (part.type === "tool-call")
        blocks.push({
          type: "text",
          text: `<tool-call>\n${JSON.stringify({ id: part.toolCallId, name: part.toolName, input: part.input })}\n</tool-call>`,
        })
      else if (part.type === "tool-result")
        blocks.push({
          type: "text",
          text: `<tool-result>\n${JSON.stringify({ id: part.toolCallId, name: part.toolName, output: outputText(part.output) })}\n</tool-result>`,
        })
      // Reasoning from other models stays private to them.
    }
    if (tags) blocks.push({ type: "text", text: `</${message.role}>` })
  }
  return merge(blocks)
}

// Adjacent text blocks become one, which keeps role-tagged transcripts compact.
function merge(blocks: ContentBlock[]) {
  const result: ContentBlock[] = []
  for (const block of blocks) {
    const last = result.at(-1)
    if (block.type === "text" && last?.type === "text") {
      result[result.length - 1] = { type: "text", text: `${last.text}\n${block.text}` }
      continue
    }
    result.push(block)
  }
  return result
}

/**
 * What agy needs to continue from `start`: history it has not seen goes into one checkpoint block, and trailing user
 * input is sent as the new prompt.
 */
export function catchUp(prompt: LanguageModelV3Prompt, start: number): ContentBlock[] {
  const missed = prompt.slice(start)
  const end = Fingerprint.groups(missed).findLast((group) => group.role !== "user")?.end ?? 0
  const blocks: ContentBlock[] = []
  if (end > 0) {
    blocks.push({
      type: "text",
      text: "<conversation-checkpoint>\nThe following is earlier conversation history that you have not seen. Treat it as prior conversation, not a new request. Continue with the user messages after this checkpoint.",
    })
    blocks.push(...messages(missed.slice(0, end), true))
    blocks.push({ type: "text", text: "</conversation-checkpoint>" })
  }
  blocks.push(...messages(missed.slice(end), false))
  return merge(blocks)
}
