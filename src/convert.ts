// opencode prompt parts → ACP content blocks, and opencode tool results → MCP tool results.

import type {
  LanguageModelV3FilePart,
  LanguageModelV3Message,
  LanguageModelV3Prompt,
  LanguageModelV3ToolResultOutput,
  LanguageModelV3ToolResultPart,
} from "@ai-sdk/provider"
import * as Fingerprint from "./sync/fingerprint.ts"
import type { CallToolResult } from "./tools/server.ts"

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
        const mime = match[1] ?? part.mediaType
        const data = match[2] ? match[3]! : Buffer.from(decodeURIComponent(match[3]!)).toString("base64")
        return file({ ...part, mediaType: mime, data })
      }
    }
    return { type: "resource_link", uri: part.data.href, name, mimeType: part.mediaType }
  }
  if (IMAGE.test(part.mediaType)) return { type: "image", mimeType: part.mediaType, data: base64(part.data) }
  const uri = `file:///${encodeURIComponent(name)}`
  if (part.mediaType.startsWith("text/") || part.mediaType === "application/json") {
    const text = typeof part.data === "string" ? Buffer.from(part.data, "base64").toString("utf8") : Buffer.from(part.data).toString("utf8")
    return { type: "resource", resource: { uri, mimeType: part.mediaType, text } }
  }
  return { type: "resource", resource: { uri, mimeType: part.mediaType, blob: base64(part.data) } }
}

function stringify(value: unknown) {
  return typeof value === "string" ? value : JSON.stringify(value)
}

function outputText(output: LanguageModelV3ToolResultOutput): string {
  switch (output.type) {
    case "text":
    case "error-text":
      return output.value
    case "json":
    case "error-json":
      return JSON.stringify(output.value)
    case "execution-denied":
      return output.reason ?? "Tool execution was denied by the user."
    case "content":
      return output.value.map((part) => (part.type === "text" ? part.text : `[${part.type}]`)).join("\n")
  }
}

/** Plain messages, with role tags when the slice contains earlier turns. Attachments stay real content blocks. */
export function messages(prompt: LanguageModelV3Prompt, tags: boolean): ContentBlock[] {
  const blocks: ContentBlock[] = []
  for (const message of prompt) {
    if (message.role === "system") continue
    if (tags) blocks.push({ type: "text", text: `<${message.role}>` })
    for (const part of message.content as LanguageModelV3Message["content"] & any[]) {
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
      text: "<conversation-checkpoint>\nThe following is conversation history from OpenCode that you have not seen. Treat it as prior conversation, not a new request. Continue with the user messages after this checkpoint.",
    })
    blocks.push(...messages(missed.slice(0, end), true))
    blocks.push({ type: "text", text: "</conversation-checkpoint>" })
  }
  blocks.push(...messages(missed.slice(end), false))
  return merge(blocks)
}

/** opencode's tool output as an MCP tool result. */
export function toolResult(
  output: LanguageModelV3ToolResultOutput,
  files: readonly LanguageModelV3FilePart[] = [],
): CallToolResult {
  const content: Record<string, unknown>[] = []
  const isError = output.type === "error-text" || output.type === "error-json" || output.type === "execution-denied"
  if (output.type === "content") {
    for (const part of output.value as any[]) {
      if (part.type === "text") content.push({ type: "text", text: part.text })
      else if ((part.type === "image-data" || part.type === "media") && IMAGE.test(part.mediaType))
        content.push({ type: "image", data: part.data, mimeType: part.mediaType })
      else if (part.type === "image-data" || part.type === "file-data" || part.type === "media")
        content.push({
          type: "resource",
          resource: { uri: `file:///${encodeURIComponent(part.filename ?? "attachment")}`, mimeType: part.mediaType, blob: part.data },
        })
      else if (part.type === "image-url" || part.type === "file-url") content.push({ type: "text", text: part.url })
      else content.push({ type: "text", text: stringify(part) })
    }
  } else {
    content.push({ type: "text", text: outputText(output) })
  }
  for (const part of files) {
    const block = file(part)
    if (block.type === "image") content.push({ type: "image", data: block.data, mimeType: block.mimeType })
    else if (block.type === "resource") content.push(block)
    else if (block.type === "resource_link") content.push({ type: "text", text: block.uri })
  }
  if (content.length === 0) content.push({ type: "text", text: "" })
  return { content, isError }
}

export interface Collected {
  readonly results: Map<string, { part: LanguageModelV3ToolResultPart; result: CallToolResult }>
  readonly aborted: boolean
  /** Index after the tool messages. */
  readonly end: number
}

/** Tool errors, recorded by the `context` hook before the AI SDK bridge flattens them to plain text. */
export const toolFlags = new Map<string, { error: boolean; aborted: boolean }>()

/** Tool results opencode sent from `start`, with any lowered tool media attached to the group's last result. */
export function collect(prompt: LanguageModelV3Prompt, start: number): Collected {
  const results = new Map<string, { part: LanguageModelV3ToolResultPart; result: CallToolResult }>()
  let end = start
  let aborted = false
  for (const group of Fingerprint.groups(prompt.slice(start))) {
    if (group.role !== "tool" && group.role !== "assistant") break
    const slice = prompt.slice(start + group.start, start + group.end)
    const tools: LanguageModelV3ToolResultPart[] = []
    const files: LanguageModelV3FilePart[] = []
    for (const message of slice) {
      if (message.role === "tool") for (const part of message.content) if (part.type === "tool-result") tools.push(part)
      // An assistant message here holds results opencode keeps inline with the call.
      if (message.role === "assistant")
        for (const part of message.content) if (part.type === "tool-result") tools.push(part as LanguageModelV3ToolResultPart)
      if (message.role === "user") for (const part of message.content) if (part.type === "file") files.push(part)
    }
    if (group.role === "assistant" && tools.length === 0) break
    for (let i = 0; i < tools.length; i++) {
      const part = tools[i]!
      const flags = toolFlags.get(part.toolCallId)
      if (flags?.aborted) aborted = true
      const result = toolResult(part.output, i === tools.length - 1 ? files : [])
      results.set(part.toolCallId, { part, result: flags?.error ? { ...result, isError: true } : result })
    }
    end = start + group.end
  }
  return { results, aborted, end }
}
