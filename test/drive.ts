// Drives the runtime the way opencode does: stream a step, run its tool calls, send the results back, repeat.
// Usage: bun test/drive.ts [model] [effort]
import type { LanguageModelV3Message, LanguageModelV3Prompt, LanguageModelV3StreamPart } from "@ai-sdk/provider"
import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { AgyLanguageModel, KIND_HEADER, SESSION_HEADER } from "../src/model.ts"
import { Runtime } from "../src/runtime.ts"

const model = process.argv[2] ?? "gemini-3.8-flash"
const effort = process.argv[3] ?? "low"
const dir = process.env.DRIVE_DIR ?? "/tmp/opencode-agy-drive"
mkdirSync(dir, { recursive: true })
writeFileSync(join(dir, "secret.txt"), "The secret word is PAPAYA.\n")
writeFileSync(join(dir, "other.txt"), "The other word is MANGO.\n")

const store = new Map<string, any>()
const runtime = new Runtime({
  directory: dir,
  storage: { get: async (k) => store.get(k), set: async (k, v) => void store.set(k, v) },
  executable: () => Bun.which("agy") ?? undefined,
  log: (m) => console.log("  [runtime]", m),
})
const lm = new AgyLanguageModel(model, runtime)
const tools = [
  {
    type: "function" as const,
    name: "read",
    description: "Read a file. filePath is relative to the working directory.",
    inputSchema: { type: "object", properties: { filePath: { type: "string" } }, required: ["filePath"] } as any,
  },
  {
    type: "function" as const,
    name: "list",
    description: "List files in the working directory.",
    inputSchema: { type: "object", properties: {} } as any,
  },
]
const run = (name: string, input: any) => {
  if (name === "read") return readFileSync(join(dir, input.filePath), "utf8")
  if (name === "list") return "secret.txt\nother.txt"
  return "unknown tool"
}

const prompt: LanguageModelV3Prompt = [{ role: "system", content: "You are a concise assistant in a test harness. Working directory: " + dir }]
const session = "ses_drive_" + Date.now()

async function step() {
  const { stream } = await lm.doStream({
    prompt,
    tools,
    headers: { [KIND_HEADER]: "primary", [SESSION_HEADER]: session },
    providerOptions: { agy: { effort } },
  } as any)
  const parts: LanguageModelV3StreamPart[] = []
  for await (const part of stream as any as AsyncIterable<LanguageModelV3StreamPart>) parts.push(part)
  return parts
}

async function turn(text: string) {
  prompt.push({ role: "user", content: [{ type: "text", text }] })
  for (let i = 0; i < 10; i++) {
    const t0 = Date.now()
    const parts = await step()
    const finish = parts.find((p) => p.type === "finish") as any
    const assistant: LanguageModelV3Message = { role: "assistant", content: [] }
    let textOut = ""
    for (const p of parts) {
      if (p.type === "text-delta") textOut += p.delta
      if (p.type === "text-end" && textOut) {
        assistant.content.push({ type: "text", text: textOut, providerOptions: p.providerMetadata as any })
        textOut = ""
      }
      if (p.type === "tool-call")
        assistant.content.push({ type: "tool-call", toolCallId: p.toolCallId, toolName: p.toolName, input: JSON.parse(p.input), providerOptions: p.providerMetadata as any })
      if (p.type === "error") console.log("  ERROR", p.error)
    }
    const calls = assistant.content.filter((p) => p.type === "tool-call") as any[]
    console.log(`  step ${i}: ${finish?.finishReason?.unified} in ${Date.now() - t0}ms, calls=${calls.map((c) => `${c.toolName}(${JSON.stringify(c.input)})`).join(",")} usage=${JSON.stringify(finish?.usage)}`)
    const said = assistant.content.filter((p) => p.type === "text").map((p: any) => p.text).join("")
    if (said) console.log("  text:", said.slice(0, 300))
    if (assistant.content.length) prompt.push(assistant)
    if (finish?.finishReason?.unified !== "tool-calls") return said
    prompt.push({
      role: "tool",
      content: calls.map((c) => ({ type: "tool-result", toolCallId: c.toolCallId, toolName: c.toolName, output: { type: "text", value: run(c.toolName, c.input) } })),
    })
  }
}

console.log("turn 1")
await turn("List the files, then read every file and tell me both words.")
console.log("turn 2 (continue: agy should only get this message)")
await turn("Which word came from secret.txt? One word answer.")
console.log("record:", JSON.stringify([...store.entries()]).slice(0, 300))
await runtime.close()
process.exit(0)
