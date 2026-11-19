// Drives the runtime the way opencode does, outside opencode. Usage: bun test/drive.ts [model] [effort]
import type { LanguageModelV3Message, LanguageModelV3Prompt, LanguageModelV3StreamPart } from "@ai-sdk/provider"
import { mkdirSync, writeFileSync } from "node:fs"
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
const asked: string[] = []
const runtime = new Runtime({
  directory: dir,
  storage: { get: async (k) => store.get(k), set: async (k, v) => void store.set(k, v) },
  executable: () => Bun.which("agy") ?? undefined,
  ask: async (input) => {
    asked.push(`${input.action}(${input.resources.join(", ")})`)
    console.log(`  [permission] ${input.action} ${JSON.stringify(input.resources)} -> allow`)
    return true
  },
  log: (m) => console.log("  [runtime]", m),
})
const lm = new AgyLanguageModel(model, runtime)
const prompt: LanguageModelV3Prompt = [{ role: "system", content: "(opencode harness prompt: never sent to agy)" }]
const session = "ses_drive_" + Date.now()

async function turn(text: string) {
  prompt.push({ role: "user", content: [{ type: "text", text }] })
  const t0 = Date.now()
  const { stream } = await lm.doStream({
    prompt,
    headers: { [KIND_HEADER]: "primary", [SESSION_HEADER]: session },
    providerOptions: { agy: { effort, agyAgent: "build", agyInstructions: "Always answer in English." } },
  } as any)
  const assistant: LanguageModelV3Message = { role: "assistant", content: [] }
  let said = ""
  for await (const p of stream as any as AsyncIterable<LanguageModelV3StreamPart>) {
    if (p.type === "text-delta") said += p.delta
    if (p.type === "text-end" && said) {
      assistant.content.push({ type: "text", text: said, providerOptions: p.providerMetadata as any })
      console.log("  text:", said.slice(0, 300).replace(/\n/g, " | "))
      said = ""
    }
    if (p.type === "tool-call") {
      console.log(`  tool-call ${p.toolName} ${p.input.slice(0, 160)} providerExecuted=${p.providerExecuted}`)
      assistant.content.push({ type: "tool-call", toolCallId: p.toolCallId, toolName: p.toolName, input: JSON.parse(p.input), providerExecuted: true, providerOptions: p.providerMetadata as any })
    }
    if (p.type === "tool-result") {
      console.log(`  tool-result ${p.toolName} error=${!!(p as any).isError} ${JSON.stringify((p as any).result).slice(0, 160)}`)
      assistant.content.push({ type: "tool-result", toolCallId: p.toolCallId, toolName: p.toolName, output: { type: "json", value: (p as any).result }, providerOptions: p.providerMetadata as any } as any)
    }
    if (p.type === "finish") console.log(`  finish ${p.finishReason.unified} in ${Date.now() - t0}ms usage=${JSON.stringify(p.usage)}`)
    if (p.type === "error") console.log("  ERROR", p.error)
  }
  prompt.push(assistant)
}

console.log("turn 1")
await turn("Read secret.txt and other.txt and tell me both words. Then run the shell command `ls` and show its output.")
console.log("turn 2 (continue: agy should only get this message)")
await turn("Which word came from secret.txt? One word answer.")
console.log("permissions asked:", asked.join("; "))
await runtime.close()
process.exit(0)
