// History sync scenarios against real agy: interrupt, continue, edit/undo, and switching in from another model.
// Usage: bun test/sync.ts
import type { LanguageModelV3Message, LanguageModelV3Prompt, LanguageModelV3StreamPart } from "@ai-sdk/provider"
import { mkdirSync } from "node:fs"
import { AgyLanguageModel, KIND_HEADER, SESSION_HEADER } from "../src/model.ts"
import { Runtime } from "../src/runtime.ts"

const dir = "/tmp/opencode-agy-sync"
mkdirSync(dir, { recursive: true })
const store = new Map<string, any>()
const logs: string[] = []
const runtime = new Runtime({
  directory: dir,
  storage: { get: async (k) => store.get(k), set: async (k, v) => void store.set(k, v) },
  executable: () => Bun.which("agy") ?? undefined,
  ask: async () => true,
  log: (m) => logs.push(m),
})
const lm = new AgyLanguageModel("gemini-3.8-flash", runtime)

async function step(session: string, prompt: LanguageModelV3Prompt, abortAfterMs?: number) {
  const controller = new AbortController()
  if (abortAfterMs) setTimeout(() => controller.abort(), abortAfterMs)
  const { stream } = await lm.doStream({
    prompt,
    headers: { [KIND_HEADER]: "primary", [SESSION_HEADER]: session },
    providerOptions: { agy: { effort: "low", agyAgent: "build", agyInstructions: "" } },
    abortSignal: controller.signal,
  } as any)
  const reply: LanguageModelV3Message = { role: "assistant", content: [] }
  let text = ""
  let finish = "none"
  try {
    for await (const p of stream as any as AsyncIterable<LanguageModelV3StreamPart>) {
      if (p.type === "text-delta") text += p.delta
      if (p.type === "text-end") reply.content.push({ type: "text", text, providerOptions: p.providerMetadata as any })
      if (p.type === "finish") finish = p.finishReason.unified
    }
  } catch (error) {
    finish = `error: ${error}`
  }
  if (!reply.content.length && text) reply.content.push({ type: "text", text, providerOptions: { agy: {} } as any })
  return { reply, text, finish }
}

const user = (text: string): LanguageModelV3Message => ({ role: "user", content: [{ type: "text", text }] })
const check = (name: string, ok: boolean, detail = "") => console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`)
const lastLog = () => logs.filter((l) => l.includes("sending")).at(-1) ?? ""

// 1. Interrupt a long reply, then continue in the same session.
{
  const session = "ses_sync_interrupt_" + Date.now()
  const prompt: LanguageModelV3Prompt = [user("Count from 1 to 300, one number per line, no other text.")]
  const first = await step(session, prompt, 2500)
  prompt.push(first.reply.content.length ? first.reply : { role: "assistant", content: [{ type: "text", text: "1" }] })
  check("interrupt stops the turn", first.finish !== "stop", `finish=${first.finish}, got ${first.text.length} chars`)
  prompt.push(user("Stop counting. Reply with just: ready"))
  const second = await step(session, prompt)
  check("continue after interrupt", /ready/i.test(second.text) && lastLog().includes("continue"), `${JSON.stringify(second.text.slice(0, 40))} | ${lastLog()}`)
}

// 2. Edit/undo: earlier history changes, so agy starts fresh with the kept conversation.
{
  const session = "ses_sync_edit_" + Date.now()
  const prompt: LanguageModelV3Prompt = [user("My favourite colour is teal. Reply with just: noted")]
  const a = await step(session, prompt)
  prompt.push(a.reply)
  // The user edits the first message
  prompt[0] = user("My favourite colour is orange. Reply with just: noted")
  prompt.push(user("What is my favourite colour? One word."))
  const b = await step(session, prompt)
  check("edited history starts fresh with a checkpoint", /orange/i.test(b.text) && lastLog().includes("fresh"), `${JSON.stringify(b.text.slice(0, 40))} | ${lastLog()}`)
}

// 3. Switching in from another model: its replies arrive as a checkpoint once, then turns continue.
{
  const session = "ses_sync_switch_" + Date.now()
  const prompt: LanguageModelV3Prompt = [
    user("The project codename is HERON."),
    { role: "assistant", content: [{ type: "text", text: "Got it, the codename is HERON." }] },
    user("What is the project codename? One word."),
  ]
  const a = await step(session, prompt)
  check("foreign history is caught up", /heron/i.test(a.text), `${JSON.stringify(a.text.slice(0, 40))} | ${lastLog()}`)
  prompt.push(a.reply, user("Spell it backwards. One word."))
  const b = await step(session, prompt)
  check("next turn only sends the new message", /noreh/i.test(b.text) && lastLog().includes("continue"), `${JSON.stringify(b.text.slice(0, 40))} | ${lastLog()}`)
}

await runtime.close()
process.exit(0)
