// Runs opencode model requests through `agy acp`, Antigravity CLI's official Agent Client Protocol server.
//
// agy stays the agent: its own system prompt, tools, slash commands and model behaviour, exactly as in the terminal
// or in the editors Google supports (Zed, JetBrains, t3code). opencode is the client: it shows agy's replies,
// thinking and tool calls, answers agy's permission prompts through opencode's own permission rules, and keeps the
// session history. opencode's own system prompt and tools are never sent to agy.
//
// - One `agy acp` process serves the whole plugin instance; each opencode session maps to one ACP session.
// - agy runs with a private home, so ~/.gemini settings, rules, skills, hooks and MCP servers don't apply.
// - opencode's history stays the source of truth. agy only receives input it has not seen.

import type { LanguageModelV3Prompt, LanguageModelV3StreamPart } from "@ai-sdk/provider"
import { tmpdir } from "node:os"
import { AcpError, AcpProcess, type PermissionRequest, type SessionUpdate } from "./agy/acp.ts"
import * as Home from "./agy/home.ts"
import * as Convert from "./convert.ts"
import * as Fingerprint from "./sync/fingerprint.ts"

export const PROVIDER_ID = "agy"
// The live agy session of an opencode session closes after this long without requests; the next one resumes it.
const IDLE_MS = 10 * 60_000

export type Kind = "primary" | "title" | "generate" | "compaction"

export interface Request {
  readonly kind: Kind | undefined
  readonly sessionID: string | undefined
  readonly agent: string | undefined
  readonly model: string
  readonly effort: string | undefined
  /** The user's own instruction files (AGENTS.md and configured instructions), appended to agy's prompt. */
  readonly instructions: string
  readonly prompt: LanguageModelV3Prompt
  readonly emit: (part: LanguageModelV3StreamPart) => void
  readonly signal: AbortSignal
}

/** Asks opencode whether agy may do something. Resolves true to allow. */
export type Ask = (input: {
  readonly sessionID: string
  readonly agent: string | undefined
  readonly action: string
  readonly resources: string[]
  readonly save?: string[]
  readonly metadata?: Record<string, unknown>
}) => Promise<boolean>

interface SessionRecord {
  acpID: string
  // Fingerprints of the opencode history this ACP session has seen
  history: string[]
}

type TurnEvent =
  | { type: "update"; update: SessionUpdate }
  | { type: "abort" }
  | { type: "end"; result?: any; error?: unknown }

interface Turn {
  readonly events: TurnEvent[]
  wake?: () => void
  ended: boolean
  readonly done: PromiseWithResolvers<void>
}

interface Live {
  acpID: string
  sessionID?: string
  agent?: string
  instructionsHash: string
  model?: string
  effort?: string
  mode?: string
  turn?: Turn
  title?: string
  used?: number
  unsubscribe: () => void
}

interface Entry {
  lock: Promise<void>
  record?: SessionRecord
  live?: Live
  idle?: ReturnType<typeof setTimeout>
  readonly titleWaiters: Array<(title: string) => void>
}

export interface Storage {
  get(key: string): Promise<unknown>
  set(key: string, value: any): Promise<void>
}

export interface Options {
  readonly directory: string
  readonly storage: Storage
  readonly executable: () => string | undefined
  readonly ask: Ask
  readonly env?: NodeJS.ProcessEnv
  readonly log?: (message: string) => void
}

export class Runtime {
  private process?: AcpProcess
  private starting?: Promise<AcpProcess>
  private readonly entries = new Map<string, Entry>()
  // ACP session id → live session, for permission prompts
  private readonly byACP = new Map<string, Live>()
  private home?: string

  constructor(private readonly options: Options) {}

  // ---------------------------------------------------------------- process

  async agy(): Promise<AcpProcess> {
    if (this.process && !this.process.ended) return this.process
    if (this.starting) return this.starting
    this.starting = (async () => {
      const executable = this.options.executable()
      if (!executable)
        throw new Error(
          "Antigravity CLI (agy) is not installed. Install it from https://antigravity.google/docs/install, then run `agy` once to sign in.",
        )
      this.home ??= Home.prepare(this.options.env)
      const env = { ...(this.options.env ?? process.env) }
      // Credential variables would silently switch agy away from the user's Google sign-in.
      for (const name of ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_APPLICATION_CREDENTIALS"])
        delete env[name]
      const agy = new AcpProcess({
        executable,
        args: ["acp", "--gemini-dir", this.home],
        cwd: this.options.directory,
        env,
        permission: (request) => this.permission(request),
      })
      try {
        await agy.start()
      } catch (error) {
        agy.kill()
        throw error
      }
      agy.exited.then(() => this.onExit(agy))
      this.process = agy
      return agy
    })().finally(() => {
      this.starting = undefined
    })
    return this.starting
  }

  private onExit(agy: AcpProcess) {
    if (this.process !== agy) return
    this.process = undefined
    this.options.log?.("agy acp exited")
    for (const entry of this.entries.values()) {
      const live = entry.live
      if (!live) continue
      this.endTurn(live, { error: new Error("agy acp exited unexpectedly") })
      this.dispose(entry, live)
    }
  }

  // agy asks before edits and commands. opencode's permission rules decide, and prompt the user when they say "ask".
  private async permission(request: PermissionRequest): Promise<string | undefined> {
    const live = this.byACP.get(request.sessionId)
    const allow = request.options.find((option) => option.kind === "allow_once") ?? request.options.find((o) => o.kind === "allow_always")
    const reject = request.options.find((option) => option.kind === "reject_once") ?? request.options.find((o) => o.kind === "reject_always")
    // agy's questions arrive as permission prompts with one option per answer. opencode can't show those choices,
    // so they are cancelled and the model asks in its reply instead.
    if (request.toolCall?.toolCallId?.startsWith("interaction_")) return undefined
    if (!live?.sessionID) return reject?.optionId
    const ask = permissionFor(request.toolCall, this.options.directory)
    try {
      const allowed = await this.options.ask({ sessionID: live.sessionID, agent: live.agent, ...ask })
      return allowed ? allow?.optionId : reject?.optionId
    } catch (error) {
      this.options.log?.(`permission check failed: ${error}`)
      return reject?.optionId
    }
  }

  // ---------------------------------------------------------------- catalog

  /** Lists the account's models and thinking levels. Only reads session options; no prompt is sent. */
  async probe() {
    const agy = await this.agy()
    const created = await agy.request("session/new", { cwd: tmpdir(), mcpServers: [] })
    const sessionId = created.sessionId
    let size: number | undefined
    const unsubscribe = agy.subscribe(sessionId, (update) => {
      if (update.sessionUpdate === "usage_update" && typeof update.size === "number") size = update.size
    })
    try {
      const modelOption = (created.configOptions ?? []).find((option: any) => option.id === "model")
      const models: Array<{ id: string; name: string; efforts: string[]; context?: number }> = []
      for (const choice of flatten(modelOption?.options ?? [])) {
        const changed = await agy.request("session/set_config_option", { sessionId, configId: "model", value: choice.value })
        const thought = (changed?.configOptions ?? []).find((option: any) => option.id === "thought_level")
        models.push({
          id: choice.value,
          name: choice.name ?? choice.value,
          efforts: flatten(thought?.options ?? []).map((option) => option.value),
          context: size,
        })
      }
      return { version: agy.version, tier: created._meta?.antigravity?.tier?.name as string | undefined, models }
    } finally {
      unsubscribe()
      await agy.request("session/close", { sessionId }).catch(() => {})
    }
  }

  // ---------------------------------------------------------------- requests

  async run(request: Request) {
    if (request.kind !== "primary" || !request.sessionID) return this.once(request)
    const entry = this.entry(request.sessionID)
    const previous = entry.lock
    let release!: () => void
    entry.lock = new Promise<void>((resolve) => (release = resolve))
    await previous
    try {
      if (entry.idle) clearTimeout(entry.idle)
      await this.primary(entry, request)
    } finally {
      this.armIdle(entry)
      release()
    }
  }

  /** The title agy generated for this session, if it does so within the timeout. */
  title(sessionID: string, timeoutMs: number) {
    const entry = this.entry(sessionID)
    if (entry.live?.title) return Promise.resolve(entry.live.title)
    return new Promise<string | undefined>((resolve) => {
      const timer = setTimeout(() => resolve(undefined), timeoutMs)
      entry.titleWaiters.push((title) => {
        clearTimeout(timer)
        resolve(title)
      })
    })
  }

  async close() {
    for (const entry of this.entries.values()) {
      if (entry.idle) clearTimeout(entry.idle)
      await this.closeLive(entry).catch(() => {})
    }
    this.process?.kill()
  }

  private entry(sessionID: string) {
    let entry = this.entries.get(sessionID)
    if (!entry) {
      entry = { lock: Promise.resolve(), titleWaiters: [] }
      this.entries.set(sessionID, entry)
    }
    return entry
  }

  private armIdle(entry: Entry) {
    if (entry.idle) clearTimeout(entry.idle)
    entry.idle = setTimeout(() => {
      entry.idle = undefined
      void this.closeLive(entry)
    }, IDLE_MS)
    entry.idle.unref?.()
  }

  private dispose(entry: Entry, live: Live) {
    live.unsubscribe()
    this.byACP.delete(live.acpID)
    if (entry.live === live) entry.live = undefined
  }

  private async closeLive(entry: Entry) {
    const live = entry.live
    if (!live) return
    await this.cancelTurn(live)
    this.dispose(entry, live)
    await this.process?.request("session/close", { sessionId: live.acpID }).catch(() => {})
  }

  private async loadRecord(sessionID: string): Promise<SessionRecord | undefined> {
    const saved = (await this.options.storage.get(`sessions/${sessionID}`).catch(() => undefined)) as SessionRecord | undefined
    if (saved && typeof saved.acpID === "string" && Array.isArray(saved.history)) return saved
    return undefined
  }

  /**
   * Where the input agy has not seen starts, or undefined when opencode's history no longer extends what agy saw
   * (an edit, an undo, or a history another provider wrote).
   */
  private unseen(prompt: LanguageModelV3Prompt, groups: Fingerprint.Group[], record: SessionRecord | undefined) {
    if (!record || record.history.length === 0) return undefined
    let matched = 0
    while (matched < record.history.length && matched < groups.length && record.history[matched] === groups[matched]!.hash)
      matched++
    if (matched < record.history.length) return undefined
    // Replies this ACP session wrote are already in agy's history.
    while (matched < groups.length && groups[matched]!.role === "assistant" && this.ours(prompt, groups[matched]!, record.acpID))
      matched++
    return groups[matched]?.start ?? prompt.length
  }

  // Assistant messages carry the ACP session that wrote them in their provider metadata.
  private ours(prompt: LanguageModelV3Prompt, group: Fingerprint.Group, acpID: string) {
    for (const message of prompt.slice(group.start, group.end)) {
      if (message.role !== "assistant") continue
      for (const part of message.content as any[]) if (part.providerOptions?.[PROVIDER_ID]?.session === acpID) return true
    }
    return false
  }

  private async primary(entry: Entry, request: Request) {
    const sessionID = request.sessionID!
    entry.record ??= await this.loadRecord(sessionID)
    const groups = Fingerprint.groups(request.prompt)
    let start = this.unseen(request.prompt, groups, entry.record)
    if (start === undefined) {
      // agy cannot rewind or fork a conversation, so a changed history starts a new session that receives the kept
      // conversation once.
      if (entry.record) this.options.log?.(`history changed for ${sessionID}; starting a new agy session`)
      await this.closeLive(entry)
      entry.record = undefined
    }
    const live = await this.ensureLive(entry, request)
    if (entry.record?.acpID !== live.acpID) start = 0
    if (live.turn && !live.turn.ended) await this.cancelTurn(live)

    let content = Convert.catchUp(request.prompt, start ?? 0)
    if (content.length === 0) content = [{ type: "text", text: "Continue." }]
    entry.record = { acpID: live.acpID, history: groups.map((group) => group.hash) }
    await this.options.storage.set(`sessions/${sessionID}`, entry.record).catch(() => {})
    this.startTurn(live, content)
    await this.readTurn(live, request)
  }

  private async ensureLive(entry: Entry, request: Request): Promise<Live> {
    const agy = await this.agy()
    const instructionsHash = Fingerprint.hash(request.instructions)
    // New instructions need the session reopened; resume keeps agy's history.
    if (entry.live && entry.live.instructionsHash !== instructionsHash) await this.closeLive(entry)
    let live = entry.live
    if (!live) {
      const created: Live = { acpID: "", instructionsHash, unsubscribe: () => {} }
      const params = {
        cwd: this.options.directory,
        mcpServers: [],
        ...(request.instructions ? { _meta: { antigravity: { systemPrompt: { append: request.instructions } } } } : {}),
      }
      const resumeID = entry.record?.acpID
      if (resumeID) {
        // Route updates before agy sends any for the resumed session.
        created.unsubscribe = agy.subscribe(resumeID, (update) => this.onUpdate(entry, created, update))
        const resumed = await agy.request("session/resume", { sessionId: resumeID, ...params }).then(
          () => true,
          (error) => {
            created.unsubscribe()
            if (error instanceof AcpError && error.isAuth) throw error
            this.options.log?.(`could not resume agy session ${resumeID}: ${error}`)
            return false
          },
        )
        if (resumed) created.acpID = resumeID
      }
      if (!created.acpID) {
        const result = await agy.request("session/new", params)
        created.acpID = result.sessionId
        created.unsubscribe = agy.subscribe(created.acpID, (update) => this.onUpdate(entry, created, update))
      }
      this.byACP.set(created.acpID, created)
      entry.live = created
      live = created
    }
    live.sessionID = request.sessionID
    live.agent = request.agent
    await this.configure(agy, live, request)
    return live
  }

  private async configure(agy: AcpProcess, live: Live, request: Request) {
    if (live.model !== request.model) {
      await agy.request("session/set_config_option", { sessionId: live.acpID, configId: "model", value: request.model })
      live.model = request.model
      live.effort = undefined
    }
    if (request.effort && live.effort !== request.effort) {
      await agy.request("session/set_config_option", { sessionId: live.acpID, configId: "thought_level", value: request.effort })
      live.effort = request.effort
    }
    // opencode's Plan agent maps to agy's plan mode. opencode's permission rules still apply either way.
    const mode = request.agent === "plan" ? "plan" : "default"
    if (live.mode !== mode && (live.mode !== undefined || mode !== "default")) {
      await agy.request("session/set_config_option", { sessionId: live.acpID, configId: "mode", value: mode }).catch(() => {})
    }
    live.mode = mode
  }

  private onUpdate(entry: Entry, live: Live, update: SessionUpdate) {
    if (update.sessionUpdate === "session_info_update" && typeof update.title === "string" && update.title) {
      live.title = update.title
      for (const waiter of entry.titleWaiters.splice(0)) waiter(update.title)
    }
    if (update.sessionUpdate === "usage_update" && typeof update.used === "number") live.used = update.used
    const turn = live.turn
    if (turn && !turn.ended) this.push(turn, { type: "update", update })
  }

  private push(turn: Turn, event: TurnEvent) {
    turn.events.push(event)
    const wake = turn.wake
    turn.wake = undefined
    wake?.()
  }

  private startTurn(live: Live, content: Convert.ContentBlock[]) {
    const turn: Turn = { events: [], ended: false, done: Promise.withResolvers<void>() }
    live.turn = turn
    this.process!.request("session/prompt", { sessionId: live.acpID, prompt: content }).then(
      (result) => this.endTurn(live, { result }, turn),
      (error) => this.endTurn(live, { error }, turn),
    )
  }

  private endTurn(live: Live, outcome: { result?: any; error?: unknown }, turn = live.turn) {
    if (!turn || turn.ended) return
    turn.ended = true
    this.push(turn, { type: "end", ...outcome })
    turn.done.resolve()
  }

  private async cancelTurn(live: Live) {
    const turn = live.turn
    if (!turn || turn.ended) return
    this.process?.notify("session/cancel", { sessionId: live.acpID })
    const timeout = new Promise<void>((resolve) => setTimeout(resolve, 15_000).unref?.())
    await Promise.race([turn.done.promise, timeout])
    if (!turn.ended) this.endTurn(live, { result: { stopReason: "cancelled" } }, turn)
  }

  /** Streams one agy turn as one opencode step. agy runs its own tools, so they arrive as provider-executed calls. */
  private async readTurn(live: Live, request: Request) {
    const turn = live.turn!
    const emit = request.emit
    const meta = { [PROVIDER_ID]: { session: live.acpID } }
    let block: { kind: "text" | "reasoning"; id: string } | undefined
    let counter = 0
    const tools = new Map<string, { name: string; done: boolean }>()
    const close = () => {
      if (!block) return
      emit({ type: `${block.kind}-end`, id: block.id, providerMetadata: meta } as LanguageModelV3StreamPart)
      block = undefined
    }
    const open = (kind: "text" | "reasoning") => {
      if (block?.kind === kind) return block.id
      close()
      block = { kind, id: `${live.acpID}:${Date.now()}:${counter++}` }
      emit({ type: `${kind}-start`, id: block.id, providerMetadata: meta } as LanguageModelV3StreamPart)
      return block.id
    }
    const toolCall = (update: SessionUpdate) => {
      const id = update.toolCallId as string
      if (!id || tools.has(id)) return
      const name = toolName(update)
      tools.set(id, { name, done: false })
      close()
      emit({
        type: "tool-call",
        toolCallId: id,
        toolName: name,
        input: JSON.stringify(toolInput(update)),
        providerExecuted: true,
        providerMetadata: meta,
      })
    }
    const toolResult = (update: SessionUpdate) => {
      const id = update.toolCallId as string
      const tool = tools.get(id)
      if (!tool || tool.done) return
      tool.done = true
      emit({
        type: "tool-result",
        toolCallId: id,
        toolName: tool.name,
        result: toolOutput(update) as any,
        isError: update.status === "failed",
        providerMetadata: meta,
      } as LanguageModelV3StreamPart)
    }
    // Every shown call needs a result, or opencode would treat it as still pending.
    const settle = (status: string) => {
      for (const [id, tool] of tools) if (!tool.done) toolResult({ sessionUpdate: "tool_call_update", toolCallId: id, status, title: tool.name })
    }
    const usage = (final?: any) => {
      const output = final?.usage?.outputTokens ?? 0
      const reasoning = final?.usage?.thoughtTokens ?? 0
      return {
        // agy reports the session's context use; opencode uses input tokens to decide when to compact.
        inputTokens: { total: live.used, noCache: undefined, cacheRead: final?.usage?.cachedReadTokens, cacheWrite: undefined },
        outputTokens: { total: output, text: Math.max(0, output - reasoning), reasoning },
      }
    }

    const onAbort = () => this.push(turn, { type: "abort" })
    if (request.signal.aborted) onAbort()
    else request.signal.addEventListener("abort", onAbort, { once: true })
    emit({ type: "response-metadata", id: live.acpID, modelId: request.model })
    try {
      while (true) {
        const event = turn.events.shift()
        if (!event) {
          await new Promise<void>((resolve) => (turn.wake = resolve))
          continue
        }
        if (event.type === "update") {
          const update = event.update
          const text = update.content?.type === "text" ? (update.content.text as string) : ""
          switch (update.sessionUpdate) {
            case "agent_message_chunk":
              // agy reports its own retries as message text; they are not part of the reply.
              if (text && !RETRY_NOTICE.test(text)) emit({ type: "text-delta", id: open("text"), delta: text })
              break
            case "agent_thought_chunk":
              if (text) emit({ type: "reasoning-delta", id: open("reasoning"), delta: text })
              break
            case "tool_call":
            case "tool_call_update":
              toolCall(update)
              if (update.status === "completed" || update.status === "failed") toolResult(update)
              break
          }
          continue
        }
        close()
        if (event.type === "abort") {
          await this.cancelTurn(live)
          settle("cancelled")
          return
        }
        settle(event.error ? "failed" : "cancelled")
        if (event.error) throw toError(event.error)
        const stop = event.result?.stopReason
        emit({
          type: "finish",
          finishReason: { unified: stop === "max_tokens" ? "length" : stop === "refusal" ? "content-filter" : "stop", raw: stop },
          usage: usage(event.result),
          providerMetadata: meta,
        })
        return
      }
    } finally {
      request.signal.removeEventListener("abort", onAbort)
    }
  }

  /** Side requests (titles, summaries, compaction) run in a throwaway agy session. */
  private async once(request: Request) {
    const agy = await this.agy()
    const created = await agy.request("session/new", { cwd: this.options.directory, mcpServers: [] })
    const live: Live = { acpID: created.sessionId, instructionsHash: "", unsubscribe: () => {} }
    const entry: Entry = { lock: Promise.resolve(), titleWaiters: [] }
    live.unsubscribe = agy.subscribe(live.acpID, (update) => this.onUpdate(entry, live, update))
    try {
      await this.configure(agy, live, request)
      // The task (title, summary) is in opencode's messages; opencode's harness system prompt is not sent.
      const history = request.prompt.some((message) => message.role === "assistant" || message.role === "tool")
      const content = history ? Convert.catchUp(request.prompt, 0) : Convert.messages(request.prompt, false)
      this.startTurn(live, content.length ? content : [{ type: "text", text: "Continue." }])
      await this.readTurn(live, request)
    } finally {
      live.unsubscribe()
      await agy.request("session/close", { sessionId: live.acpID }).catch(() => {})
    }
  }
}

const RETRY_NOTICE = /^\s*⚠ API error \(attempt \d+\)/

function flatten(options: any[]): Array<{ value: string; name?: string }> {
  // Select options may be grouped
  return options.flatMap((option) => (Array.isArray(option?.options) ? flatten(option.options) : [option]))
}

function toolName(update: SessionUpdate) {
  const name = typeof update.name === "string" && update.name ? update.name : typeof update.kind === "string" ? update.kind : "tool"
  return name.replace(/[^a-zA-Z0-9_-]/g, "_")
}

function toolInput(update: SessionUpdate) {
  const input = { ...((update.rawInput ?? update.args ?? {}) as Record<string, unknown>) }
  delete input.toolSummary
  if (typeof update.title === "string" && update.title) input.title ??= update.title
  return input
}

function toolOutput(update: SessionUpdate) {
  const parts: string[] = []
  for (const item of (update.content ?? []) as any[]) {
    if (item?.type === "content" && item.content?.type === "text") parts.push(item.content.text)
    if (item?.type === "diff") parts.push(`${item.path}\n${unifiedDiff(item.oldText ?? "", item.newText ?? "")}`)
  }
  const raw = update.rawOutput
  return {
    title: update.title ?? "",
    status: update.status ?? "",
    output: parts.length > 0 ? parts.join("\n") : raw === undefined ? "" : typeof raw === "string" ? raw : JSON.stringify(raw),
  }
}

// A compact line diff, so edits read well in opencode's tool output.
function unifiedDiff(before: string, after: string) {
  const a = before.split("\n")
  const b = after.split("\n")
  let start = 0
  while (start < a.length && start < b.length && a[start] === b[start]) start++
  let endA = a.length
  let endB = b.length
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) {
    endA--
    endB--
  }
  return [...a.slice(start, endA).map((line) => `-${line}`), ...b.slice(start, endB).map((line) => `+${line}`)].join("\n")
}

/** opencode permission for an agy tool call. Actions match opencode's own tools, so its rules apply as usual. */
export function permissionFor(toolCall: PermissionRequest["toolCall"], directory: string) {
  const call = toolCall as any
  const input = (call.rawInput ?? call.args ?? {}) as Record<string, any>
  const name = String(call.name ?? "")
  const kind = String(call.kind ?? "")
  const paths = [
    ...((call.locations ?? []) as Array<{ path?: string }>).map((location) => location.path),
    input.Path,
    input.TargetFile,
    input.AbsolutePath,
    input.path,
    input.file_path,
  ].filter((path): path is string => typeof path === "string" && path.length > 0)
  const absolute = [...new Set(paths.map((path) => (path.startsWith("/") ? path : `${directory}/${path}`)))]
  const metadata = { agy: { tool: name, title: call.title } }
  if (kind === "execute" || name === "run_command") {
    const command = String(input.CommandLine ?? input.command ?? call.title ?? name)
    return { action: "shell", resources: [command], save: [command], metadata }
  }
  if (kind === "edit" || kind === "delete" || kind === "move" || /write|edit|replace|delete/.test(name))
    return { action: "edit", resources: absolute.length ? absolute : ["*"], save: ["*"], metadata }
  if (kind === "read" || name === "read_file") return { action: "read", resources: absolute.length ? absolute : ["*"], metadata }
  if (name === "search_web") return { action: "websearch", resources: [String(input.query ?? input.Query ?? "*")], metadata }
  if (kind === "fetch" || /url|fetch|browser/.test(name)) {
    const url = String(input.Url ?? input.url ?? input.URL ?? "*")
    return { action: "webfetch", resources: [url], metadata }
  }
  return { action: name || "agy", resources: [String(call.title ?? name ?? "*")], metadata }
}

export function toError(error: unknown) {
  if (error instanceof AcpError) {
    if (error.isAuth) return new Error("Antigravity CLI is not signed in. Run `agy` in a terminal and sign in, then try again.")
    const detail = typeof error.data === "string" ? error.data : error.data ? JSON.stringify(error.data) : ""
    return new Error(`Antigravity: ${error.message}${detail ? ` (${detail})` : ""}`)
  }
  return error instanceof Error ? error : new Error(String(error))
}
