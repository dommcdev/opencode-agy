// Runs opencode model requests through `agy acp`.
//
// - One `agy acp` process serves the whole opencode server; each opencode session maps to one ACP session.
// - agy runs with a private home and a custom main agent whose `tools: []` removes Antigravity's built-in tools.
//   opencode's tools reach the model over MCP, so opencode runs every tool itself (permissions, UI, history).
// - When the model calls a tool, the MCP request is held, the opencode step ends with a tool call, and the result
//   opencode sends with its next request answers the held MCP request. The agy turn keeps running throughout, so a
//   tool round trip costs no extra model request.
// - opencode's history stays the source of truth. agy only receives input it has not seen.

import type { LanguageModelV3FunctionTool, LanguageModelV3Prompt, LanguageModelV3StreamPart } from "@ai-sdk/provider"
import { randomBytes, randomUUID } from "node:crypto"
import { tmpdir } from "node:os"
import { AcpError, AcpProcess, type PermissionRequest, type SessionUpdate } from "./agy/acp.ts"
import * as Home from "./agy/home.ts"
import * as Convert from "./convert.ts"
import * as Fingerprint from "./sync/fingerprint.ts"
import * as Names from "./tools/names.ts"
import { ToolServer, type CallToolResult, type ToolDefinition } from "./tools/server.ts"

export const PROVIDER_ID = "agy"
// The live agy session of an opencode session closes after this long without requests; the next one resumes it.
const IDLE_MS = 10 * 60_000
// agy fails MCP tool calls after 3 minutes. A tool that runs longer in opencode (permission prompts, questions,
// subagents, long commands) is detached before that: the agy turn is cancelled and the result arrives as the next
// prompt, so the model never sees a timeout and never re-runs the tool on its own.
const TOOL_DEADLINE_MS = Number(process.env.OPENCODE_AGY_TOOL_DEADLINE_MS) || 170_000
// Parallel calls from one model response reach the MCP server within milliseconds of each other.
const TOOL_BATCH_MS = 150

export type Kind = "primary" | "title" | "generate" | "compaction"

export interface Request {
  readonly kind: Kind | undefined
  readonly sessionID: string | undefined
  readonly model: string
  readonly effort: string | undefined
  readonly system: string
  readonly prompt: LanguageModelV3Prompt
  readonly tools: readonly LanguageModelV3FunctionTool[]
  readonly emit: (part: LanguageModelV3StreamPart) => void
  readonly signal: AbortSignal
}

interface SessionRecord {
  acpID: string
  // Fingerprints of the opencode history this ACP session has seen
  history: string[]
}

interface HeldCall {
  readonly id: string
  readonly name: string
  readonly input: Record<string, unknown>
  readonly resolve: (result: CallToolResult) => void
}

type TurnEvent =
  | { type: "update"; update: SessionUpdate }
  | { type: "call"; call: HeldCall }
  | { type: "flush" }
  | { type: "abort" }
  | { type: "end"; result?: any; error?: unknown }

interface Turn {
  readonly events: TurnEvent[]
  wake?: () => void
  ended: boolean
  readonly done: PromiseWithResolvers<void>
  // agy tool_call ids for opencode tools, waiting to be matched with their MCP request
  readonly announced: Array<{ id: string; name: string; input: unknown }>
  readonly held: Map<string, HeldCall>
  deadline?: ReturnType<typeof setTimeout>
}

interface Live {
  acpID: string
  readonly key: string
  readonly systemHash: string
  readonly toolsHash: string
  model?: string
  effort?: string
  tools: readonly LanguageModelV3FunctionTool[]
  turn?: Turn
  title?: string
  used?: number
  unsubscribe: () => void
  unregister: () => void
}

interface Entry {
  lock: Promise<void>
  record?: SessionRecord
  live?: Live
  idle?: ReturnType<typeof setTimeout>
  // Tool calls of a turn that was detached at the tool deadline; their results lead the next prompt.
  detached?: Set<string>
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
  readonly env?: NodeJS.ProcessEnv
  readonly log?: (message: string) => void
}

function definitions(tools: readonly LanguageModelV3FunctionTool[]): ToolDefinition[] {
  return tools.map((tool) => ({
    name: Names.short(tool.name),
    description: tool.description ?? "",
    inputSchema: tool.inputSchema ?? { type: "object", properties: {} },
  }))
}

function stripSummary(input: unknown) {
  if (!input || typeof input !== "object") return input
  const copy = { ...(input as Record<string, unknown>) }
  delete copy.toolSummary
  return copy
}

const same = (a: unknown, b: unknown) => Fingerprint.hash(a) === Fingerprint.hash(b)

export class Runtime {
  private process?: AcpProcess
  private starting?: Promise<AcpProcess>
  private readonly tools = new ToolServer()
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
      await this.tools.start()
      const env = { ...(this.options.env ?? process.env) }
      // Credential variables would silently switch agy away from the user's Google sign-in.
      for (const name of ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_APPLICATION_CREDENTIALS"])
        delete env[name]
      const agy = new AcpProcess({
        executable,
        args: [
          "acp",
          // Each session gets its own worker; agy only accepts HTTP MCP servers from the client in this mode.
          "--process-per-session",
          "--gemini-dir",
          this.home,
          "--agent",
          Home.AGENT,
          // No bundled skills or slash commands: opencode already expanded its own commands, and text starting
          // with "/" must reach the model as typed.
          "--include-default-customizations=false",
        ],
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

  private permission(request: PermissionRequest): string | undefined {
    const name = request.toolCall?.name ?? ""
    const live = this.byACP.get(request.sessionId)
    // opencode asks for permission itself when it runs the tool, so its tools are always allowed here.
    if (live && name.startsWith(Names.PREFIX) && Names.original(name, live.tools)) {
      const option =
        request.options.find((option) => option.kind === "allow_always") ??
        request.options.find((option) => option.kind === "allow_once")
      return option?.optionId
    }
    // Anything else would be an Antigravity tool, which this agent should not have.
    this.options.log?.(`rejected agy permission request for ${name || request.toolCall?.title}`)
    return request.options.find((option) => option.kind === "reject_once" || option.kind === "reject_always")?.optionId
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
    const sessionID = request.sessionID
    const entry = this.entry(sessionID)
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
    const entry = this.entries.get(sessionID)
    if (!entry) return Promise.resolve(undefined)
    if (entry.live?.title) return Promise.resolve(entry.live.title)
    return new Promise<string | undefined>((resolve) => {
      const timer = setTimeout(() => resolve(undefined), timeoutMs)
      entry.titleWaiters.push((title) => {
        clearTimeout(timer)
        resolve(title)
      })
    })
  }

  /** opencode stopped running the session, so a turn still waiting for tool results may never get them. */
  idle(sessionID: string) {
    const entry = this.entries.get(sessionID)
    if (entry) this.armIdle(entry, true)
  }

  async close() {
    for (const entry of this.entries.values()) {
      if (entry.idle) clearTimeout(entry.idle)
      await this.closeLive(entry).catch(() => {})
    }
    this.process?.kill()
    this.tools.close()
  }

  private entry(sessionID: string) {
    let entry = this.entries.get(sessionID)
    if (!entry) {
      entry = { lock: Promise.resolve(), titleWaiters: [] }
      this.entries.set(sessionID, entry)
    }
    return entry
  }

  private armIdle(entry: Entry, stopped = false) {
    if (entry.idle) clearTimeout(entry.idle)
    // A turn holding tool calls is busy until opencode answers, however long that takes,
    // unless opencode reported that it stopped running the session.
    const turn = entry.live?.turn
    if (!stopped && turn && !turn.ended && turn.held.size > 0) return
    entry.idle = setTimeout(() => {
      entry.idle = undefined
      void this.closeLive(entry)
    }, IDLE_MS)
    entry.idle.unref?.()
  }

  private dispose(entry: Entry, live: Live) {
    live.unsubscribe()
    live.unregister()
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

  private async saveRecord(sessionID: string, record: SessionRecord | undefined) {
    if (record) await this.options.storage.set(`sessions/${sessionID}`, record).catch(() => {})
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

    // 1. A turn waiting for tool results continues with them.
    const waiting = entry.live?.turn
    if (waiting && !waiting.ended && waiting.held.size > 0 && start !== undefined) {
      if (await this.continueTurn(entry, entry.live!, request, groups, start)) return
    }

    // 2. Otherwise a new agy turn gets what it has not seen.
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

    const detached = entry.detached
    entry.detached = undefined
    let content = detached?.size ? this.detachedResults(request.prompt, start ?? 0, detached) : undefined
    content ??= Convert.catchUp(request.prompt, start ?? 0)
    if (content.length === 0) content = [{ type: "text", text: "Continue." }]

    entry.record = { acpID: live.acpID, history: groups.map((group) => group.hash) }
    await this.saveRecord(sessionID, entry.record)
    this.startTurn(live, content)
    await this.readStep(entry, live, request)
  }

  // Results of tool calls that outlived agy's MCP deadline, followed by anything the user added since.
  private detachedResults(prompt: LanguageModelV3Prompt, start: number, ids: Set<string>) {
    const collected = Convert.collect(prompt, start)
    const found = [...collected.results.entries()].filter(([id]) => ids.has(id))
    if (found.length === 0) return undefined
    const blocks: Convert.ContentBlock[] = [
      {
        type: "text",
        text: "Your previous tool calls took longer than the harness allows for one call, so your turn was paused while OpenCode finished them. They have now completed. Do not run them again. Their results:",
      },
    ]
    for (const [id, { part, result }] of found) {
      blocks.push({ type: "text", text: `<tool-result id="${id}" name="${part.toolName}"${result.isError ? ' error="true"' : ""}>` })
      for (const item of result.content as any[]) {
        if (item.type === "text") blocks.push({ type: "text", text: item.text })
        else if (item.type === "image") blocks.push({ type: "image", data: item.data, mimeType: item.mimeType })
        else if (item.type === "resource") blocks.push(item)
      }
      blocks.push({ type: "text", text: "</tool-result>" })
    }
    const after = Convert.catchUp(prompt, collected.end)
    blocks.push(...(after.length ? after : [{ type: "text" as const, text: "Continue with the task." }]))
    return blocks
  }

  private async continueTurn(entry: Entry, live: Live, request: Request, groups: Fingerprint.Group[], start: number) {
    const turn = live.turn!
    const collected = Convert.collect(request.prompt, start)
    const held = [...turn.held.values()]
    if (collected.aborted || held.some((call) => !collected.results.has(call.id))) {
      // A stopped opencode turn needs a fresh agy turn.
      await this.cancelTurn(live)
      return false
    }
    if (turn.deadline) clearTimeout(turn.deadline)
    // Input opencode queued while the tools ran rides along with the last result.
    const extra = Convert.catchUp(request.prompt, collected.end)
      .filter((block) => block.type === "text")
      .map((block) => (block as { text: string }).text)
      .join("\n")
    turn.held.clear()
    held.forEach((call, index) => {
      let result = collected.results.get(call.id)!.result
      if (index === held.length - 1 && extra)
        result = { ...result, content: [...result.content, { type: "text", text: `\n<user-message>\n${extra}\n</user-message>` }] }
      call.resolve(result)
    })
    entry.record = { acpID: live.acpID, history: groups.map((group) => group.hash) }
    await this.saveRecord(request.sessionID!, entry.record)
    await this.readStep(entry, live, request)
    return true
  }

  private async ensureLive(entry: Entry, request: Request): Promise<Live> {
    const agy = await this.agy()
    const systemHash = Fingerprint.hash(request.system)
    const toolsHash = Fingerprint.hash(definitions(request.tools))
    // A new system prompt or tool list needs the session reopened; resume keeps agy's history.
    if (entry.live && (entry.live.systemHash !== systemHash || entry.live.toolsHash !== toolsHash)) await this.closeLive(entry)
    let live = entry.live
    if (!live) {
      const created: Live = {
        acpID: "",
        key: randomBytes(12).toString("hex"),
        systemHash,
        toolsHash,
        tools: request.tools,
        unsubscribe: () => {},
        unregister: () => {},
      }
      created.unregister = this.tools.register(created.key, {
        tools: () => definitions(created.tools),
        call: (name, args) => this.onToolCall(created, name, args),
      })
      const params = {
        cwd: this.options.directory,
        mcpServers: request.tools.length > 0 ? [this.tools.config(created.key)] : [],
        _meta: { antigravity: { systemPrompt: { append: request.system } } },
      }
      const resumeID = entry.record?.acpID
      if (resumeID) {
        // Route updates before agy sends any for the resumed session.
        created.unsubscribe = agy.subscribe(resumeID, (update) => this.onUpdate(entry, created, update))
        const resumed = await agy.request("session/resume", { sessionId: resumeID, ...params }).then(
          () => true,
          (error) => {
            created.unsubscribe()
            if (error instanceof AcpError && error.isAuth) {
              created.unregister()
              throw error
            }
            this.options.log?.(`could not resume agy session ${resumeID}: ${error}`)
            return false
          },
        )
        if (resumed) created.acpID = resumeID
      }
      if (!created.acpID) {
        const result = await agy.request("session/new", params).catch((error) => {
          created.unregister()
          throw error
        })
        created.acpID = result.sessionId
        created.unsubscribe = agy.subscribe(created.acpID, (update) => this.onUpdate(entry, created, update))
      }
      this.byACP.set(created.acpID, created)
      entry.live = created
      live = created
    }
    live.tools = request.tools
    if (live.model !== request.model) {
      await agy.request("session/set_config_option", { sessionId: live.acpID, configId: "model", value: request.model })
      live.model = request.model
      live.effort = undefined
    }
    if (request.effort && live.effort !== request.effort) {
      await agy.request("session/set_config_option", { sessionId: live.acpID, configId: "thought_level", value: request.effort })
      live.effort = request.effort
    }
    return live
  }

  private onUpdate(entry: Entry, live: Live, update: SessionUpdate) {
    if (update.sessionUpdate === "session_info_update" && typeof update.title === "string" && update.title) {
      live.title = update.title
      for (const waiter of entry.titleWaiters.splice(0)) waiter(update.title)
    }
    if (update.sessionUpdate === "usage_update" && typeof update.used === "number") live.used = update.used
    const turn = live.turn
    if (!turn || turn.ended) return
    if (
      (update.sessionUpdate === "tool_call" || update.sessionUpdate === "tool_call_update") &&
      typeof update.name === "string" &&
      update.name.startsWith(Names.PREFIX) &&
      typeof update.toolCallId === "string" &&
      !turn.held.has(update.toolCallId) &&
      !turn.announced.some((call) => call.id === update.toolCallId)
    ) {
      turn.announced.push({ id: update.toolCallId, name: update.name, input: stripSummary(update.rawInput ?? update.args) })
    }
    this.push(turn, { type: "update", update })
  }

  private onToolCall(live: Live, name: string, args: Record<string, unknown>): Promise<CallToolResult> {
    const turn = live.turn
    const tool = Names.original(name, live.tools)
    if (!turn || turn.ended) return Promise.resolve({ isError: true, content: [{ type: "text", text: "This turn has ended." }] })
    if (!tool) return Promise.resolve({ isError: true, content: [{ type: "text", text: `Tool ${name} is not available.` }] })
    // Use agy's own id for the call when it announced one, so both sides name the call the same way.
    const full = Names.PREFIX + name
    let index = turn.announced.findIndex((call) => call.name === full && same(call.input, args))
    if (index < 0) index = turn.announced.findIndex((call) => call.name === full)
    const id = index >= 0 ? turn.announced.splice(index, 1)[0]!.id : `agy_${randomUUID()}`
    return new Promise<CallToolResult>((resolve) => {
      const call: HeldCall = { id, name: tool, input: args, resolve }
      turn.held.set(id, call)
      this.push(turn, { type: "call", call })
    })
  }

  private push(turn: Turn, event: TurnEvent) {
    turn.events.push(event)
    const wake = turn.wake
    turn.wake = undefined
    wake?.()
  }

  private startTurn(live: Live, content: Convert.ContentBlock[]) {
    const turn: Turn = { events: [], ended: false, done: Promise.withResolvers<void>(), announced: [], held: new Map() }
    live.turn = turn
    const agy = this.process!
    agy.request("session/prompt", { sessionId: live.acpID, prompt: content }).then(
      (result) => this.endTurn(live, { result }, turn),
      (error) => this.endTurn(live, { error }, turn),
    )
    return turn
  }

  private endTurn(live: Live, outcome: { result?: any; error?: unknown }, turn = live.turn) {
    if (!turn || turn.ended) return
    turn.ended = true
    if (turn.deadline) clearTimeout(turn.deadline)
    // A held MCP request must not stay open after its turn ended.
    for (const call of turn.held.values()) call.resolve({ isError: true, content: [{ type: "text", text: "Cancelled." }] })
    turn.held.clear()
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

  /** Streams agy's updates as one opencode step: until the model calls tools, or the turn ends. */
  private async readStep(entry: Entry, live: Live, request: Request) {
    const turn = live.turn!
    const emit = request.emit
    const meta = { [PROVIDER_ID]: { session: live.acpID } }
    let block: { kind: "text" | "reasoning"; id: string } | undefined
    let counter = 0
    const batch: HeldCall[] = []
    let flushTimer: ReturnType<typeof setTimeout> | undefined
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
        switch (event.type) {
          case "update": {
            const update = event.update
            const text = update.content?.type === "text" ? (update.content.text as string) : ""
            if (update.sessionUpdate === "agent_message_chunk" && text) emit({ type: "text-delta", id: open("text"), delta: text })
            if (update.sessionUpdate === "agent_thought_chunk" && text) emit({ type: "reasoning-delta", id: open("reasoning"), delta: text })
            break
          }
          case "call":
            close()
            batch.push(event.call)
            flushTimer ??= setTimeout(() => this.push(turn, { type: "flush" }), TOOL_BATCH_MS)
            break
          case "flush": {
            for (const call of batch)
              emit({ type: "tool-call", toolCallId: call.id, toolName: call.name, input: JSON.stringify(call.input), providerMetadata: meta })
            emit({ type: "finish", finishReason: { unified: "tool-calls", raw: "tool_use" }, usage: usage(), providerMetadata: meta })
            // The opencode step is over while the agy turn waits for results. Detach before agy's MCP deadline.
            turn.deadline = setTimeout(() => this.detach(entry, live, turn), TOOL_DEADLINE_MS)
            turn.deadline.unref?.()
            return
          }
          case "abort":
            close()
            await this.cancelTurn(live)
            return
          case "end": {
            close()
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
        }
      }
    } finally {
      request.signal.removeEventListener("abort", onAbort)
    }
  }

  // An opencode tool is still running past agy's MCP deadline: end the agy turn now and deliver the results as the
  // next prompt. Cancelling (rather than answering) keeps agy from spending a model request on a placeholder.
  private detach(entry: Entry, live: Live, turn: Turn) {
    if (live.turn !== turn || turn.ended || turn.held.size === 0) return
    entry.detached = new Set(turn.held.keys())
    this.options.log?.(`detaching ${turn.held.size} long-running tool call(s) from agy session ${live.acpID}`)
    void this.cancelTurn(live)
  }

  /** Side requests (titles, summaries, compaction) run in a throwaway session without tools. */
  private async once(request: Request) {
    const agy = await this.agy()
    const created = await agy.request("session/new", {
      cwd: this.options.directory,
      mcpServers: [],
      _meta: { antigravity: { systemPrompt: { append: request.system } } },
    })
    const live: Live = {
      acpID: created.sessionId,
      key: "",
      systemHash: "",
      toolsHash: "",
      tools: [],
      unsubscribe: () => {},
      unregister: () => {},
    }
    const entry: Entry = { lock: Promise.resolve(), titleWaiters: [] }
    live.unsubscribe = agy.subscribe(live.acpID, (update) => this.onUpdate(entry, live, update))
    try {
      await agy.request("session/set_config_option", { sessionId: live.acpID, configId: "model", value: request.model })
      if (request.effort)
        await agy.request("session/set_config_option", { sessionId: live.acpID, configId: "thought_level", value: request.effort })
      const history = request.prompt.some((message) => message.role === "assistant" || message.role === "tool")
      const content = history ? Convert.catchUp(request.prompt, 0) : Convert.messages(request.prompt, false)
      this.startTurn(live, content.length ? content : [{ type: "text", text: "Continue." }])
      await this.readStep(entry, live, { ...request, tools: [] })
    } finally {
      live.unsubscribe()
      await agy.request("session/close", { sessionId: live.acpID }).catch(() => {})
    }
  }
}

function flatten(options: any[]): Array<{ value: string; name?: string }> {
  // Select options may be grouped
  return options.flatMap((option) => (Array.isArray(option?.options) ? flatten(option.options) : [option]))
}

export function toError(error: unknown) {
  if (error instanceof AcpError) {
    if (error.isAuth) return new Error("Antigravity CLI is not signed in. Run `agy` in a terminal and sign in, then try again.")
    const detail = typeof error.data === "string" ? error.data : error.data ? JSON.stringify(error.data) : ""
    return new Error(`Antigravity: ${error.message}${detail ? ` (${detail})` : ""}`)
  }
  return error instanceof Error ? error : new Error(String(error))
}
