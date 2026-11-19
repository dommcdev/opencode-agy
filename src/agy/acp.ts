// JSON-RPC client for `agy acp`, the official Agent Client Protocol server in Antigravity CLI.
// One `agy acp --process-per-session` front process serves every session of this plugin instance; agy gives each
// session its own worker process.

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process"

export const CLIENT_INFO = { name: "opencode", title: "OpenCode", version: "agy-plugin" }

export interface SessionUpdate {
  readonly sessionUpdate: string
  readonly [key: string]: any
}

export interface PermissionRequest {
  readonly sessionId: string
  readonly toolCall: { readonly toolCallId: string; readonly name?: string; readonly title?: string; readonly rawInput?: unknown }
  readonly options: ReadonlyArray<{ readonly optionId: string; readonly name: string; readonly kind: string }>
}

export class AcpError extends Error {
  constructor(
    readonly code: number,
    message: string,
    readonly data?: unknown,
  ) {
    super(message)
  }
  get isAuth() {
    return this.code === -32000 || /auth/i.test(this.message)
  }
}

export interface Options {
  readonly executable: string
  readonly args: readonly string[]
  readonly cwd: string
  readonly env: NodeJS.ProcessEnv
  /** Answers agy's permission prompts. Returns the chosen optionId, or undefined to cancel. */
  readonly permission: (request: PermissionRequest) => string | undefined
}

export class AcpProcess {
  private child: ChildProcessWithoutNullStreams
  private nextID = 0
  private readonly waiting = new Map<number, { resolve: (value: any) => void; reject: (error: Error) => void }>()
  private readonly listeners = new Map<string, Set<(update: SessionUpdate) => void>>()
  private buffer = ""
  private stderr: string[] = []
  readonly exited: Promise<void>
  ended = false
  initialize?: any

  constructor(private readonly options: Options) {
    this.child = spawn(options.executable, [...options.args], {
      cwd: options.cwd,
      env: options.env,
      stdio: ["pipe", "pipe", "pipe"],
    })
    this.child.stdout.setEncoding("utf8")
    this.child.stdout.on("data", (chunk: string) => this.read(chunk))
    this.child.stderr.setEncoding("utf8")
    this.child.stderr.on("data", (chunk: string) => {
      this.stderr.push(chunk)
      if (this.stderr.length > 50) this.stderr.shift()
    })
    // A failed write after exit must not crash the opencode server.
    this.child.stdin.on("error", () => {})
    this.exited = new Promise((resolve) => {
      const done = () => {
        if (this.ended) return
        this.ended = true
        const error = new Error(`agy acp exited${this.lastError() ? `: ${this.lastError()}` : ""}`)
        for (const call of this.waiting.values()) call.reject(error)
        this.waiting.clear()
        resolve()
      }
      this.child.on("exit", done)
      this.child.on("error", (error) => {
        this.stderr.push(String(error))
        done()
      })
    })
  }

  private lastError() {
    return this.stderr.join("").trim().split("\n").slice(-3).join(" ").slice(0, 500)
  }

  private write(message: Record<string, unknown>) {
    if (this.ended) return
    this.child.stdin.write(JSON.stringify({ jsonrpc: "2.0", ...message }) + "\n")
  }

  private read(chunk: string) {
    this.buffer += chunk
    let newline: number
    while ((newline = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, newline).trim()
      this.buffer = this.buffer.slice(newline + 1)
      if (!line) continue
      let message: any
      try {
        message = JSON.parse(line)
      } catch {
        continue
      }
      this.dispatch(message)
    }
  }

  private dispatch(message: any) {
    // Response to one of our requests
    if (message.id !== undefined && message.method === undefined) {
      const call = this.waiting.get(message.id)
      if (!call) return
      this.waiting.delete(message.id)
      if (message.error) call.reject(new AcpError(message.error.code, message.error.message, message.error.data))
      else call.resolve(message.result)
      return
    }
    // Request from agy to us
    if (message.id !== undefined) {
      this.answer(message)
      return
    }
    if (message.method === "session/update") {
      const sessionID = message.params?.sessionId
      for (const listener of this.listeners.get(sessionID) ?? []) listener(message.params.update)
    }
  }

  private answer(message: any) {
    if (message.method === "session/request_permission") {
      const optionId = this.options.permission(message.params)
      this.write({
        id: message.id,
        result: { outcome: optionId === undefined ? { outcome: "cancelled" } : { outcome: "selected", optionId } },
      })
      return
    }
    // We advertise no client file system, terminal or elicitation support, so agy should never ask for them.
    this.write({ id: message.id, error: { code: -32601, message: `Method not found: ${message.method}` } })
  }

  request<T = any>(method: string, params: unknown): Promise<T> {
    if (this.ended) return Promise.reject(new Error(`agy acp has exited${this.lastError() ? `: ${this.lastError()}` : ""}`))
    const id = ++this.nextID
    return new Promise<T>((resolve, reject) => {
      this.waiting.set(id, { resolve, reject })
      this.write({ id, method, params })
    })
  }

  notify(method: string, params: unknown) {
    this.write({ method, params })
  }

  subscribe(sessionID: string, listener: (update: SessionUpdate) => void) {
    const set = this.listeners.get(sessionID) ?? new Set()
    set.add(listener)
    this.listeners.set(sessionID, set)
    return () => {
      set.delete(listener)
      if (set.size === 0) this.listeners.delete(sessionID)
    }
  }

  async start(timeoutMs = 30_000) {
    const init = this.request("initialize", {
      protocolVersion: 1,
      // No client file system or terminal: every action goes through opencode's own tools instead.
      clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false },
      clientInfo: CLIENT_INFO,
    })
    let timer: ReturnType<typeof setTimeout> | undefined
    const timeout = new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error(`agy acp did not start within ${timeoutMs / 1000} seconds`)), timeoutMs)
    })
    try {
      this.initialize = await Promise.race([init, timeout, this.exited.then(() => this.request("initialize", {}))])
      return this.initialize
    } finally {
      clearTimeout(timer)
    }
  }

  get version(): string | undefined {
    return this.initialize?.agentInfo?.version
  }

  kill() {
    if (this.ended) return
    this.child.stdin.end()
    this.child.kill("SIGTERM")
    const force = setTimeout(() => this.child.kill("SIGKILL"), 3000)
    force.unref?.()
  }
}
