// A minimal MCP server (Streamable HTTP, JSON responses) that offers opencode's tools to agy.
// agy only advertises HTTP MCP support with `--process-per-session`, so the plugin hosts this on 127.0.0.1 with a
// random bearer token. Each agy session gets its own path, so tool lists and calls stay per session.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http"
import { randomBytes } from "node:crypto"
import type { AddressInfo } from "node:net"

export interface ToolDefinition {
  readonly name: string
  readonly description: string
  readonly inputSchema: unknown
}

export interface CallToolResult {
  readonly content: ReadonlyArray<Record<string, unknown>>
  readonly isError?: boolean
}

export interface Endpoint {
  /** Current tool list for this session. */
  tools(): readonly ToolDefinition[]
  /** Runs one tool call. Resolves when opencode has the result. */
  call(name: string, args: Record<string, unknown>, meta: Record<string, unknown> | undefined): Promise<CallToolResult>
}

export class ToolServer {
  private server?: Server
  private port = 0
  private readonly token = randomBytes(24).toString("hex")
  private readonly endpoints = new Map<string, Endpoint>()

  async start() {
    if (this.server) return
    const server = createServer((req, res) => {
      this.handle(req, res).catch((error) => {
        if (!res.headersSent) res.writeHead(500, { "content-type": "application/json" })
        res.end(JSON.stringify({ jsonrpc: "2.0", id: null, error: { code: -32603, message: String(error) } }))
      })
    })
    // opencode tool calls can wait on permission prompts, questions and subagents for a long time.
    server.requestTimeout = 0
    server.headersTimeout = 0
    server.timeout = 0
    server.keepAliveTimeout = 60_000
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject)
      server.listen(0, "127.0.0.1", () => resolve())
    })
    server.unref()
    this.server = server
    this.port = (server.address() as AddressInfo).port
  }

  register(key: string, endpoint: Endpoint) {
    this.endpoints.set(key, endpoint)
    return () => {
      if (this.endpoints.get(key) === endpoint) this.endpoints.delete(key)
    }
  }

  /** ACP `McpServer` entry for one session. */
  config(key: string) {
    return {
      type: "http" as const,
      name: "opencode",
      url: `http://127.0.0.1:${this.port}/mcp/${key}`,
      headers: [{ name: "Authorization", value: `Bearer ${this.token}` }],
    }
  }

  close() {
    this.server?.close()
    this.server = undefined
  }

  private async handle(req: IncomingMessage, res: ServerResponse) {
    if (req.headers.authorization !== `Bearer ${this.token}`) {
      res.writeHead(401).end()
      return
    }
    const key = /^\/mcp\/([^/?]+)/.exec(req.url ?? "")?.[1]
    const endpoint = key ? this.endpoints.get(key) : undefined
    if (!endpoint) {
      res.writeHead(404).end()
      return
    }
    if (req.method === "DELETE") {
      res.writeHead(200).end()
      return
    }
    if (req.method !== "POST") {
      // No server-initiated stream
      res.writeHead(405, { allow: "POST" }).end()
      return
    }
    const chunks: Buffer[] = []
    for await (const chunk of req) chunks.push(chunk as Buffer)
    const body = JSON.parse(Buffer.concat(chunks).toString("utf8"))
    const messages = Array.isArray(body) ? body : [body]
    const replies = (await Promise.all(messages.map((message) => this.rpc(endpoint, message)))).filter(
      (reply) => reply !== undefined,
    )
    if (replies.length === 0) {
      res.writeHead(202).end()
      return
    }
    res.writeHead(200, { "content-type": "application/json" })
    res.end(JSON.stringify(Array.isArray(body) ? replies : replies[0]))
  }

  private async rpc(endpoint: Endpoint, message: any) {
    // Notifications (initialized, cancelled) need no reply
    if (message.id === undefined || message.id === null) return undefined
    const result = (value: unknown) => ({ jsonrpc: "2.0", id: message.id, result: value })
    const error = (code: number, text: string) => ({ jsonrpc: "2.0", id: message.id, error: { code, message: text } })
    switch (message.method) {
      case "initialize":
        return result({
          protocolVersion: message.params?.protocolVersion ?? "2025-06-18",
          capabilities: { tools: { listChanged: false } },
          serverInfo: { name: "opencode", version: "1.0.0" },
        })
      case "ping":
        return result({})
      case "tools/list":
        return result({
          tools: endpoint.tools().map((tool) => ({
            name: tool.name,
            description: tool.description,
            inputSchema: tool.inputSchema,
            annotations: { readOnlyHint: false },
          })),
        })
      case "tools/call": {
        const params = message.params ?? {}
        try {
          return result(await endpoint.call(String(params.name), params.arguments ?? {}, params._meta))
        } catch (cause) {
          return result({ isError: true, content: [{ type: "text", text: cause instanceof Error ? cause.message : String(cause) }] })
        }
      }
      case "resources/list":
        return result({ resources: [] })
      case "resources/templates/list":
        return result({ resourceTemplates: [] })
      case "prompts/list":
        return result({ prompts: [] })
      default:
        return error(-32601, `Method not found: ${message.method}`)
    }
  }
}
