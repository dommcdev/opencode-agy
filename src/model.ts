// The AI SDK language model opencode calls. Every request is handed to the agy runtime.

import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider"
import { PROVIDER_ID, type Kind, type Runtime } from "./runtime.ts"

export const KIND_HEADER = "x-opencode-agy-kind"
export const SESSION_HEADER = "x-opencode-session-id"
/** Provider options set by the plugin's `context` hook. */
export const AGENT_OPTION = "agyAgent"
export const INSTRUCTIONS_OPTION = "agyInstructions"

export class AgyLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3"
  readonly provider = PROVIDER_ID
  readonly supportedUrls = {}

  constructor(
    readonly modelId: string,
    private readonly runtime: Runtime,
  ) {}

  async doGenerate(options: LanguageModelV3CallOptions) {
    const { stream } = await this.doStream(options)
    const content: LanguageModelV3Content[] = []
    let text = ""
    let finish: Extract<LanguageModelV3StreamPart, { type: "finish" }> | undefined
    for await (const part of stream as any as AsyncIterable<LanguageModelV3StreamPart>) {
      if (part.type === "text-delta") text += part.delta
      if (part.type === "finish") finish = part
      if (part.type === "error") throw part.error
    }
    if (text) content.push({ type: "text", text })
    return {
      content,
      finishReason: finish?.finishReason ?? { unified: "stop" as const, raw: undefined },
      usage: finish?.usage ?? {
        inputTokens: { total: undefined, noCache: undefined, cacheRead: undefined, cacheWrite: undefined },
        outputTokens: { total: undefined, text: undefined, reasoning: undefined },
      },
      warnings: [],
    }
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const abort = new AbortController()
    const signal = options.abortSignal ? AbortSignal.any([abort.signal, options.abortSignal]) : abort.signal
    const headers = options.headers ?? {}
    const kind = headers[KIND_HEADER] as Kind | undefined
    const settings = (options.providerOptions?.[PROVIDER_ID] ?? {}) as Record<string, unknown>
    const text = (value: unknown) => (typeof value === "string" ? value : undefined)
    if (process.env.OPENCODE_AGY_DUMP) {
      const { mkdirSync, writeFileSync } = await import("node:fs")
      mkdirSync(process.env.OPENCODE_AGY_DUMP, { recursive: true })
      writeFileSync(
        `${process.env.OPENCODE_AGY_DUMP}/${Date.now()}-${kind ?? "unknown"}.json`,
        JSON.stringify({ model: this.modelId, kind, headers, prompt: options.prompt, providerOptions: options.providerOptions }, null, 1),
      )
    }
    let cancelled = false
    let task: Promise<void> | undefined
    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      start: (controller) => {
        controller.enqueue({ type: "stream-start", warnings: [] })
        task = this.runtime
          .run({
            kind: kind ?? (headers[SESSION_HEADER] ? "primary" : undefined),
            sessionID: headers[SESSION_HEADER],
            agent: text(settings[AGENT_OPTION]),
            model: this.modelId,
            effort: text(settings.effort),
            instructions: text(settings[INSTRUCTIONS_OPTION]) ?? "",
            prompt: options.prompt,
            signal,
            emit: (part) => {
              if (!cancelled) controller.enqueue(part)
            },
          })
          .then(
            () => {
              if (!cancelled) controller.close()
            },
            (error) => {
              if (!cancelled) controller.error(error)
            },
          )
      },
      cancel: () => {
        cancelled = true
        abort.abort()
        return task
      },
    })
    return { stream }
  }
}
