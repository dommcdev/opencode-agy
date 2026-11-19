// opencode-agy: use Google's Antigravity CLI (`agy`) as an opencode provider. opencode runs the user's own `agy`
// through its official Agent Client Protocol server (`agy acp`), the way Zed, JetBrains and t3code do. It never reads
// agy's credentials, never talks to Google itself, and never sends opencode's own harness prompt or tools to agy.

import type { Plugin } from "@opencode/plugin"
import { existsSync } from "node:fs"
import { delimiter, join } from "node:path"
import { AGENT_OPTION, AgyLanguageModel, INSTRUCTIONS_OPTION, KIND_HEADER } from "./model.ts"
import { PROVIDER_ID, Runtime } from "./runtime.ts"

const PACKAGE = "opencode-agy"
const METHOD_ID = "agy-login"
const CATALOG_KEY = "catalog"
const DEFAULT_CONTEXT = 1_048_576
// The user's own instruction files: AGENTS.md and `instructions` from opencode config.
const USER_INSTRUCTIONS = new Set(["core/instructions"])

interface CatalogModel {
  id: string
  name: string
  efforts: string[]
  context?: number
}

function findAgy(env: NodeJS.ProcessEnv = process.env): string | undefined {
  if (env.OPENCODE_AGY_PATH) return env.OPENCODE_AGY_PATH
  const bun = (globalThis as any).Bun
  if (bun?.which) return bun.which("agy") ?? undefined
  for (const dir of (env.PATH ?? "").split(delimiter)) {
    const candidate = join(dir, process.platform === "win32" ? "agy.exe" : "agy")
    if (dir && existsSync(candidate)) return candidate
  }
  return undefined
}

function toModel(model: CatalogModel): any {
  return {
    id: model.id,
    modelID: model.id,
    providerID: PROVIDER_ID,
    name: model.name,
    ...(model.id.includes("flash") ? { family: "gemini-flash" } : {}),
    capabilities: { tools: true, input: ["text", "image", "pdf"], output: ["text"], temperature: false },
    // Antigravity's thinking levels. Without a variant, agy uses the model's default level.
    variants: model.efforts.map((effort) => ({ id: effort, settings: { effort } })),
    time: { released: 0 },
    // Usage is covered by the user's Antigravity plan.
    cost: [],
    status: "active",
    enabled: true,
    // 0 means unknown to opencode; agy picks its own output budget.
    limit: { context: model.context ?? DEFAULT_CONTEXT, output: 0 },
  }
}

function userInstructions(system: ReadonlyArray<{ text: string; metadata?: any }>) {
  const texts: string[] = []
  for (const part of system) {
    const sources = part.metadata?.experimental?.instructions
    if (!Array.isArray(sources)) continue
    for (const source of sources) if (USER_INSTRUCTIONS.has(source?.key) && source.text) texts.push(source.text)
  }
  return texts.join("\n\n")
}

const plugin: Plugin.Plugin = {
  id: "opencode-agy",
  async setup(ctx) {
    const log = (message: string) => console.error(`[opencode-agy] ${message}`)
    const runtime = new Runtime({
      directory: ctx.location.directory,
      storage: { get: (key) => ctx.storage.get(key), set: (key, value) => ctx.storage.set(key, value) },
      executable: () => findAgy(),
      // agy's permission prompts go through opencode's permission rules, which ask the user when needed.
      ask: (input) =>
        ctx.permission
          .assert({
            sessionID: input.sessionID as any,
            ...(input.agent ? { agent: input.agent as any } : {}),
            action: input.action,
            resources: input.resources,
            ...(input.save ? { save: input.save } : {}),
            ...(input.metadata ? { metadata: input.metadata } : {}),
          })
          .then(
            () => true,
            () => false,
          ),
      log: process.env.OPENCODE_AGY_DEBUG ? log : undefined,
    })

    const saved = (await ctx.storage.get(CATALOG_KEY).catch(() => undefined)) as { models?: CatalogModel[] } | undefined
    const catalog = { models: Array.isArray(saved?.models) ? saved.models : [] }

    await ctx.integration.transform((editor) => {
      editor.update(PROVIDER_ID, (integration) => {
        integration.name = "Antigravity CLI"
      })
      // Like Claude Agent's login method, this only records the choice. The sign-in belongs to `agy`.
      editor.method.update({
        integrationID: PROVIDER_ID,
        method: { id: METHOD_ID, type: "external", label: "Antigravity CLI login (sign in once by running `agy`)" },
      })
    })

    await ctx.provider.transform((editor) => {
      editor.add({
        info: { id: PROVIDER_ID, name: "Antigravity CLI", activation: "auto", package: `aisdk:${PACKAGE}` } as any,
        models: catalog.models.map(toModel),
      })
    })

    await ctx.aisdk.hook(
      "sdk",
      (event) => {
        if (event.package !== PACKAGE) return
        event.sdk = { languageModel: (modelID: string) => new AgyLanguageModel(modelID, runtime) }
      },
      { providerID: PROVIDER_ID },
    )

    await ctx.session.hook(
      "model.request",
      (event) => {
        event.headers[KIND_HEADER] = event.kind
      },
      { providerID: PROVIDER_ID },
    )

    await ctx.session.hook(
      "context",
      (event) => {
        // agy chooses its own output budget.
        delete event.options.maxTokens
        event.options[AGENT_OPTION] = event.agent
        event.options[INSTRUCTIONS_OPTION] = userInstructions(event.system)
      },
      { providerID: PROVIDER_ID },
    )

    // agy retries failed model calls itself; replaying the turn here would also repeat quota errors.
    await ctx.session.hook(
      "retry",
      (event) => {
        event.decision = { retry: false }
      },
      { providerID: PROVIDER_ID },
    )

    // agy already titles every conversation, so reuse that instead of spending a model request on it.
    await ctx.session.hook(
      "title",
      async (event) => {
        const title = await runtime.title(event.sessionID, 20_000)
        if (title) event.result = title
      },
      { providerID: PROVIDER_ID },
    )

    const report = async (status: { status: "needs_auth"; message: string; url?: string } | undefined) => {
      const connection = await ctx.integration.connection.active(PROVIDER_ID).catch(() => undefined)
      if (!connection) return
      await ctx.integration.connection.status({ integrationID: PROVIDER_ID, connection, status }).catch(() => {})
    }

    // Refresh the model list from the account. This only opens a session to read its options; it sends no prompt.
    void (async () => {
      if (!findAgy()) {
        await report({
          status: "needs_auth",
          message: "Antigravity CLI is not installed. Install it, then run `agy` once to sign in.",
          url: "https://antigravity.google/docs/install",
        })
        return
      }
      try {
        const probe = await runtime.probe()
        const models = probe.models.filter((model) => model.id)
        if (JSON.stringify(models) !== JSON.stringify(catalog.models)) {
          catalog.models = models
          await ctx.storage.set(CATALOG_KEY, { models } as any)
          await ctx.provider.reload()
        }
        await report(undefined)
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        log(`could not list models: ${message}`)
        if (/auth|sign/i.test(message))
          await report({ status: "needs_auth", message: "Antigravity CLI is not signed in. Run `agy` in a terminal and sign in." })
      }
    })()

    return async () => {
      await runtime.close()
    }
  },
}

export default plugin
