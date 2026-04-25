# ADR-0002: LiteLLM proxy for model routing

**Date:** 2026-04-25
**Status:** pending
**Pattern:** Gateway / proxy in front of model providers, OpenAI-compatible interface upstream.

## Context

Horcrux makes LLM calls at four sites: intent classification, planning, candidate scoring, synthesis. Two model tiers (Haiku × 3 sites, Sonnet × 1 site). The lab is single-provider (Anthropic) but explicitly evaluates whether the toolchain is portable.

PydanticAI can talk to Anthropic directly via the `anthropic` Python SDK. Adding a proxy is technically optional. The question is whether the proxy earns its keep in the lab — and whether deferring it now would mean a larger refactor later.

## Decision

Run **LiteLLM in proxy mode** (not library mode) on `localhost:4000`. PydanticAI agents address models by alias (`"haiku"`, `"sonnet"`) defined in `litellm_config.yaml`; the proxy translates to the underlying Anthropic IDs.

Specifically:

- `litellm_config.yaml` defines named aliases mapped to provider model strings, with response caching enabled and per-model rate limits.
- PydanticAI agents are configured with `OPENAI_BASE_URL=http://localhost:4000` and use the OpenAI-compatible interface.
- LiteLLM's web UI at `localhost:4000/ui` is the per-call observability surface (alongside LangSmith for graph-level traces).
- Anthropic's `cache_control` headers are verified to flow through LiteLLM's adapter end-to-end before relying on prompt caching.

Three running services on the developer's laptop: Temporal dev server, Qdrant, LiteLLM proxy. Acceptable for a lab.

## Alternatives Considered

**Library mode (`litellm` as a Python import).** Simpler — no separate process to run. Rejected because the proxy's web UI is a primary teaching surface for the lab; library mode hides the routing entirely. Library mode also doesn't naturally support response caching across processes (worker + CLI), which the proxy gives for free.

**No proxy — call Anthropic directly via PydanticAI.** Rejected because evaluating LiteLLM as part of the toolchain is an explicit lab goal. Without the proxy, the "swap providers via config edit" capability is theoretical rather than demonstrated. Also forfeits centralised spend tracking and the response cache.

**Other proxies (OpenRouter, Portkey, Helicone).** All viable. LiteLLM was selected on the basis of largest provider catalogue, OSS license, and the heaviest community footprint at the time of writing — meaning the most abundant integration documentation for a weekend lab.

## Consequences

**Positive**
- Provider portability is real, not hypothetical: the swap-Sonnet-for-Opus stretch experiment is one YAML line and a proxy restart, no application code touched.
- Centralised spend tracking and response cache shared across the worker process and the CLI process.
- LangSmith and LiteLLM provide complementary observability views (graph vs. per-call). Both are useful.

**Negative / risks**
- Adds a third service to the dev environment. Manageable for a single developer; would need orchestration in a team setting.
- Anthropic-specific features can lag in LiteLLM's adapter (notably new beta features). Mitigation: verify `cache_control` end-to-end before relying on it; consider direct Anthropic SDK calls for any feature LiteLLM hasn't surfaced.
- A network hop adds ~5-15ms per LLM call. Negligible relative to LLM latency itself.

**Follow-ups**
- Verify Anthropic prompt caching survives the LiteLLM hop. If not, document the workaround or accept the missed cache discount.
- Consider whether LiteLLM's response cache is sufficient or whether agent-level memoisation (cache the deterministic `intent_agent` response by query hash) is also worthwhile.

## Rollback

The proxy is purely a routing layer — no persistent state, no schema migrations. Rollback is:
1. Stop the proxy process.
2. Replace agent model strings with raw Anthropic IDs (`"claude-haiku-4-5-20251001"`, `"claude-sonnet-4-6"`).
3. Unset `OPENAI_BASE_URL`.

PydanticAI's Anthropic provider takes over directly. No data is affected. ~10 minutes of work.
