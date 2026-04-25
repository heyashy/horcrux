# ADR-0004: Centralised config via pydantic-settings

**Date:** 2026-04-25
**Status:** pending
**Pattern:** Singleton-as-module (a `Settings` instance defined once at module import, accessed everywhere via a single import).

## Context

The lab integrates six external services (Anthropic, Qdrant, Temporal, LiteLLM, LangSmith, sentence-transformers/HuggingFace). Each carries configuration: hostnames, ports, API keys, model identifiers, collection names, embedding parameters. Without discipline this fans out into:

- `os.getenv("ANTHROPIC_API_KEY")` calls scattered through every module.
- Inconsistent defaults across files (`port = 6333` here, `port = int(os.getenv("QDRANT_PORT") or 6333)` there).
- Late failure: missing config discovered at the moment of first network call rather than at startup.
- Implicit secret handling, with leak risk via logs or `repr` output.

CLAUDE.md mandates: "No `os.getenv` inside core logic; config assembled at entrypoints only," and "Fail fast on missing config." This ADR captures *how* we satisfy both rules in this codebase.

## Decision

A single `horcrux/config.py` module exposes a typed `Settings` object built from `pydantic-settings`. The module instantiates `settings = Settings()` once at import, and all other modules access config exclusively via `from horcrux.config import settings`.

Specifics:

- `pydantic-settings` (v2) is the underlying library — the first-party Pydantic settings package.
- Configuration is grouped into nested `BaseSettings` subclasses (`QdrantConfig`, `TemporalConfig`, `LiteLLMConfig`, `LangSmithConfig`, `EmbeddingConfig`) for namespacing. The top-level `Settings` composes them.
- Required fields (notably `anthropic_api_key`) have no default — `Settings()` raises `ValidationError` on import if `.env` is missing them.
- Secrets use `SecretStr`. `repr` shows `'**********'`; explicit `.get_secret_value()` is required to retrieve the raw value.
- Env var names use `__` as the nested delimiter (`QDRANT__PORT=6333` resolves to `settings.qdrant.port`). Case-insensitive.
- The LiteLLM proxy reads `.env` via its YAML's `os.environ/...` syntax independently — both readers look at the same source of truth without chaining.
- Tests override config via `monkeypatch.setattr(config, "settings", Settings(...))` in a `conftest.py` autouse fixture, or per-test via `settings.model_copy(update={...})`.

## Alternatives Considered

**Hand-rolled singleton class with `__new__` override.** Standard Python singleton pattern. Rejected as over-engineered: Python module imports are *already* singletons (a module is loaded once and cached in `sys.modules`), and `pydantic-settings` provides typed validation for free. Reinventing this would add ceremony with no benefit.

**Direct `os.getenv` calls at usage sites.** The default Python pattern. Rejected on the basis of CLAUDE.md rules and the well-known fragility (silent defaults, late failure, no type safety, no validation).

**`python-dotenv` + a `Config` dataclass populated at startup.** Lighter weight than `pydantic-settings`. Rejected because the validation, type coercion, and `SecretStr` features of `pydantic-settings` are exactly what this lab benefits from, and the project already depends on Pydantic v2 for PydanticAI. Adding `pydantic-settings` is a near-zero-cost extension; adding `python-dotenv` and re-implementing validation by hand is a step backwards.

**`omegaconf` / `hydra` / TOML config files.** Powerful but designed for ML experiment management with parameter sweeps and overrides. Rejected as overkill for a six-service lab — would add cognitive load and a different mental model from the rest of the (Pydantic-centric) codebase.

**Configuration injected as function arguments rather than imported.** The "pure" functional approach. Rejected because the lab uses framework-managed entry points (Temporal workers, PydanticAI agents at module top level) where dependency injection requires significant ceremony for marginal gain. The singleton is pragmatic for a single-process, single-tenant lab.

## Consequences

**Positive**
- One file to read to understand every external dependency the application has.
- Startup-time validation: a misconfigured env reveals itself in the first second, not after a 7-minute OCR run.
- Secrets are typed and don't leak to logs by accident.
- Adding a new external service is one block in one file (`class FooConfig(BaseSettings): ...`).
- Test isolation is cheap: a conftest fixture monkeypatches `settings` to a controlled instance.

**Negative / risks**
- Module-import-time instantiation means `from horcrux.config import settings` will *raise* if `.env` is broken. Tests must provide a fixture or environment before importing. Mitigation: documented in the design doc and enforced via `tests/conftest.py`.
- No runtime reload. Changing `.env` requires a process restart. Considered a feature, not a bug — config drift mid-run is a class of bug we want to make impossible.
- Pydantic v2 only. We're already on v2 via PydanticAI; no migration risk. Worth flagging if a future dependency forces a v1 downgrade.

**Follow-ups**
- A `make doctor` Makefile target that imports `horcrux.config.settings` and prints a redacted summary — useful for first-run diagnosis ("which fields are set, which use defaults").
- If LangSmith tracing is conditional on `settings.langsmith.tracing_enabled`, ensure the `LANGCHAIN_*` env vars are set *before* any `langgraph` / `pydantic_ai` import — otherwise tracing won't attach. Wire this in `horcrux/__init__.py` if needed.

## Rollback

This is a code-organisation decision with no infrastructure or schema impact. Rollback is mechanical:

1. Inline the relevant `settings.xxx` accesses back to literal values or `os.getenv` calls at each call site.
2. Delete `horcrux/config.py`.
3. Remove `pydantic-settings` from dependencies.

Reversible in under an hour. No data is affected.
