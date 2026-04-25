"""Test-session bootstrapping.

`horcrux.config` instantiates its `Settings` singleton at import time and fails
fast on missing required env vars. That's correct production behaviour but
breaks test collection if `.env` is absent — the import error fires before
pytest can show useful output.

Setting env defaults here, before any test module is imported, gives
`Settings()` what it needs to construct without polluting the real
`.env` or hitting any external service.
"""

import os

# Stub credentials — never used in tests because unit tests never make real
# LLM calls. Integration tests that need a real key set their own env.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-stub")

# Don't ship test traces to a real LangSmith project.
os.environ.setdefault("LANGSMITH__TRACING_ENABLED", "false")
os.environ.setdefault("LANGSMITH__PROJECT", "horcrux-tests")
