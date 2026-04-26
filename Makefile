# The Temporal CLI installer drops the binary at ~/.temporalio/bin/. Adding it
# to PATH here so Makefile targets work regardless of whether the user's shell
# profile (~/.zshrc / ~/.bashrc) has been sourced in this session.
export PATH := $(HOME)/.temporalio/bin:$(HOME)/.local/bin:$(PATH)

.PHONY: help preflight local temporal proxy worker run ingest chapters chunks embed search answer chat research doctor test integration-test smoke lint format

# Default target — print available commands.
help:
	@echo "Horcrux — make targets"
	@echo ""
	@echo "  Setup:"
	@echo "    make preflight         Check that required system tools are installed"
	@echo ""
	@echo "  Infrastructure (each in its own terminal):"
	@echo "    make local             Start Qdrant (docker compose, detached)"
	@echo "    make temporal          Start Temporal dev server (foreground)"
	@echo "    make proxy             Start LiteLLM proxy on :4000 (foreground)"
	@echo ""
	@echo "  App processes:"
	@echo "    make worker            Start the Temporal worker (foreground)"
	@echo "    make ingest            Trigger the OCR ingest workflow (full corpus)"
	@echo "    make chapters          Derive chapters.json from raw_pages.json (silver tier)"
	@echo "    make chunks            Derive chunks.json from chapters + alias dict (gold tier)"
	@echo "    make embed             Embed chunks and upsert into Qdrant (requires \`make local\`)"
	@echo "    make search Q=\"...\"    Hybrid retrieval smoke (CHARS=\"slug,slug\" K=10 optional)"
	@echo "    make answer Q=\"...\"    End-to-end retrieve + synthesise (requires \`make proxy\`)"
	@echo "    make chat              Multi-turn REPL with history (requires \`make proxy\`)"
	@echo "    make research Q=\"...\"  Multi-step research with planner + aggregator"
	@echo "    make run Q=\"query\"     Run a single query end-to-end"
	@echo ""
	@echo "  Diagnostics & quality:"
	@echo "    make doctor            Print resolved config (secrets redacted)"
	@echo "    make test              Unit tests"
	@echo "    make integration-test  Integration tests (requires running infra)"
	@echo "    make smoke             Smoke tests (end-to-end)"
	@echo "    make lint              ruff check"
	@echo "    make format            ruff format"

# ── Setup ─────────────────────────────────────────────────────────

# Check that all required system tools are installed.
# The Python deps come from `uv sync`; this checks the binaries that don't.
preflight:
	@echo "Checking required tools..."
	@command -v uv >/dev/null 2>&1 \
		&& echo "  ✓ uv" \
		|| echo "  ✗ uv             (install: https://docs.astral.sh/uv/)"
	@command -v docker >/dev/null 2>&1 \
		&& echo "  ✓ docker" \
		|| echo "  ✗ docker         (install: https://docs.docker.com/engine/install/)"
	@command -v tesseract >/dev/null 2>&1 \
		&& echo "  ✓ tesseract" \
		|| echo "  ✗ tesseract      (apt install tesseract-ocr / brew install tesseract)"
	@command -v temporal >/dev/null 2>&1 \
		&& echo "  ✓ temporal CLI" \
		|| echo "  ✗ temporal CLI   (install: curl -sSf https://temporal.download/cli.sh | sh; then add ~/.temporalio/bin to PATH)"
	@uv run python -c "import litellm" >/dev/null 2>&1 \
		&& echo "  ✓ litellm (python pkg)" \
		|| echo "  ✗ litellm        (run: uv sync)"
	@uv run python -c "import spacy; spacy.load('en_core_web_sm')" >/dev/null 2>&1 \
		&& echo "  ✓ spaCy en_core_web_sm" \
		|| echo "  ✗ spaCy model    (run: uv run python -m spacy download en_core_web_sm)"
	@uv run python -c "import fastcoref" >/dev/null 2>&1 \
		&& echo "  ✓ fastcoref (python pkg)" \
		|| echo "  ✗ fastcoref      (run: uv sync)"
	@test -f .env \
		&& echo "  ✓ .env file" \
		|| echo "  ✗ .env file      (cp .env.example .env, then add a real ANTHROPIC_API_KEY)"

# ── Infrastructure ────────────────────────────────────────────────

local:
	docker compose up -d qdrant

temporal:
	./scripts/temporal-dev.sh

proxy:
	uv run litellm --config litellm_config.yaml --port 4000

# ── App processes ─────────────────────────────────────────────────

worker:
	uv run python -m horcrux.worker

ingest:
	uv run python -m horcrux.main ingest

chapters:
	uv run python scripts/build_chapters.py

chunks:
	uv run python scripts/build_chunks.py

embed:
	uv run python scripts/build_embeddings.py

search:
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make search Q=\"your query\" [CHARS=\"slug1,slug2\"] [K=10]"; \
		exit 1; \
	fi
	@uv run python scripts/search.py "$(Q)" --chars "$(CHARS)" --top-k $${K:-10}

answer:
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make answer Q=\"your question\" [CHARS=\"slug1,slug2\"] [K=10]"; \
		exit 1; \
	fi
	@uv run python scripts/answer.py "$(Q)" --chars "$(CHARS)" --top-k $${K:-10}

chat:
	@uv run python scripts/chat.py

research:
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make research Q=\"your research question\""; \
		exit 1; \
	fi
	@uv run python scripts/research.py "$(Q)"

run:
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make run Q=\"your query\""; \
		exit 1; \
	fi
	uv run python -m horcrux.main "$(Q)"

# ── Diagnostics & quality ─────────────────────────────────────────

doctor:
	@uv run python -c "from horcrux.config import settings; print(settings.model_dump_json(indent=2))"

test:
	uv run pytest -m unit

integration-test:
	uv run pytest -m integration

smoke:
	uv run pytest -m smoke

lint:
	uv run ruff check .

format:
	uv run ruff format .
