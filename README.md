# ⚡ Vibe Validator

> Turn fuzzy app ideas into actionable intelligence — in under 10 seconds.

Vibe Validator is an AI-powered web app that takes your side-hustle idea and instantly generates a structured viability report. It analyzes market size, surfaces competing products, suggests monetization models, and outputs a ready-to-paste starter prompt for no-code tools like Lovable, Claude, or Base44. Built for non-technical founders who ship fast.

---

## Features

- 📊 **Instant viability reports** — market size estimates (TAM/SAM), competitor landscape, key risks, and opportunities for any described app idea
- 🚀 **Ready-to-paste builder prompts** — tailored starter prompts for vibe-coding tools so you can go from idea to building in one click
- ⚡ **HTMX-powered SPA experience** — snappy partial page updates with zero JavaScript framework overhead
- 🔒 **Structured, validated output** — LLM responses are parsed into strict Pydantic models, ensuring consistent and testable report sections
- 🎯 **Zero-friction UI** — paste your idea, click validate, get a shareable report in seconds — no account required

---

## Quick Start

**Requirements:** Python 3.11+, an OpenAI API key

```bash
# 1. Clone the repository
git clone https://github.com/your-org/vibe_validator.git
cd vibe_validator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# 4. Start the development server
uvicorn vibe_validator.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser. Paste an idea, click **Validate**, and your report appears inline — no page reload.

---

## Usage Examples

### Web UI

1. Navigate to `http://localhost:8000`
2. Enter your app idea in the text area, e.g.:
   > *"A subscription app that sends curated Notion templates to freelancers every week"*
3. Click **Validate My Idea**
4. Receive a full viability report with market analysis, competitors, monetization suggestions, and a copy-paste prompt for your builder of choice

### API (direct POST)

```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "idea=A subscription app that sends curated Notion templates to freelancers every week"
```

The endpoint returns an HTML partial (designed for HTMX), but you can inspect the structured data directly.

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

All tests mock the OpenAI API — no API key required to run the test suite.

---

## Project Structure

```
vibe_validator/
├── pyproject.toml                  # Project metadata and dependency declarations
├── requirements.txt                # Pinned pip dependencies
├── .env.example                    # Environment variable template
│
├── vibe_validator/
│   ├── __init__.py                 # Package init, exposes FastAPI app instance
│   ├── main.py                     # App entrypoint: route definitions (/, /validate, /health)
│   ├── analyzer.py                 # Core pipeline: calls OpenAI, parses viability report
│   ├── models.py                   # Pydantic models for requests and structured reports
│   ├── prompts.py                  # System and user prompt templates for the LLM
│   └── templates/
│       ├── index.html              # Landing page with HTMX-powered idea submission form
│       ├── report.html             # Partial template: renders the viability report sections
│       └── error.html              # Partial template: inline error display
│
└── tests/
    ├── test_analyzer.py            # Unit tests for analyzer (mocked OpenAI responses)
    ├── test_routes.py              # Integration tests for FastAPI routes
    └── test_models.py              # Unit tests for Pydantic model validation
```

---

## Configuration

Copy `.env.example` to `.env` and set the following variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | — | Your OpenAI API key ([get one here](https://platform.openai.com/api-keys)) |
| `OPENAI_MODEL` | No | `gpt-4o` | Model used for analysis. Use `gpt-3.5-turbo` for a budget option |
| `OPENAI_MAX_TOKENS` | No | `2048` | Maximum tokens returned per analysis request |

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
OPENAI_MAX_TOKENS=2048
```

> **Never commit your `.env` file to version control.** It is included in `.gitignore` by default.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI |
| Server | Uvicorn |
| Templating | Jinja2 |
| Frontend interactivity | HTMX |
| Styling | TailwindCSS (CDN) |
| AI | OpenAI API (GPT-4o) |
| Data validation | Pydantic v2 |
| Testing | pytest, pytest-asyncio |

---

## License

MIT © Vibe Validator

See [LICENSE](LICENSE) for full terms.

---

*Built with [Jitter](https://github.com/jitter-ai) — an AI agent that ships code daily.*
