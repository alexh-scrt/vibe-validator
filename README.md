# ⚡ Vibe Validator

> AI-powered viability reports for your side-hustle app idea — in under 10 seconds.

Vibe Validator takes a fuzzy app idea and instantly generates a structured report covering:

- 📈 **Market size estimates** (TAM, SAM, growth rate)
- 🔍 **Similar existing products** with differentiation angles
- 💰 **Monetisation models** with pros, cons, and ARPU estimates
- ⚠️ **Key risks** and ✨ **key opportunities**
- 🚀 **Ready-to-paste starter prompt** for Lovable, Claude, Base44, or any AI builder

Built for the wave of non-technical founders shipping products in hours, not months.

---

## Table of Contents

1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the App](#running-the-app)
7. [Running Tests](#running-tests)
8. [Project Structure](#project-structure)
9. [API Reference](#api-reference)
10. [Contributing](#contributing)
11. [License](#license)

---

## Features

- **Zero-friction UI** — paste your idea, click Validate, get a report. No signup, no data stored.
- **Structured AI output** — GPT-4o returns a consistent JSON payload validated by Pydantic models.
- **HTMX-powered** — single-page feel with partial HTML swaps. No JavaScript framework needed.
- **Graceful error handling** — API errors, parse failures, and validation issues all surface as friendly inline messages.
- **Clipboard-ready starter prompt** — one click copies a prompt you can drop straight into any vibe-coding tool.
- **Accessible & responsive** — semantic HTML, ARIA roles, and TailwindCSS responsive utilities throughout.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| AI | OpenAI API (GPT-4o by default) |
| Templating | Jinja2 |
| Frontend | HTMX, TailwindCSS (CDN) |
| Validation | Pydantic v2 |
| Testing | pytest, pytest-asyncio, httpx |

---

## Prerequisites

- **Python 3.11 or newer**
- An **OpenAI API key** — get one at <https://platform.openai.com/api-keys>
- `pip` (or a virtual-environment manager like `venv`, `pyenv`, or `uv`)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/vibe_validator.git
cd vibe_validator
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install the package in editable mode (includes dev dependencies):

```bash
pip install -e ".[dev]"
```

---

## Configuration

Copy the environment variable template and fill in your values:

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
# Required
OPENAI_API_KEY=sk-your-real-key-here

# Optional — defaults shown
OPENAI_MODEL=gpt-4o
OPENAI_MAX_TOKENS=2048
APP_HOST=0.0.0.0
APP_PORT=8000
APP_ENV=development
```

> **Never commit your `.env` file.** It is already listed in `.gitignore`.

### Environment variable reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | — | Your OpenAI secret key |
| `OPENAI_MODEL` | No | `gpt-4o` | Model used for analysis |
| `OPENAI_MAX_TOKENS` | No | `2048` | Max tokens per completion |
| `APP_HOST` | No | `0.0.0.0` | Uvicorn bind host |
| `APP_PORT` | No | `8000` | Uvicorn bind port |
| `APP_ENV` | No | `development` | `development` enables auto-reload |

---

## Running the App

### Development (auto-reload)

```bash
uvicorn vibe_validator.main:app --reload
```

Or use the console script (if installed via `pip install -e .`):

```bash
vibe-validator
```

The app will be available at <http://localhost:8000>.

### Production

```bash
APP_ENV=production uvicorn vibe_validator.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2
```

### Docker (optional)

Create a minimal `Dockerfile`:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "vibe_validator.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t vibe-validator .
docker run -p 8000:8000 --env-file .env vibe-validator
```

---

## Running Tests

```bash
# Run all tests
pytest

# With verbose output
pytest -v

# Run a specific test module
pytest tests/test_analyzer.py -v

# Run with coverage (requires pytest-cov)
pip install pytest-cov
pytest --cov=vibe_validator --cov-report=term-missing
```

> All tests mock the OpenAI API — no real API key is required to run the test suite.

---

## Project Structure

```
vibe_validator/
├── __init__.py          # Package init; exposes the FastAPI app instance
├── main.py              # FastAPI routes and app entrypoint
├── analyzer.py          # OpenAI call, JSON extraction, and report parsing
├── models.py            # Pydantic models (IdeaRequest, ViabilityReport, …)
├── prompts.py           # System and user prompt templates
└── templates/
    ├── index.html       # Landing page with HTMX form
    ├── report.html      # Report partial template
    └── error.html       # Error partial template
tests/
├── test_scaffold.py     # Smoke tests for project scaffold
├── test_models.py       # Unit tests for Pydantic models
├── test_analyzer.py     # Unit tests for analyzer (mocked OpenAI)
└── test_routes.py       # Integration tests for FastAPI routes
.env.example             # Environment variable template
requirements.txt         # Pinned pip dependencies
pyproject.toml           # Project metadata and build config
README.md                # This file
```

---

## API Reference

### `GET /`

Serves the main landing page.

**Response:** `200 OK` — HTML page

---

### `POST /validate`

Validates a side-hustle app idea and returns a viability report.

**Content-Type:** `application/x-www-form-urlencoded`

**Form fields:**

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `idea` | `string` | ✅ Yes | 20 – 2000 chars | Free-text description of the app idea |

**Success response:** `200 OK` — `text/html` — rendered `report.html` partial

**Error responses:**

| Status | Condition |
|---|---|
| `422 Unprocessable Entity` | Idea fails validation (too short, too long, blank) |
| `502 Bad Gateway` | OpenAI API unreachable or authentication failed |
| `500 Internal Server Error` | Response parse error or schema validation failure |

All error responses return a `text/html` partial rendered from `error.html`.

---

### `GET /health`

Health check endpoint.

**Response:** `200 OK` — JSON

```json
{"status": "ok", "version": "0.1.0"}
```

---

### Interactive API docs

When the server is running, visit:

- **Swagger UI:** <http://localhost:8000/docs>
- **ReDoc:** <http://localhost:8000/redoc>

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes and add tests.
4. Ensure all tests pass: `pytest`
5. Open a pull request.

Please follow PEP 8 style, include type hints on all functions, and add docstrings to public APIs.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for non-technical founders shipping with AI.*
