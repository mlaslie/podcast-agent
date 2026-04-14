# 🎙️ Podcast Agent

An AI-powered podcast creation service built with [Google ADK](https://google.github.io/adk-docs/). Turns any source material — GCS documents, uploaded PDFs, or a free-form topic — into a professional, natural-sounding podcast MP3 with dynamically generated album art and embedded ID3 tags.

The agent pipeline handles everything: research → script writing → TTS audio production → album art generation → MP3 encoding. Supports both a two-host conversation format and a single-narrator educational format.

---

## How It Works

```
User Input → Source Collector → Script Writer → Audio Producer → MP3 + Album Art
```

1. **Source Collector** — fetches content from GCS files/folders, user-uploaded files, and/or searches the web for free-form topics; date-grounded to prevent hallucination on time-relative queries
2. **Script Writer** — transforms research into a natural script; solo mode produces an educational monologue, duo mode produces a two-host dialogue
3. **Audio Producer** — synthesises audio via Cloud TTS (single-speaker or multi-speaker depending on mode), generates episode-specific album art via Gemini image model, encodes to MP3 with embedded ID3 tags

---

## Project Structure

```
└── root_agent
    ├── __init__.py
    ├── agent_configuration.json
    ├── agent.py
    ├── .env
    ├── requirements.txt
    └── sub_agents
        ├── audio_producer_agent.py
        ├── script_writer_agent.py
        └── source_collector_agent.py
```

---

## Prerequisites

- Python 3.11+
- A Google Cloud project with the following APIs enabled:
  - Vertex AI API
  - Cloud Text-to-Speech API
  - Cloud Storage API
- `gcloud` CLI installed and authenticated

---

## Installation

**1. Clone the repo**

```bash
git clone https://github.com/mlaslie/podcast-agent.git
cd podcast-agent
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Copy `env_example.txt` to `.env` and fill in your values:

```bash
cp env_example.txt .env
```

```ini
# .env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

**5. Configure `agent_configuration.json`**

The repo includes a ready-to-use `agent_configuration.json` with placeholders. Update the two values marked `YOUR_VALUE`:

```json
"vertex_ai": {
  "project": "YOUR_GCP_PROJECT_ID"
},
"output": {
  "gcs_output_bucket": "gs://YOUR_GCS_BUCKET"
}
```

See [Output Modes](#output-modes) for all available options.

---

## Output Modes

### `gemini_enterprise: true` _(recommended for Gemini Enterprise / Agent Engine)_

Produces three outputs automatically:

| Output | Format | How it's delivered |
|---|---|---|
| Album art | JPEG artifact | Renders inline in the Gemini Enterprise chat |
| Podcast audio | WAV artifact | Plays in the browser via the Gemini Enterprise audio player |
| Podcast download | MP3 → GCS | Clickable GCS link in chat — full ID3 tags + embedded album art for personal players |

> ⚠️ `gcs_output_bucket` must be configured when `gemini_enterprise` is `true`.

### `gemini_enterprise: false` _(ADK web / API)_

| Output | Format | How it's delivered |
|---|---|---|
| Album art | JPEG artifact | Renders inline in the chat |
| Podcast download | MP3 artifact | Download button in the chat — full ID3 tags + embedded album art |

### `output_mode: "gcs"` _(GCS only, no artifacts)_

Tagged MP3 uploaded to GCS. Returns a clickable authenticated URL.

### `output_mode: "local"` _(local disk, development only)_

Tagged MP3 written to `local_output_dir` on disk.

---

## Running the Agent

**Local (ADK web)**

```bash
adk web
```

Open [http://localhost:8000](http://localhost:8000) and select **root_agent**.

**Agent Engine (Gemini Enterprise)**

Deploy via `vertexai.agent_engines` and connect through the Gemini Enterprise interface. Ensure `gemini_enterprise: true` and `gcs_output_bucket` are set in `agent_configuration.json` before deploying.

---

## Usage

The agent greets you with a single intake message covering all required fields:

| Field | Example |
|---|---|
| **Speakers** | `1 speaker` (solo educational) · `2 speakers` (conversation) |
| **Sources** | `gs://my-bucket/docs/` · `gs://my-bucket/report.pdf` · upload a file · describe a topic for Google Search |
| **Target Length** | `3 minutes` · `10 minutes` *(approximate)* |
| **Target Audience** | `software developer` · `high school student` · `general public` |
| **Additional Context** | Areas to focus on, angles to take, or topics to avoid *(optional)* |

Confirm when prompted. In Gemini Enterprise mode you'll receive the album art inline in chat, a playable WAV in the browser, and a GCS download link for the tagged MP3.
