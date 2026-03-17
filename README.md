# 🎙️ Podcast Agent

An AI-powered podcast creation service built with [Google ADK](https://google.github.io/adk-docs/). Turns any source material — GCS documents, PDFs, or a free-form topic — into a professional, natural-sounding two-host podcast audio file (WAV).

The agent pipeline handles everything: research → script writing → multi-speaker TTS audio production.

---

## How It Works

```
User Input → Source Collector → Script Writer → Audio Producer → WAV File
```

1. **Source Collector** — fetches content from GCS files/folders and/or searches the web for free-form topics
2. **Script Writer** — transforms research into a natural two-host dialogue script
3. **Audio Producer** — converts the script to a multi-speaker WAV via Gemini TTS

---

## Project Structure

```
└── root_agent
    ├── __init.py__
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
  - Cloud Storage API (if using GCS sources or output)
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

Copy the provided `env_example.txt` to `.env` and fill in your values:

```bash
cp env_example.txt .env
```

```ini
# .env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

**5. Update `agent_configuration.json`**

Replace the placeholder values with your own project settings:

```json
{
  "vertex_ai": {
    "project":  "your-gcp-project-id",
    "location": "us-central1"
  },
  "output": {
    "gcs_output_bucket": "gs://your-output-bucket"
  }
}
```

---

## Running the Agent

Start the ADK web interface from the project root:

```bash
adk web
```

Then open [http://localhost:8000](http://localhost:8000) in your browser and select **root_agent** from the dropdown.

---

## Usage

The orchestrator will guide you through a short intake form asking for:

| Field | Example |
|---|---|
| **Sources** | `gs://my-bucket/docs/` · `gs://my-bucket/report.pdf` · `"the fall of the Roman Empire"` |
| **Target Length** | `1-2 min` · `3-6 min` · `10-15 min` · `unlimited` |
| **Target Audience** | `"software engineers new to ML"` |
| **Output Destination** | GCS bucket path or local |
| **Additional Context** | Tone, key points to cover, things to avoid _(optional)_ |

Confirm when prompted and the pipeline will run automatically, returning the finished WAV file path or GCS URL.