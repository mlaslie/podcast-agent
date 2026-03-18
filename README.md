# 🎙️ Podcast Agent

An AI-powered podcast creation service built with [Google ADK](https://google.github.io/adk-docs/). Turns any source material — GCS documents, PDFs, or a free-form topic — into a professional, natural-sounding two-host podcast audio file (WAV).

The agent pipeline handles everything: research → script writing → multi-speaker TTS audio production.

Known Bugs:
 - Sometimes TTS background text makes it into the podcast (rare)
 - Sometimes in GCS output mode, the wrong URL will be gave, file will be in GCS

Updates 20260318-1422:
 - Migrated from genAI SDK to texttospeech SDK in order to eliminate scene bleed into the script
    - now uses MultiSpeakerMarkup turns to drive TTS
 - In certain situations, scene bleeding into script was still happening, the following updates have helped:
    - removed markdown from scene description, this seemed to sometimes confuse TTS model and drive bleeding into script
    - flattend scene text into a single dense block of text
    - added explicit system prefix: "System Instructions (Do not read aloud):"
 - Updated requirements.txt to include 'google-cloud-texttospeech'

Updates 20260317-1914:
 - Support for UI based uploads of files (adk web or Gemini Enterprise)
 - Support for artifact based podcast output (output to web UI directly)
 
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
    "output_mode":        "artifact",
    "local_output_dir":   "output",
    "gcs_output_bucket":  "gs://podcast_agent_output"    
  }
}
```

output_mode:
 - artifact = the podcast will be provided to the user via the UI directly
 - gcs = output podcast to user provide GCS bucket (or gcs_output_bucket if not provided)
 
local_output_dir: used for local testing, not supported currently

gcs_output_bucket: if output_mode = "gcs" this is the default bucket where podcasts will be saved

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
