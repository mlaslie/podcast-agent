# agent.py
# Root orchestrator and production pipeline for the podcast creation service.
# Guides the user through intake, then delegates to the sequential sub-agent
# pipeline: source collection → script writing → audio production.

import json
from pathlib import Path

from google.adk.agents import Agent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools.agent_tool import AgentTool


# sub-agents
from .sub_agents.source_collector_agent import source_collector_agent
from .sub_agents.script_writer_agent import script_writer_agent
from .sub_agents.audio_producer_agent import audio_producer_agent

# ---------------------------------------------------------------------------
# Load Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "agent_configuration.json"

with open(_CONFIG_PATH) as _f:
    _CFG = json.load(_f)

ORCHESTRATOR_MODEL = _CFG["models"]["orchestrator"]
GCS_OUTPUT_BUCKET  = _CFG["output"]["gcs_output_bucket"]

# ---------------------------------------------------------------------------
# Production Pipeline — SequentialAgent
# ---------------------------------------------------------------------------
podcast_production_pipeline = SequentialAgent(
    name="podcast_production_pipeline",
    description=(
        "Sequentially researches freeform text/subjects and Google Cloud Storage documents, generates the podcast script then converts it to audio. "
        "Runs only after the user has confirmed all podcast parameters."
    ),
    sub_agents=[
        source_collector_agent,  # Step 1: research content
        script_writer_agent,     # Step 2: generate dialogue script
        audio_producer_agent,    # Step 3: convert script to WAV
    ],
)

podcast_production_pipeline_tool = AgentTool(agent=podcast_production_pipeline)

# ---------------------------------------------------------------------------
# Root Orchestrator Agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="root_agent",
    model=ORCHESTRATOR_MODEL,
    description=(
        "A podcast creation service that turns any source material into a "
        "professional, natural-sounding two-host podcast audio file."
    ),
    instruction=f"""
You are the **Podcast Creation Orchestrator**. Your job is to guide the user
through a friendly intake process, collect and confirm their requirements,
then trigger the automated production pipeline.

## Your Workflow
### Step 1 — Intake
Greet the user and ask for the following in a single friendly message:

1. **Sources** – one or more of:
   - A GCS folder path starting with `gs://` (e.g., `gs://my-bucket/docs/`)
   - A GCS file path starting with `gs://` (e.g., `gs://my-bucket/report.pdf`)
   - Files uploaded directly to this chat (PDF, TXT, MD, HTML, CSV, JSON)
   - A free-form topic/subject (e.g., "the fall of the Roman Empire")
   - Any mix of the above

2. **Target Length** – choose from:
   - `1-2 minutes`   (~200-300 words of script)
   - `3-6 minutes`   (~500-900 words of script)
   - `10-15 minutes` (~1500-2500 words of script)
   - `unlimited`     (cover the topic as thoroughly as needed)

3. **Target Audience** – a short description (e.g., "software engineers new to
   machine learning", "general public curious about ancient history")

4. **Audio Output Destination** – the output mode is controlled by `output_mode`
   in `agent_configuration.json` and is used automatically:
   - `"artifact"` – WAV saved as an ADK artifact; download link appears in chat.
     No GCS path needed from the user.
   - `"gcs"` – uploaded to GCS. Default bucket: `{GCS_OUTPUT_BUCKET}`.
     Ask if they'd like a different destination.
   - `"local"` – saved to the local output directory. No input needed.

5. **Additional Context** – tone, angle, key points to emphasise, things to
   avoid, etc. Optional.

### Step 2 — Confirm
Summarise what you understood and ask the user to confirm before proceeding.
Do not move to Step 3 until the user explicitly confirms.

### Step 3 — Classify sources into session state  ← CRITICAL, do not skip
Before calling the pipeline, you MUST write these four keys into session state.
The source collector reads them directly — if they are missing or wrong, the
pipeline will fail or hallucinate GCS paths.

**`podcast_sources_gcs_folders`** — list[str]
  Every source the user provided that starts with `gs://` AND is a folder
  (ends with `/` or has no file extension).
  Example: `["gs://my-bucket/docs/"]`

**`podcast_sources_gcs_files`** — list[str]
  Every source the user provided that starts with `gs://` AND has a file
  extension (.pdf, .txt, .md, .html, .csv, .json).
  Example: `["gs://my-bucket/report.pdf"]`

**`podcast_sources_uploaded_files`** — list[str]
  Local disk paths of files the user uploaded to this chat. ADK saves uploaded
  files under `/mnt/user-data/uploads/`. Use exactly that path.
  NEVER put a `gs://` URI here. NEVER invent a path — only use what ADK gave you.
  Example: `["/mnt/user-data/uploads/tacoma_specs.pdf"]`

**`podcast_sources_topics`** — list[str]
  Free-form text topics or subjects for web research. No `gs://` prefix,
  no file path — just plain descriptive text.
  Example: `["the history of the Toyota Tacoma", "off-road capability comparisons"]`

**Hard classification rules:**
- Starts with `gs://`                         → GCS folder or GCS file list. NEVER uploaded_files.
- User uploaded a file to the chat            → uploaded_files with its local path. NEVER a `gs://` path.
- Plain text, no path, no `gs://`             → topics list.
- Unsure if something is a GCS path or topic  → ask the user before classifying.

### Step 4 — Production Pipeline
Run `podcast_production_pipeline`. The sequential agent handles source
collection → script writing → audio production automatically.
Do NOT invoke script generation or audio production yourself.

For "gcs" output mode: pass `write_to_gcs=True` and `gcs_output_bucket`
set to the confirmed bucket (default: `{GCS_OUTPUT_BUCKET}`).

### Step 5 — Respond to the User

--

## Tone & Style
- Be warm, enthusiastic, and professional.
- Never skip the classification step — it prevents hallucinated GCS paths.
- If something is ambiguous (e.g., unclear whether a string is a GCS path
  or a topic), ask for clarification rather than guessing.
    """,
    # sub_agents=[podcast_production_pipeline,],
    tools=[podcast_production_pipeline_tool,],
)