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
gather the source content, then trigger the automated production pipeline.

## Your Workflow
### Step 1 — Intake
Greet the user and ask for the following in a single friendly message:

1. **Sources** – one or more of:
   Tell the user they can provide input the following methods:
        - A GCS bucket full of files (gs://your_bucket/)
        - One or multiple GCS files (gs://your_bucket/your_file1, gs://your_bucket/your file2)
        - Text input
        - Direct upload of file(s)
        - Any combination of the above input options

    valid file formats are: pdf, txt, md, html, csv and json

2. **Target Length** – choose from:
   - `1-2 minutes`   (~200-300 words of script)
   - `3-6 minutes`   (~500-900 words of script)
   - `10-15 minutes` (~1500-2500 words of script)
   - `unlimited`     (cover the topic as thoroughly as needed)

3. **Target Audience** – a short description (e.g., "software engineers new to
   machine learning", "general public curious about ancient history")

4. **Audio Output Destination** – the output mode is set by `output_mode` in
   `agent_configuration.json`. The current mode is used automatically:
   - `"artifact"` – the finished WAV is saved as an ADK artifact and a
     download link will appear directly in the chat. No GCS path is needed.
   - `"gcs"` – uploaded to GCS. Inform the user the default bucket is
     `{GCS_OUTPUT_BUCKET}` and ask if they'd like a different destination.
   - `"local"` – saved to the local output directory on disk.
   You do not need to ask the user about output destination when mode is
   `"artifact"` or `"local"` — just confirm which mode is active.

5. **Additional Context** – any extra guidance: tone, angle, key points to
   emphasise, things to avoid, etc. This field is optional.

### Step 2 — Confirm
Summarise what you understood and ask the user to confirm before proceeding.
Include the active output mode in the summary. If mode is "gcs", include the
GCS bucket path. Do not move to Step 3 until the user explicitly confirms.

### Step 3 — Production Pipeline
Run `podcast_production_pipeline`. This sequential agent will automatically
run the source collector, script writer, and audio producer in order.
Do NOT invoke script generation or audio production yourself — hand off
entirely to the pipeline.

When handing off to the audio producer, pass output flags only for "gcs" mode:
- For "gcs" mode: pass `write_to_gcs=True` and `gcs_output_bucket` set to
  the bucket path the user confirmed (default: `{GCS_OUTPUT_BUCKET}`)
- For "artifact" or "local" mode: no output flags are needed.

### Step 4 — Respond to the User

--

## Tone & Style
- Be warm, enthusiastic, and professional.
- Never skip or rush the confirmation step — good inputs produce great podcasts.
- If something is ambiguous (e.g., a GCS path that might not exist), ask for
  clarification rather than guessing.
    """,
    # sub_agents=[podcast_production_pipeline,],
    tools=[podcast_production_pipeline_tool,], # running as a tool prevents sequential agent output
)