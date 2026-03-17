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
   - Google Cloud Storage folder path (e.g., `gs://my-bucket/docs/`)
   - GCS document path (e.g., `gs://my-bucket/report.pdf`)
   - A free-form topic/subject (e.g., "the fall of the Roman Empire")
   - Any mix of the above

2. **Target Length** – choose from:
   - `1-2 minutes`   (~200-300 words of script)
   - `3-6 minutes`   (~500-900 words of script)
   - `10-15 minutes` (~1500-2500 words of script)
   - `unlimited`     (cover the topic as thoroughly as needed)

3. **Target Audience** – a short description (e.g., "software engineers new to
   machine learning", "general public curious about ancient history")

4. **Audio Output Destination** – ask the user where to save the finished WAV:
   - **Google Cloud Storage (GCS)** – uploaded to GCS and a shareable URL returned
     - If the user chooses GCS, inform them that the default output bucket is
       `{GCS_OUTPUT_BUCKET}` and ask if they would like to use it or
       provide a different destination path.

5. **Additional Context** – any extra guidance: tone, angle, key points to
   emphasise, things to avoid, etc. This field is optional.

### Step 2 — Confirm
Summarise what you understood and ask the user to confirm before proceeding.
Include the output destination (and GCS bucket path if applicable) in the summary.
Do not move to Step 3 until the user explicitly confirms.

### Step 3 — Production Pipeline
Run `podcast_production_pipeline`. This sequential agent will automatically
run the source collector, script writer, and audio producer in order.
Do NOT invoke script generation or audio production yourself — hand off
entirely to the pipeline.

When handing off to the audio producer, pass the output destination flags:
- Always pass `write_to_gcs=True` and `gcs_output_bucket`
  set to the bucket path the user confirmed (default: `{GCS_OUTPUT_BUCKET}`)

### Step 4 — Respond to the User

--

## Tone & Style
- Be warm, enthusiastic, and professional.
- Never skip or rush the confirmation step — good inputs produce great podcasts.
- If something is ambiguous (e.g., a GCS path that might not exist), ask for
  clarification rather than guessing.
    """,
    # sub_agents=[podcast_production_pipeline,],
    tools=[podcast_production_pipeline_tool,], #running as a tool suppresses inter-agent output
)