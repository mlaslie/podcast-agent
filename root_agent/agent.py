# agent.py
# Root orchestrator and production pipeline for the podcast creation service.
# Guides the user through intake, then delegates to the sequential sub-agent
# pipeline: source collection → script writing → audio production.

import json
from pathlib import Path

from google.adk.agents import Agent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.apps import App
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.tools.agent_tool import AgentTool


# sub-agents
from .sub_agents.source_collector_agent import source_collector_agent
from .sub_agents.script_writer_agent import script_writer_agent
from .sub_agents.audio_producer_agent import audio_producer_agent

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
THIS_VERSION = "version_20260326-2218"

# ---------------------------------------------------------------------------
# Load Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "agent_configuration.json"

with open(_CONFIG_PATH) as _f:
    _CFG = json.load(_f)

ORCHESTRATOR_MODEL = _CFG["models"]["orchestrator"]
GCS_OUTPUT_BUCKET  = _CFG["output"]["gcs_output_bucket"]

_SOLO     = _CFG["solo_host"]
_HOST_1   = _CFG["hosts"]["host_1"]
_HOST_2   = _CFG["hosts"]["host_2"]

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
        "professional, natural-sounding podcast audio file with one or two hosts."
    ),
    instruction=f"""
You are the **Podcast Creation Orchestrator**. Your job is to guide the user
through a friendly intake process, collect and confirm their requirements,
gather the source content, then trigger the automated production pipeline.

## Your Workflow
### Step 1 — Intake
Greet the user with the following message, exactly as written:

---
👋 Welcome to the **Gemini Podcast Agent! ({THIS_VERSION})**

To get started, I'll need a few details:

🎙️ **Speakers** — 1-speaker or 2-speaker podcast?
- *1 speaker* — a single narrator delivers a clear, focused monologue. Great for tutorials, explainers, and training content.
- *2 speakers* — two hosts have a natural back-and-forth conversation. Great for discussions, news briefings, and general interest topics.

📚 **Source(s)** — Where should I pull the content from? Mix and match:
- Google Search (just describe the topic)
- GCS bucket — `gs://your_bucket/your_folder/`
- GCS file(s) — `gs://your_bucket/your_file.pdf`
- Direct text input
- File upload (pdf, txt, md, html, csv, json)

⏱️ **Target Length** — How long should the podcast be? *(e.g. 3 minutes, 10 minutes — note: length is approximate)*

🎯 **Target Audience** — Who is this for? *(e.g. software developer, high school student, general public)*

💡 **Additional Context** *(optional)* — Any areas to focus on, angles to take, or topics to avoid?
---

After the user responds, extract the narrator mode, sources, target length, target audience,
and any additional context from their reply. If anything is ambiguous or missing, ask a
focused follow-up question for just that item rather than repeating the full intake.

### Step 2 — Confirm
Summarise what you understood and ask the user to confirm before proceeding.
Include the active output mode and the chosen narrator mode in the summary.
If mode is "gcs", include the GCS bucket path.
Do not move to Step 3 until the user explicitly confirms.

### Step 3 — Production Pipeline
Run `podcast_production_pipeline`. Pass the confirmed `narrator_mode` value
(`"solo"` for 1 narrator, `"duo"` for 2 hosts) as part of the state so that
the script writer and audio producer can adapt accordingly.
This sequential agent will automatically run the source collector, script
writer, and audio producer in order.
Do NOT invoke script generation or audio production yourself — hand off
entirely to the pipeline.

When handing off to the pipeline, include in the context:
- `narrator_mode`: `"solo"` or `"duo"`

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
    tools=[podcast_production_pipeline_tool,],
)

# ---------------------------------------------------------------------------
# App — wires SaveFilesAsArtifactsPlugin so uploaded files are automatically
# saved as ADK artifacts when they arrive. Name must match the agent directory
# name ("root_agent") so adk web can locate sessions correctly.
# ---------------------------------------------------------------------------
app = App(
    name="root_agent",
    root_agent=root_agent,
    plugins=[SaveFilesAsArtifactsPlugin()],
)