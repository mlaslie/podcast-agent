# agent.py
# Root orchestrator and production pipeline for the podcast creation service.
# Guides the user through intake, then delegates to the sequential sub-agent
# pipeline: source collection → script writing → audio production.

import json
from pathlib import Path
from typing import Optional

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
THIS_VERSION = "version_20260324-1040"

# ---------------------------------------------------------------------------
# Load Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "agent_configuration.json"

with open(_CONFIG_PATH) as _f:
    _CFG = json.load(_f)

ORCHESTRATOR_MODEL    = _CFG["models"]["orchestrator"]
GCS_OUTPUT_BUCKET     = _CFG["output"]["gcs_output_bucket"]

_SOLO     = _CFG["solo_host"]
_HOST_1   = _CFG["hosts"]["host_1"]
_HOST_2   = _CFG["hosts"]["host_2"]

_LANG_CFG             = _CFG.get("language", {})
DEFAULT_LANGUAGE_CODE = _LANG_CFG.get("code", "en-US")
DEFAULT_LANGUAGE_NAME = _LANG_CFG.get("name", "English (United States)")

# ---------------------------------------------------------------------------
# Language Resolution
# ---------------------------------------------------------------------------
# Maps user language input (plain names, regional variants, BCP-47 codes)
# to validated (code, display_name) pairs from the Gemini TTS supported list.

_SUPPORTED_LANGUAGES = {
    "ar-EG": "Arabic (Egypt)", "bn-BD": "Bangla (Bangladesh)",
    "nl-NL": "Dutch (Netherlands)", "en-IN": "English (India)",
    "en-US": "English (United States)", "fr-FR": "French (France)",
    "de-DE": "German (Germany)", "hi-IN": "Hindi (India)",
    "id-ID": "Indonesian (Indonesia)", "it-IT": "Italian (Italy)",
    "ja-JP": "Japanese (Japan)", "ko-KR": "Korean (South Korea)",
    "mr-IN": "Marathi (India)", "pl-PL": "Polish (Poland)",
    "pt-BR": "Portuguese (Brazil)", "ro-RO": "Romanian (Romania)",
    "ru-RU": "Russian (Russia)", "es-ES": "Spanish (Spain)",
    "ta-IN": "Tamil (India)", "te-IN": "Telugu (India)",
    "th-TH": "Thai (Thailand)", "tr-TR": "Turkish (Turkey)",
    "uk-UA": "Ukrainian (Ukraine)", "vi-VN": "Vietnamese (Vietnam)",
    "af-ZA": "Afrikaans (South Africa)", "sq-AL": "Albanian (Albania)",
    "am-ET": "Amharic (Ethiopia)", "ar-001": "Arabic (World)",
    "hy-AM": "Armenian (Armenia)", "az-AZ": "Azerbaijani (Azerbaijan)",
    "eu-ES": "Basque (Spain)", "be-BY": "Belarusian (Belarus)",
    "bg-BG": "Bulgarian (Bulgaria)", "my-MM": "Burmese (Myanmar)",
    "ca-ES": "Catalan (Spain)", "ceb-PH": "Cebuano (Philippines)",
    "cmn-CN": "Chinese Mandarin (China)", "cmn-TW": "Chinese Mandarin (Taiwan)",
    "hr-HR": "Croatian (Croatia)", "cs-CZ": "Czech (Czech Republic)",
    "da-DK": "Danish (Denmark)", "en-AU": "English (Australia)",
    "en-GB": "English (United Kingdom)", "et-EE": "Estonian (Estonia)",
    "fil-PH": "Filipino (Philippines)", "fi-FI": "Finnish (Finland)",
    "fr-CA": "French (Canada)", "gl-ES": "Galician (Spain)",
    "ka-GE": "Georgian (Georgia)", "el-GR": "Greek (Greece)",
    "gu-IN": "Gujarati (India)", "ht-HT": "Haitian Creole (Haiti)",
    "he-IL": "Hebrew (Israel)", "hu-HU": "Hungarian (Hungary)",
    "is-IS": "Icelandic (Iceland)", "jv-JV": "Javanese (Java)",
    "kn-IN": "Kannada (India)", "kok-IN": "Konkani (India)",
    "lo-LA": "Lao (Laos)", "la-VA": "Latin (Vatican City)",
    "lv-LV": "Latvian (Latvia)", "lt-LT": "Lithuanian (Lithuania)",
    "lb-LU": "Luxembourgish (Luxembourg)", "mk-MK": "Macedonian (North Macedonia)",
    "mai-IN": "Maithili (India)", "mg-MG": "Malagasy (Madagascar)",
    "ms-MY": "Malay (Malaysia)", "ml-IN": "Malayalam (India)",
    "mn-MN": "Mongolian (Mongolia)", "ne-NP": "Nepali (Nepal)",
    "nb-NO": "Norwegian Bokmål (Norway)", "nn-NO": "Norwegian Nynorsk (Norway)",
    "or-IN": "Odia (India)", "ps-AF": "Pashto (Afghanistan)",
    "fa-IR": "Persian (Iran)", "pt-PT": "Portuguese (Portugal)",
    "pa-IN": "Punjabi (India)", "sr-RS": "Serbian (Serbia)",
    "sd-IN": "Sindhi (India)", "si-LK": "Sinhala (Sri Lanka)",
    "sk-SK": "Slovak (Slovakia)", "sl-SI": "Slovenian (Slovenia)",
    "es-419": "Spanish (Latin America)", "es-MX": "Spanish (Mexico)",
    "sw-KE": "Swahili (Kenya)", "sv-SE": "Swedish (Sweden)",
    "ur-PK": "Urdu (Pakistan)",
}

# Default BCP-47 for bare language names that have multiple regional variants
_LANGUAGE_DEFAULTS = {
    "english": "en-US", "spanish": "es-ES", "french": "fr-FR",
    "portuguese": "pt-BR", "chinese": "cmn-CN", "mandarin": "cmn-CN",
    "arabic": "ar-EG", "norwegian": "nb-NO",
}

# Casual region word aliases used in keyword search
_REGION_ALIASES = {
    "uk": "kingdom", "britain": "kingdom", "usa": "states",
    "us": "states", "america": "states", "latam": "latin",
    "brasil": "brazil",
}


def resolve_language(user_input: str) -> dict[str, str]:
    """
    Resolves a user language string to a dictionary with 'code' and 'name'.
    Returns an 'error' key if the input does not match any supported language.

    Accepts:
    - Plain language names:           "french", "German", "Japanese"
    - Regional variants:              "Spanish Mexico", "Portuguese Brazil"
    - BCP-47 codes (any case):        "fr-FR", "fr-fr", "de-DE", "es-MX"
    - Display names with parens:      "French (France)", "Portuguese (Brazil)"
    """
    import re as _re
    raw        = user_input.strip()
    normalised = raw.lower()

    def _normalise_bcp47(s: str) -> str:
        parts = s.split("-")
        if len(parts) >= 2:
            return f"{parts[0].lower()}-{'-'.join(p.upper() for p in parts[1:])}"
        return s.lower()

    # 1. BCP-47 exact match
    bcp_try = _normalise_bcp47(raw)
    if bcp_try in _SUPPORTED_LANGUAGES:
        return {"code": bcp_try, "name": _SUPPORTED_LANGUAGES[bcp_try]}

    # 2. Exact display name match
    for code, name in _SUPPORTED_LANGUAGES.items():
        if name.lower() == normalised:
            return {"code": code, "name": name}

    # 3. Bare language name defaults
    if normalised in _LANGUAGE_DEFAULTS:
        code = _LANGUAGE_DEFAULTS[normalised]
        return {"code": code, "name": _SUPPORTED_LANGUAGES[code]}

    # 4. Keyword search with region aliases
    tokens   = set(_re.sub(r"[(),]", " ", normalised).split())
    expanded = set(tokens)
    for t in tokens:
        if t in _REGION_ALIASES:
            expanded.add(_REGION_ALIASES[t])

    best_code, best_name, best_score = None, None, 0
    for code, name in _SUPPORTED_LANGUAGES.items():
        name_tokens = set(_re.sub(r"[(),]", " ", name.lower()).split())
        score       = len(expanded & name_tokens)
        if score > best_score:
            best_score, best_code, best_name = score, code, name

    if best_score >= 1 and best_score >= len(tokens) - 1:
        return {"code": best_code, "name": best_name}

    return {"error": f"Language '{user_input}' is not supported."}


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

**CRITICAL LANGUAGE RULE:**
You must conduct this ENTIRE conversation (all your responses, questions, and summaries) in **{DEFAULT_LANGUAGE_NAME}**.

## Your Workflow
### Step 1 — Intake
Greet the user with a translated version of the following message, adapting it naturally to {DEFAULT_LANGUAGE_NAME}:

---
👋 Welcome to the **Gemini Podcast Agent!**

To get started, I'll need a few details:

🎙️ **Speakers** — 1-speaker or 2-speaker podcast?
- *1 speaker* — a single narrator delivers a clear, focused monologue. Great for tutorials, explainers, and training content.
- *2 speakers* — two hosts have a natural back-and-forth conversation. Great for discussions, news briefings, and general interest topics.

🌐 **Language** — What language should the podcast be in? *(default: {DEFAULT_LANGUAGE_NAME})*
- You can use plain language names (e.g. French, German, Japanese), regional variants (e.g. Spanish Mexico, Portuguese Brazil), or BCP-47 codes (e.g. fr-FR, de-DE, es-MX)

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

After the user responds, extract narrator mode, language, sources, target length,
target audience, and any additional context from their reply.

For **language**: if the user provides one, validate it against the supported Gemini TTS
language list. Accept plain names (French, German), regional variants (Spanish Mexico),
or BCP-47 codes (fr-FR, de-DE, es-MX) — resolve them all to a BCP-47 code and display
name. If the language is not supported, inform the user and ask them to choose another.
If the user provides no language, use the configured default: {DEFAULT_LANGUAGE_CODE} ({DEFAULT_LANGUAGE_NAME}).

If anything else is ambiguous or missing, ask a focused follow-up question for just that
item rather than repeating the full intake.

### Step 2 — Confirm
Summarise what you understood and ask the user to confirm before proceeding.
Include the active output mode, narrator mode, and **language** in the summary.
If mode is "gcs", include the GCS bucket path.
Do not move to Step 3 until the user explicitly confirms.

### Step 3 — Production Pipeline
Run `podcast_production_pipeline`. Pass the confirmed values as part of the
state so that all sub-agents can adapt accordingly.
This sequential agent will automatically run the source collector, script
writer, and audio producer in order.
Do NOT invoke script generation or audio production yourself — hand off
entirely to the pipeline.

When handing off to the pipeline, include in the context:
- `narrator_mode`: `"solo"` or `"duo"`
- `language_code`: the resolved BCP-47 code (e.g. `"fr-FR"`, `"de-DE"`)
- `language_name`: the display name (e.g. `"French (France)"`, `"German (Germany)"`)

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
    tools=[podcast_production_pipeline_tool, resolve_language],
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